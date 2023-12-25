import logging
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np
import torch
from tqdm import trange

from .. import params
from . import models
from .midi_writer import RegressionPostProcessor, write_events_to_midi

logger = logging.getLogger(__name__)


class PianoTranscription(object):
    def __init__(
        self,
        model_type: str = "NotePedal",
        ckpt_path: Path | None = None,
        device: str = "cuda",
        infer_batch: int = 1,
    ):
        """Class for transcribing piano solo recording."""
        if ckpt_path is None:
            ckpt_path = (
                Path(torch.hub.get_dir())
                / "piano_tr"
                / "note_F1=0.9677_pedal_F1=0.9186.pth"
            )
        logger.info("Checkpoint path: %s", ckpt_path)
        if not ckpt_path.is_file():
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            zenodo_path = "https://zenodo.org/record/4034264/files/CRNN_note_F1%3D0.9677_pedal_F1%3D0.9186.pth?download=1"
            torch.hub.download_url_to_file(
                zenodo_path, ckpt_path.as_posix(), progress=True
            )

        self.infer_batch = infer_batch
        self.onset_threshold = 0.3
        self.offset_threshod = 0.3
        self.frame_threshold = 0.1
        self.pedal_offset_threshold = 0.2

        # Build model
        self.model = getattr(models, model_type)(
            frames_per_second=params.frames_per_second, classes_num=params.classes_num
        )
        # Load model
        checkpoint = torch.load(ckpt_path, map_location=device)
        self.model.load_state_dict(checkpoint["model"])

        # Parallel
        logger.info("Using %s for inference", device)
        if "cuda" in str(device):
            # self.model.to(device)
            logger.info("GPU number: %d", torch.cuda.device_count())
            self.model = self.model.to(device)
        else:
            logger.info("Using CPU.")

    @property
    def segment_samples(self):
        return int(params.sample_rate * params.segment_seconds)

    def transcribe(self, audio: np.ndarray | str | Path, midi_path):
        """Transcribe an audio recording.

        Args:
          audio: (audio_samples,)
          midi_path: str, path to write out the transcribed MIDI.

        Returns:
          transcribed_dict, dict: {'output_dict':, ..., 'est_note_events': ...}

        """
        if isinstance(audio, (str, Path)):
            audio, _ = librosa.load(audio, sr=params.sample_rate, mono=True)
        audio = audio[None, :]  # (1, audio_samples)

        # Enframe to segments
        audio_len = audio.shape[1]
        segments = self.enframe(audio)
        # Forward
        output_dict = self.batched_forward(segments, batch_size=self.infer_batch)
        # Deframe to original length
        for key in output_dict.keys():
            output_dict[key] = self.deframe(output_dict[key])[0:audio_len]
        # output_dict: {
        #   'reg_onset_output': (N, segment_frames, classes_num),
        #   'reg_offset_output': (N, segment_frames, classes_num),
        #   'frame_output': (N, segment_frames, classes_num),
        #   'velocity_output': (N, segment_frames, classes_num)}

        # Post processor
        post_processor = RegressionPostProcessor(
            params.frames_per_second,
            classes_num=params.classes_num,
            onset_threshold=self.onset_threshold,
            offset_threshold=self.offset_threshod,
            frame_threshold=self.frame_threshold,
            pedal_offset_threshold=self.pedal_offset_threshold,
        )

        # Post process output_dict to MIDI events
        (est_note_events, est_pedal_events) = post_processor.output_dict_to_midi_events(
            output_dict
        )

        # Write MIDI events to file
        if midi_path:
            write_events_to_midi(
                start_time=0,
                note_events=est_note_events,
                pedal_events=est_pedal_events,
                midi_path=midi_path,
            )
            logger.info("Write out to %s", midi_path)

        return {
            "output_dict": output_dict,
            "est_note_events": est_note_events,
            "est_pedal_events": est_pedal_events,
        }

    def enframe(self, audio):
        """Enframe long sequence to short segments.

        Args:
          x: (1, audio_samples)

        Returns:
          batch: (N, segment_samples)
        """
        # Pad audio to be evenly divided by segment_samples
        audio_len = audio.shape[1]
        ss = self.segment_samples
        pad_len = int(np.ceil(audio_len / ss)) * ss - audio_len
        audio = np.concatenate((audio, np.zeros((1, pad_len))), axis=1)
        assert audio.shape[1] % self.segment_samples == 0

        batch = []
        pointer = 0
        while pointer + ss <= audio.shape[1]:
            batch.append(audio[:, pointer : pointer + ss])
            pointer += ss // 2
        return np.concatenate(batch, axis=0)

    def deframe(self, x):
        """Deframe predicted segments to original sequence.

        Args:
          x: (N, segment_frames, classes_num)

        Returns:
          y: (audio_frames, classes_num)
        """
        if x.shape[0] == 1:
            return x[0]
        x = x[:, 0:-1, :]
        """Remove an extra frame in the end of each segment caused by the
        'center=True' argument when calculating spectrogram."""
        (N, segment_samples, _) = x.shape
        assert segment_samples % 4 == 0

        y = []
        y.append(x[0, 0 : int(segment_samples * 0.75)])
        for i in range(1, N - 1):
            y.append(x[i, int(segment_samples * 0.25) : int(segment_samples * 0.75)])
        y.append(x[-1, int(segment_samples * 0.25) :])
        y = np.concatenate(y, axis=0)
        return y

    @torch.no_grad()
    def batched_forward(self, x: np.ndarray, batch_size: int) -> dict[str, np.ndarray]:
        """Forward data to model in mini-batch.

        Returns:
        output_dict: dict, e.g. {
            'frame_output': (segments_num, frames_num, classes_num),
            'onset_output': (segments_num, frames_num, classes_num),
            ...}
        """

        output_dict = defaultdict(list)
        device = next(self.model.parameters()).device
        self.model.eval()
        xt = torch.from_numpy(x).float()
        for begin in trange(0, len(x), batch_size):
            batch = xt[begin : begin + batch_size].to(device)
            batch_output_dict = self.model(batch)
            for key in batch_output_dict.keys():
                output_dict[key].append(batch_output_dict[key].cpu())
        return {
            key: torch.cat(output_dict[key], dim=0).numpy()
            for key in output_dict.keys()
        }
