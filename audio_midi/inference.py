import os
from pathlib import Path

import numpy as np
import torch

from . import config, models
from .pytorch_utils import forward
from .utilities import RegressionPostProcessor, write_events_to_midi


class PianoTranscription(object):
    def __init__(
        self,
        model_type: str = "NotePedal",
        ckpt_path: Path | None = None,
        segment_samples: int = 16000 * 10,
        device: str = "cuda",
    ):
        """Class for transcribing piano solo recording."""
        if ckpt_path is None:
            ckpt_path = (
                Path(torch.hub.get_dir())
                / "piano_tr"
                / "note_F1=0.9677_pedal_F1=0.9186.pth"
            )
        print(f"Checkpoint path: {ckpt_path}")
        if not ckpt_path.is_file():
            os.makedirs(ckpt_path.parent, exist_ok=True)
            zenodo_path = "https://zenodo.org/record/4034264/files/CRNN_note_F1%3D0.9677_pedal_F1%3D0.9186.pth?download=1"
            torch.hub.download_url_to_file(zenodo_path, ckpt_path, progress=True)

        print(f"Using {device} for inference")
        self.segment_samples = segment_samples
        self.frames_per_second = config.frames_per_second
        self.classes_num = config.classes_num
        self.onset_threshold = 0.3
        self.offset_threshod = 0.3
        self.frame_threshold = 0.1
        self.pedal_offset_threshold = 0.2

        # Build model
        self.model = getattr(models, model_type)(
            frames_per_second=self.frames_per_second, classes_num=self.classes_num
        )

        # Load model
        checkpoint = torch.load(ckpt_path, map_location=device)
        self.model.load_state_dict(checkpoint["model"], strict=False)

        # Parallel
        if "cuda" in str(device):
            self.model.to(device)
            print("GPU number: {}".format(torch.cuda.device_count()))
            self.model = torch.nn.DataParallel(self.model)
        else:
            print("Using CPU.")

    def transcribe(self, audio, midi_path):
        """Transcribe an audio recording.

        Args:
          audio: (audio_samples,)
          midi_path: str, path to write out the transcribed MIDI.

        Returns:
          transcribed_dict, dict: {'output_dict':, ..., 'est_note_events': ...}

        """
        audio = audio[None, :]  # (1, audio_samples)

        # Pad audio to be evenly divided by segment_samples
        audio_len = audio.shape[1]
        pad_len = (
            int(np.ceil(audio_len / self.segment_samples)) * self.segment_samples
            - audio_len
        )

        audio = np.concatenate((audio, np.zeros((1, pad_len))), axis=1)

        # Enframe to segments
        segments = self.enframe(audio, self.segment_samples)
        """(N, segment_samples)"""

        # Forward
        output_dict = forward(self.model, segments, batch_size=1)
        """{'reg_onset_output': (N, segment_frames, classes_num), ...}"""

        # Deframe to original length
        for key in output_dict.keys():
            output_dict[key] = self.deframe(output_dict[key])[0:audio_len]
        """output_dict: {
          'reg_onset_output': (N, segment_frames, classes_num), 
          'reg_offset_output': (N, segment_frames, classes_num), 
          'frame_output': (N, segment_frames, classes_num), 
          'velocity_output': (N, segment_frames, classes_num)}"""

        # Post processor
        post_processor = RegressionPostProcessor(
            self.frames_per_second,
            classes_num=self.classes_num,
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
            print("Write out to {}".format(midi_path))

        transcribed_dict = {
            "output_dict": output_dict,
            "est_note_events": est_note_events,
            "est_pedal_events": est_pedal_events,
        }

        return transcribed_dict

    def enframe(self, x, segment_samples):
        """Enframe long sequence to short segments.

        Args:
          x: (1, audio_samples)
          segment_samples: int

        Returns:
          batch: (N, segment_samples)
        """
        assert x.shape[1] % segment_samples == 0
        batch = []

        pointer = 0
        while pointer + segment_samples <= x.shape[1]:
            batch.append(x[:, pointer : pointer + segment_samples])
            pointer += segment_samples // 2

        batch = np.concatenate(batch, axis=0)
        return batch

    def deframe(self, x):
        """Deframe predicted segments to original sequence.

        Args:
          x: (N, segment_frames, classes_num)

        Returns:
          y: (audio_frames, classes_num)
        """
        if x.shape[0] == 1:
            return x[0]

        else:
            x = x[:, 0:-1, :]
            """Remove an extra frame in the end of each segment caused by the
            'center=True' argument when calculating spectrogram."""
            (N, segment_samples, classes_num) = x.shape
            assert segment_samples % 4 == 0

            y = []
            y.append(x[0, 0 : int(segment_samples * 0.75)])
            for i in range(1, N - 1):
                y.append(
                    x[i, int(segment_samples * 0.25) : int(segment_samples * 0.75)]
                )
            y.append(x[-1, int(segment_samples * 0.25) :])
            y = np.concatenate(y, axis=0)
            return y
