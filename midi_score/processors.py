import abc

import numpy as np
import torch
from torch import Tensor

from .constants import (
    N_per_beat,
    keyNumber2Name,
    min_bpm,
    tolerance,
    tsDenominators,
    tsNumerators,
)
from .models.beat import RNNJointBeatModel
from .models.hand_part import RNNHandPartModel
from .models.key_signature import RNNKeySignatureModel
from .models.quantisation import RNNJointQuantisationModel
from .models.time_signature import RNNTimeSignatureModel


class MIDIProcessor(abc.ABC):
    """
    Abstract base class for processing MIDI data.
    """

    def __init__(self, model, ckpt_path=None):
        self._model = model
        if ckpt_path is not None:
            self.load(ckpt_path)
        self._model.eval()

    def load(self, ckpt_path):
        self._model.load_state_dict(torch.load(ckpt_path))

    @abc.abstractmethod
    def process(self, midi_notes: Tensor):
        pass


class RNNJointBeatProcessor(MIDIProcessor):
    def __init__(self, ckpt_path="_model_state_dicts/beat/RNNJointBeatModel.pth"):
        super().__init__(RNNJointBeatModel(), ckpt_path)

    @torch.no_grad()
    def process(self, notes: Tensor, **kwargs):
        beat_probs, downbeat_probs, _ = self._model(notes.unsqueeze(0))
        onsets = notes[:, 1]
        return self.pps(
            beat_probs.squeeze(0), downbeat_probs.squeeze(0), onsets, **kwargs
        )

    @staticmethod
    def pps(
        beat_probs: Tensor,
        downbeat_probs: Tensor,
        onsets: Tensor,
        prob_thresh=0.5,
        penalty=1.0,
    ):
        """
        Post-processing with dynamic programming.

        Parameters
        ----------
        beat_probs: shape=(n_notes,)
            beat probabilities at each note
        downbeat_probs: shape=(n_notes,)
            downbeat probabilities at each note
        onsets: shape=(n_notes,)
            onset times of each note

        Returns
        -------
        beats: shape=(n_beats,)
            beat times
        """
        n_notes = len(beat_probs)

        # ========== Dynamic thrsholding ==========
        # window length in seconds
        wlen_beats = (60.0 / min_bpm) * 4
        wlen_downbeats = (60.0 / min_bpm) * 8

        # initialize beat and downbeat thresholds
        thresh_beats = torch.ones(n_notes) * prob_thresh
        thresh_downbeats = torch.ones(n_notes) * prob_thresh

        l_b, r_b, l_db, r_db = 0, 0, 0, 0  # sliding window indices

        for i, onset in enumerate(onsets):
            # udpate pointers
            while onsets[l_b] < onset - wlen_beats / 2:
                l_b += 1
            while r_b < n_notes and onsets[r_b] < onset + wlen_beats / 2:
                r_b += 1
            while onsets[l_db] < onset - wlen_downbeats / 2:
                l_db += 1
            while r_db < n_notes and onsets[r_db] < onset + wlen_downbeats / 2:
                r_db += 1
            # update beat and downbeat thresholds
            thresh_beats[i] = beat_probs[l_b:r_b].max() * prob_thresh
            thresh_downbeats[i] = downbeat_probs[l_db:r_db].max() * prob_thresh

        # threshold beat and downbeat probabilities
        beats = onsets[beat_probs > thresh_beats]
        downbeats = onsets[downbeat_probs > thresh_downbeats]

        # ========= Remove in-note-beats that are too close to each other =========

        def get_beats_min(beats_: Tensor):
            one_true = torch.ones(1).bool()
            return beats_[
                torch.cat([one_true, torch.diff(beats_).abs() > tolerance * 2])
            ]

        beats_min = get_beats_min(beats)
        beats_max = get_beats_min(beats.flip(0)).flip(0)
        beats = torch.mean(torch.stack([beats_min, beats_max]), dim=0)
        downbeats_min = get_beats_min(downbeats)
        downbeats_max = get_beats_min(downbeats.flip(0)).flip(0)
        downbeats = torch.mean(torch.stack([downbeats_min, downbeats_max]), dim=0)

        # ========= Merge downbeats to beats if they are not in beat prediction =========
        beats_to_merge = []
        for downbeat in downbeats:
            if torch.abs(beats - downbeat).min() > tolerance * 2:
                beats_to_merge.append(downbeat)
        beats = torch.sort(torch.cat([beats, torch.tensor(beats_to_merge)])).values
        return RNNJointBeatProcessor.beat_complement_dp(beats.numpy(), penalty)

    @staticmethod
    def beat_complement_dp(beats: np.ndarray, penalty=1.0):
        # === dynamic programming for adding out-of-note beats prediction ======
        # minimize objective function:
        #   O = sum(abs(log((t[k] - t[k-1]) / (t[k-1] - t[k-2]))))      (O1)
        #       + lam1 * insertions                                     (O2)
        #       + lam2 * deletions                                      (O3)
        #   t[k] is the kth beat after dynamic programming.
        # ======================================================================

        # ========= fill up out-of-note beats by inter-beat intervals =========
        # fill up by neighboring beats
        wlen = 5  # window length for getting neighboring inter-beat intervals (+- wlen)
        ibis = np.diff(beats)
        beats_filled = []

        for i in range(len(beats) - 1):
            beats_filled.append(beats[i])

            # current and neighboring inter-beat intervals
            ibi = ibis[i]
            ibis_near = ibis[max(0, i - wlen) : min(len(ibis), i + wlen + 1)]
            ibis_near_median = np.median(ibis_near)

            for ratio in [2, 3, 4]:
                if abs(ibi / ibis_near_median - ratio) / ratio < 0.15:
                    for j in range(1, ratio):
                        beats_filled.append(beats[i] + j * ibi / ratio)
        beats = np.sort(np.array(beats_filled))

        # ============= insertions and deletions ======================
        beats_dp: list[list[int]] = [
            [beats[0], beats[1]],  # no insertion
            [beats[0], beats[1]],  # insert one beat
            [beats[0], beats[1]],  # insert two beats
            [beats[0], beats[1]],  # insert three beats
        ]
        obj_dp = [0, 0, 0, 0]
        for i in range(2, len(beats)):
            # insert j beats
            for j in range(4):
                ibi = (beats[i] - beats[i - 1]) / (j + 1)
                objs = []
                for j_prev in range(4):
                    o1 = np.abs(
                        np.log(ibi / (beats_dp[j_prev][-1] - beats_dp[j_prev][-2]))
                    )
                    o = obj_dp[j_prev] + o1 + penalty * j
                    objs.append(o)
                j_prev_best = np.argmin(objs)
                beats_dp[j] = (
                    beats_dp[j_prev_best]
                    + [beats[i - 1] + ibi * k for k in range(1, j + 1)]
                    + [beats[i]]
                )
                obj_dp[j] = objs[j_prev_best]
        return np.array(beats_dp[np.argmin(obj_dp)])


class RNNJointQuantisationProcessor(MIDIProcessor):
    def __init__(
        self, ckpt_path="_model_state_dicts/quantisation/RNNJointQuantisationModel.pth"
    ):
        super().__init__(RNNJointQuantisationModel(), ckpt_path)

    @torch.no_grad()
    def process(self, notes: Tensor):
        # Forward pass
        (
            beat_probs,
            downbeat_probs,
            onset_pos_probs,
            note_value_probs,
        ) = self._model(notes.unsqueeze(0))

        # Post-processing
        onset_pos_idx = onset_pos_probs[0].argmax(dim=0).squeeze(0)  # (n_notes,)
        note_values_idx = note_value_probs[0].argmax(dim=0).squeeze(0)  # (n_notes,)
        beat_probs = beat_probs.squeeze(0)
        downbeat_probs = downbeat_probs.squeeze(0)
        onsets = notes[:, 1]
        return self.pps(
            onset_pos_idx, note_values_idx, beat_probs, downbeat_probs, onsets
        )

    def pps(
        self,
        onset_pos_idx: Tensor,
        note_values_idx: Tensor,
        beat_probs: Tensor,
        downbeat_probs: Tensor,
        onsets: Tensor,
    ):
        # post-processing
        # Convert onset position and note value indexes to actual values
        # Use predicted beat as a reference

        # get beats prediction from beat_probs and downbeat_probs
        beats = RNNJointBeatProcessor.pps(beat_probs, downbeat_probs, onsets)

        # get predicted onset positions and note values in beats
        onset_positions_raw = onset_pos_idx / N_per_beat
        note_values_raw = note_values_idx / N_per_beat

        # Convert onset positions within a beat to absolute positions
        onset_positions = torch.zeros_like(onset_positions_raw)
        note_values = torch.zeros_like(note_values_raw)

        beat_idx = 0
        note_idx = 0

        while note_idx < len(onset_positions_raw):
            if beat_idx < len(beats) and onsets[note_idx] >= beats[beat_idx]:
                beat_idx += 1

            onset_positions[note_idx] = onset_positions_raw[note_idx] + beat_idx
            note_values[note_idx] = note_values_raw[note_idx]
            note_idx += 1

        return beats, onset_positions, note_values


class RNNHandPartProcessor(MIDIProcessor):
    def __init__(self, ckpt_path="_model_state_dicts/hand_part/RNNHandPartModel.pth"):
        super().__init__(RNNHandPartModel(), ckpt_path)

    @torch.no_grad()
    def process(self, notes: Tensor) -> Tensor:
        hand_probs = self._model(notes.unsqueeze(0)).squeeze(0)
        return (hand_probs > 0.5).int()


class RNNKeySignatureProcessor(MIDIProcessor):
    def __init__(
        self, ckpt_path="_model_state_dicts/key_signature/RNNKeySignatureModel.pth"
    ):
        super().__init__(RNNKeySignatureModel(), ckpt_path)

    @torch.no_grad()
    def process(self, notes: Tensor):
        key_probs = self._model(notes.unsqueeze(0))
        key_idx = key_probs[0].argmax(dim=0)  # (seq_len,)
        return self.pps(notes[:, 1].numpy(), key_idx.numpy())

    def pps(self, onsets: np.ndarray, key_idx: np.ndarray):
        ks_changes = []
        for onset, kidx in zip(onsets, key_idx):
            ks_cur = keyNumber2Name[kidx]
            if not ks_changes or ks_changes[-1][1] != ks_cur:
                ks_changes.append((onset, ks_cur))
        return ks_changes


class RNNTimeSignatureProcessor(MIDIProcessor):
    def __init__(
        self, ckpt_path="_model_state_dicts/time_signature/RNNTimeSignatureModel.pth"
    ):
        super().__init__(RNNTimeSignatureModel(), ckpt_path)

    def process(self, notes: Tensor):
        td_probs, tn_probs = self._model(notes.unsqueeze(0))
        tn_idx = tn_probs[0].argmax(dim=0).numpy()  # (seq_len,)
        td_idx = td_probs[0].argmax(dim=0).numpy()  # (seq_len,)
        return self.pps(notes[:, 1].numpy(), tn_idx, td_idx)

    def pps(self, onsets: np.ndarray, tn_idx: np.ndarray, td_idx: np.ndarray):
        ts_changes = []
        for onset, tn, td in zip(onsets, tn_idx, td_idx):
            time_sig = f"{tsNumerators[tn]}/{tsDenominators[td]}"
            if not ts_changes or ts_changes[-1][1] != time_sig:
                ts_changes.append((onset, time_sig))
        return ts_changes
