import abc

import numpy as np
import torch
from torch import Tensor

from ..constants import NATURAL_KEY_ORDER
from .models import RNNHandPartModel, RNNKeySignatureModel


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
            onset: float
            kidx: int
            ks_cur = NATURAL_KEY_ORDER[kidx]
            if not ks_changes or ks_changes[-1][1] != ks_cur:
                ks_changes.append((onset, ks_cur))
        return ks_changes
