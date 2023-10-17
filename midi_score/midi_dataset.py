import logging
from pathlib import Path

import numpy as np
import torch
from lightning_fabric.utilities import move_data_to_device
from torch import Tensor, nn
from torch.utils.data import Dataset

from .dataset.script import dataset as datautils

logger = logging.getLogger(__name__)
BEAT_EPS = 0.02  # tolerance for beat alignment: 0.02s = 20ms


class MIDIDataset(Dataset):
    def __init__(
        self,
        dataset_dir: Path,
        feature_pickle: Path,
        split: str | list[str],
        annot_kinds: str | list[str] | None = None,
    ):
        meta, dataset = datautils.read_full_dataset(dataset_dir, feature_pickle)
        assert len(meta) == len(dataset)

        self.split = split
        if annot_kinds is None:
            self.annots = datautils.ANNOT_KEYS  # Request all features
        elif isinstance(annot_kinds, str):
            self.annots = [annot_kinds]
        else:
            self.annots = annot_kinds
        selection = meta["split"] == split
        if datautils.NO_ASAP_KEYS.intersection(self.annots):
            logger.warning(
                "Requesting annotations that are not available in ASAP dataset; skipping ASAP dataset."
            )
            selection &= meta["source"] != "ASAP"
        selection = np.nonzero(selection)[0]
        self.metadata = meta.iloc[selection].reset_index()
        # int() is only for type checking
        self.dataset = [dataset[int(i)] for i in selection]
        self.augments = nn.Sequential()

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index) -> tuple[Tensor, list[Tensor]]:
        notes, annots = self.augments(self.dataset[index])
        annots = [annots[k] for k in self.annots]
        return notes, annots


class TimeSeqMIDIDataset(MIDIDataset):
    def __init__(
        self,
        dataset_dir: Path,
        f_pickle: Path,
        split: str | list[str],
        seq_len_sec: float,
        grid_len: float,
        device: str = "cpu",
        annot_kinds: str | list[str] | None = None,
    ):
        if annot_kinds is None:
            annot_kinds = datautils.ANNOT_KEYS.copy()
            annot_kinds.remove("downbeats")
        elif "downbeats" in annot_kinds:
            raise ValueError("Downbeats are provided with beats in TimeSeqMIDIDataset")
        super().__init__(dataset_dir, f_pickle, split, annot_kinds)
        self.grid_len = grid_len
        self.seq_n_samples = int(seq_len_sec / self.grid_len)
        self.device = device
        self.dataset = move_data_to_device(self.dataset, self.device)

    @torch.no_grad()
    def __getitem__(self, index) -> tuple[Tensor, list[Tensor]]:
        notes, annots = self.augments(self.dataset[index])
        annots: dict[str, Tensor]
        # Shallow copy, needed when self.augments is trivial
        # because otherwise `annots` is modified below, it goes directly to the dataset
        annots = annots.copy()
        notes = time_encode_notes(notes)
        note_chunks = split_and_pad(notes, self.seq_n_samples, dim=0)
        ret_annots = []
        for kind in self.annots:
            if kind != "beats":
                raise NotImplementedError(f"Unsupported annotation kind: {kind}")
            beats, downbeats = annots["beats"], annots["downbeats"]
            beats = time_encode_beats(beats, downbeats, notes.shape[0])
            beats = split_and_pad(beats, self.seq_n_samples, dim=0)
            ret_annots.append(beats)
        return note_chunks, ret_annots


def collate_fn(batch):
    # use concat instead of stack
    notes, annots = zip(*batch)
    notes = torch.cat(notes, dim=0)
    annots = [torch.cat(a, dim=0) for a in zip(*annots)]
    return notes, annots


def split_and_pad(xs: Tensor, group_size: int, dim: int = 0, fill_value: float = 0.0):
    splitted = list(torch.split(xs, group_size, dim=dim))
    assert len(splitted) >= 1
    last = splitted[-1]
    if (last_len := last.shape[dim]) < group_size:
        shape = list(last.shape)
        shape[dim] = group_size - last_len
        filler = torch.full(shape, fill_value, dtype=last.dtype, device=last.device)
        splitted[-1] = torch.concat([last, filler], dim=dim)
        assert splitted[-1].shape[dim] == group_size
    return torch.stack(splitted, dim=dim)


def time_encode_notes(notes: Tensor, grid_len: float = 0.05):
    """
    Convert `notes` tensor from a MIDI file into time-encoded format.

    notes: A tensor of shape [n_notes, 4]
           where each row is a tuple of (pitch, onset, duration, velocity)
    grid_len: Time resolution of time encoding

    Returns: A tensor of shape [grid_size, 128]
             where grid_size is the number of `grid_len`-sized intervals
    """

    # Find the maximum end time to determine the length of the encoded tensor
    starts = notes[:, 1]
    ends = starts + notes[:, 2]
    grid_size = int(torch.ceil(ends.max() / grid_len))
    # Initialize a [intervals x 128] tensor filled with zeros
    encoded = torch.zeros((grid_size, 128), device=notes.device)
    istarts = torch.floor(starts / grid_len).long()
    iends = torch.ceil(ends / grid_len).long()
    for i in range(istarts.shape[0]):
        pitch = int(notes[i, 0].item())
        encoded[istarts[i] : iends[i], pitch] = 1
    return encoded


def time_encode_beats(
    beats: Tensor, downbeats: Tensor, note_grid_size: int, grid_len: float = 0.05
):
    beats_idx = torch.floor(beats / grid_len).long()
    beats_idx = beats_idx[beats_idx < note_grid_size]
    dbeats_idx = torch.floor(downbeats / grid_len).long()
    dbeats_idx = dbeats_idx[dbeats_idx < note_grid_size]
    ret = torch.zeros(note_grid_size, device=beats.device, dtype=torch.long)
    ret[beats_idx] = 1
    ret[dbeats_idx] = 2
    return ret


class RandomTempoChange(nn.Module):
    def __init__(
        self, prob: float = 0.5, min_ratio: float = 0.8, max_ratio: float = 1.2
    ):
        super().__init__()
        self.prob = prob
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def forward(self, inputs):
        notes, annots = inputs
        change, ratio = torch.rand(2).tolist()
        if change > self.prob:
            return notes, annots
        ratio = self.min_ratio + (self.max_ratio - self.min_ratio) * ratio
        inv_ratio = 1 / ratio
        notes[:, 1:3] *= inv_ratio
        annots["beats"] *= inv_ratio
        annots["downbeats"] *= inv_ratio
        annots["time_signatures"][:, 0] *= inv_ratio
        annots["key_signatures"][:, 0] *= inv_ratio
        return notes, annots


class RandomPitchShift(nn.Module):
    def __init__(self, prob: float = 0.5, min_shift: int = -12, max_shift: int = 12):
        super().__init__()
        self.prob = prob
        self.min_shift = min_shift
        self.max_shift = max_shift

    def forward(self, inputs):
        notes, annots = inputs
        change = torch.rand(1).item()
        if change > self.prob:
            return notes, annots
        shift = torch.randint(self.min_shift, self.max_shift, (1,)).item()
        notes[:, 0] += shift
        keysig = annots["key_signatures"]
        keysig[:, 1] = (keysig[:, 1] + shift) % 24
        return notes, annots


class RandomAddRemoveNotes(nn.Module):
    def __init__(
        self,
        add_note_p: float = 0.5,
        add_ratio: float = 0.2,
        remove_note_p: float = 0.5,
        remove_ratio: float = 0.2,
    ):
        super().__init__()
        if add_note_p + remove_note_p > 1.0:
            total = add_note_p + remove_note_p
            logger.warning(
                "extra_note_prob (%.3f) + missing_note_p (%.3f) > 1; reset to %.3f, %.3f respectively",
                add_note_p,
                remove_note_p,
                (add_note_p := add_note_p / total),
                (remove_note_p := remove_note_p / total),
            )
        self.add_note_p = add_note_p
        self.add_ratio = add_ratio
        self.remove_note_p = remove_note_p
        self.remove_ratio = remove_ratio

    def forward(self, inputs):
        notes, annots = inputs
        rand = torch.rand(1).item()
        if rand < self.add_note_p:
            notes, annots = self.add_notes(notes, annots)
        elif rand > 1.0 - self.remove_note_p:
            notes, annots = self.remove_notes(notes, annots)
        return notes, annots

    def add_notes(self, notes, annots):
        new_seq = torch.repeat_interleave(notes, 2, dim=0)
        # pitch shift for extra notes (+-12)
        shift = torch.randint(-12, 12 + 1, (len(new_seq),))
        shift[::2] = 0
        new_seq[:, 0] += shift
        new_seq[:, 0][new_seq[:, 0] < 0] += 12
        new_seq[:, 0][new_seq[:, 0] > 127] -= 12

        # keep a random ratio of extra notes
        probs = torch.rand(len(new_seq))
        probs[::2] = 0.0
        kept = probs < self.add_ratio
        new_seq = new_seq[kept]
        for key in ["onsets_musical", "note_value", "hands"]:
            if annots[key] is not None:
                annots[key] = torch.repeat_interleave(annots[key], 2, dim=0)[kept]
        if annots["hands"] is not None:
            # mask out hand for extra notes during training
            hands_mask = torch.ones(len(notes) * 2)
            hands_mask[1::2] = 0
            annots["hands_mask"] = hands_mask[kept]
        return new_seq, annots

    def remove_notes(self, notes, annots):
        # find successive concurrent notes
        candidates = torch.diff(notes[:, 1]) < BEAT_EPS
        # randomly select a ratio of candidates to be removed
        candidates_probs = candidates * torch.rand(len(candidates))
        kept = torch.cat(
            [torch.tensor([True]), candidates_probs < (1 - self.remove_ratio)]
        )
        # remove selected candidates
        notes = notes[kept]
        for key in ["onsets_musical", "note_value", "hands"]:
            if annots[key] is not None:
                annots[key] = annots[key][kept]
        return notes, annots
