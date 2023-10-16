from pathlib import Path

import torch
from lightning_fabric.utilities import move_data_to_device
from torch import Tensor

from .dataset.script.dataset import MIDIDataset


class TimeSeqMIDIDataset(MIDIDataset):
    def __init__(
        self,
        dataset_dir: Path,
        f_pickle: Path,
        split: str | list[str],
        seq_len_sec: float,
        intv_size: float,
        device: str = "cpu",
        annot_kinds: str | list[str] | None = None,
    ):
        super().__init__(dataset_dir, f_pickle, split, annot_kinds)
        self.intv_size = intv_size
        self.seq_n_samples = int(seq_len_sec / self.intv_size)
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
        for kind in self.annots:
            if kind in ("beats", "downbeats"):
                beats = time_encode_beats(annots[kind], notes.shape[0])
                annots[kind] = split_and_pad(beats, self.seq_n_samples, dim=0)
            else:
                raise NotImplementedError(f"Unsupported annotation kind: {kind}")
        annots_list = [annots[k] for k in self.annots]
        return note_chunks, annots_list


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


def time_encode_beats(beats: Tensor, note_grid_size: int, grid_len: float = 0.05):
    beats = beats[beats < note_grid_size * grid_len]
    beats_idx = torch.floor(beats / grid_len).long()
    ret = torch.zeros(note_grid_size, device=beats.device)
    ret[beats_idx] = 1
    return ret
