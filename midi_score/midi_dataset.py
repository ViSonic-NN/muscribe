from pathlib import Path

import pretty_midi
import torch
from torch import Tensor
from lightning_fabric.utilities import move_data_to_device
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


def time_encode_notes(notes: Tensor, iv_size: float = 0.05):
    """
    Convert `notes` tensor from a MIDI file into time-encoded format.

    notes: A tensor of shape [n_notes, 4]
           where each row is a tuple of (pitch, onset, duration, velocity)
    interval: Time resolution of time encoding

    Returns: A tensor of shape [intvs, 128]
             where intvs is the number of `interval`-sized intervals
    """

    # Find the maximum end time to determine the length of the encoded tensor
    starts = notes[:, 1]
    ends = starts + notes[:, 2]
    n_intervals = int(torch.ceil(ends.max() / 0.05))
    # Initialize a [intervals x 128] tensor filled with zeros
    encoded = torch.zeros((n_intervals, 128), device=notes.device)
    istarts = torch.floor(starts / iv_size).long()
    iends = torch.ceil(ends / iv_size).long()
    for i in range(istarts.shape[0]):
        encoded[istarts[i] : iends[i], int(notes[i, 0].item())] = 1
    return encoded


def time_encode_beats(beats: Tensor, note_time_len: int, interval: float = 0.05):
    # beats: [n_beats]
    encoding = torch.zeros(note_time_len, device=beats.device, dtype=torch.long)
    beats = torch.floor(beats / interval).long()
    encoding[beats[beats < note_time_len]] = 1
    return encoding


def read_note_sequence(midi_file):
    """
    Load MIDI file into note sequence.

    Parameters
    ----------
    midi_file : str
        Path to MIDI file.

    Returns
    -------
    note_seq: (torch.tensor) in the shape of (n, 4), where n is the number of notes, and 4 is the number of features including (pitch, onset, offset, velocity). The note sequence is sorted by onset time.
    """
    midi_data = pretty_midi.PrettyMIDI(midi_file)
    notes = []
    for instrument in midi_data.instruments:
        notes.extend(instrument.notes)
    notes = sorted(notes, key=lambda x: x.start)
    return torch.tensor(
        [
            [note.pitch, note.start, note.end - note.start, note.velocity]
            for note in notes
        ]
    )
