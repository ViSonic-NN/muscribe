from pathlib import Path
import math
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
    
class TimeSeqMIDIDatasetWithNoteEmbedding(MIDIDataset):
    def __init__(
        self,
        dataset_dir: Path,
        f_pickle: Path,
        split: str | list[str],
        seq_len_sec: float,
        intv_size: float,
        device: str = "cpu",
        d_model: int = 128,
        annot_kinds: str | list[str] | None = None
        
    ):
        super().__init__(dataset_dir, f_pickle, split, annot_kinds)
        self.intv_size = intv_size
        self.seq_n_samples = int(seq_len_sec / self.intv_size)
        self.device = device
        self.dataset = move_data_to_device(self.dataset, self.device)
        self.d_model = d_model
        
    @torch.no_grad()
    def __getitem__(self, index) -> tuple[Tensor, list[Tensor]]:
        notes, annots = self.augments(self.dataset[index])
        annots: dict[str, Tensor]
        # Shallow copy, needed when self.augments is trivial
        # because otherwise `annots` is modified below, it goes directly to the dataset
        # self.dropout = nn.Dropout(p=dropout)
        max_time = torch.max(annots['beats'])
        notes = notes[notes[:,1] <= max_time]
        times = torch.floor(notes[:,1]/self.intv_size).unsqueeze(1)
        durations = torch.floor(notes[:,2]/self.intv_size).unsqueeze(1)
        max_len = notes.shape[0]
        div_term = torch.exp(
                torch.arange(0, self.d_model, 2) * (-math.log(10000.0) / self.d_model)
            )
        pe = torch.zeros(max_len, self.d_model)
        de = torch.zeros(max_len, self.d_model)
        pe[:, 0::2] = torch.sin(times * div_term)
        pe[:, 1::2] = torch.cos(times * div_term)
        de[:, 0::2] = torch.sin(durations * div_term)
        de[:, 1::2] = torch.cos(durations * div_term)
        annots = annots.copy()
        note_encoded = encode_notes(notes)
        note_encoded = note_encoded + pe[:note_encoded.shape[0]] # type: ignore
        note_encoded = note_encoded + de[:note_encoded.shape[0]]  # type: ignore
        note_chunks = split_and_pad(note_encoded, self.seq_n_samples, dim=0)
        for kind in self.annots:
            if kind in ("beats", "downbeats"):
                beats = note_encode_beats(annots[kind], notes)
                annots[kind] = split_and_pad(beats, self.seq_n_samples, dim=0)
            else:
                raise NotImplementedError(f"Unsupported annotation kind: {kind}")
        annots_list = [annots[k] for k in self.annots]
        return note_chunks, annots_list
    


        

def encode_notes(notes):
    encoded = torch.zeros(notes.shape[0], 128)
    for i, x in enumerate(notes):
        encoded[i,int(x[0])] = 1
    return encoded

def note_encode_beats(beats, notes):
    encoded = torch.zeros(notes.shape[0])
    rounded_beats = [ '%.2f' % elem for elem in beats]
    all_beats = set(rounded_beats)
    rounded_notes = torch.round(notes, decimals= 3)
    for i, x in enumerate(rounded_notes):
        if ('%.2f' % x[1] in all_beats):
            encoded[i] = 1
    return encoded





def position_encoding_notes(notes):
    times = torch.floor(notes[:,1].unsqueeze(1)/0.05)
    max_len = notes.shape[0]
    div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(times * div_term)
    pe[:, 1::2] = torch.cos(times * div_term)



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
