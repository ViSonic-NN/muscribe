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
        device: str | torch.device = "cpu",
    ):
        super().__init__(dataset_dir, f_pickle, split, ["beats"])
        self.dataset = move_data_to_device(self.dataset, device)
        seq_n_samples = int(seq_len_sec / grid_len)
        self.subprocessors = [
            TimeEncoder(
                notes, annots["beats"], annots["downbeats"], grid_len, seq_n_samples
            )
            for notes, annots in self.dataset
        ]
        piece_sizes = [len(x) for x in self.subprocessors]
        self.idx2idx = torch.cumsum(torch.tensor(piece_sizes), dim=0)

    def __len__(self):
        return int(self.idx2idx[-1])

    @torch.no_grad()
    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        data_idx = torch.searchsorted(self.idx2idx, index, right=True)
        assert 0 <= data_idx < len(self.dataset)
        rem = index - (int(self.idx2idx[data_idx - 1]) if data_idx > 0 else 0)
        subp = self.subprocessors[data_idx]
        return subp[rem]


class TimeEncoder:
    """
    Convert `notes` tensor from a MIDI file into time-encoded format,
    and does so lazily -- only when a particular time interval is requested.

    notes: A tensor of shape [n_notes, 4]
           where each row is a tuple of (pitch, onset, duration, velocity)
    grid_len: Time resolution of time encoding
    split_size: Number of time steps in each group
    """

    def __init__(
        self,
        notes: Tensor,
        beats: Tensor,
        downbeats: Tensor,
        grid_len: float,
        split_size: int,
    ) -> None:
        self.grid_len = grid_len
        self.split_size = split_size
        # Decide which grids each note spans.
        # NOTE: if a note extends beyond its group, we just drop the excessive part
        # (and also NOT add it to the next group).
        self.ngroups, self.nstarts, rem_starts = self._group_onsets(notes[:, 1])
        self.nends = torch.ceil((rem_starts + notes[:, 2]) / grid_len).long()
        self.nends.clamp_max_(split_size)
        self.npitches = notes[:, 0].long()
        # itable: [split_size, 1]
        itable = torch.arange(split_size, dtype=torch.long, device=notes.device)
        self.itable = itable.unsqueeze(-1)
        # Process beats and downbeats
        self.bgroups, self.bstarts, _ = self._group_onsets(beats)
        self.dgroups, self.dstarts, _ = self._group_onsets(downbeats)

    @classmethod
    def get_whole_encoding(cls, notes: Tensor, grid_len: float, split_size: int):
        zeros = torch.zeros(0, device=notes.device)
        encoder = cls(notes, zeros, zeros, grid_len, split_size)
        ngroups = len(encoder)
        actual_len = int(encoder.nends[encoder.ngroups == ngroups - 1].max())
        return torch.stack([encoder[i][0] for i in range(ngroups)]), actual_len

    def __len__(self):
        return int(self.ngroups.max() + 1)

    def __getitem__(self, grp_idx: int):
        # Create the encoded output, by first creating a mask for each split group
        # and use the mask to index into the output tensor.
        device = self.nstarts.device
        mask = self.ngroups == grp_idx
        istarts_, iends_ = self.nstarts[mask], self.nends[mask]
        # note_mask: [grid_size, group_n_notes] is a time-wise True-False mask for each note
        note_mask = (istarts_[None, :] <= self.itable) & (self.itable < iends_[None, :])
        # `index_add_` does self[:, index[i]] += src[:, i] when dim == 1
        # (effectively reducing masks along pitch-dimension) on a [grid_size, 128] tensor.
        notes_enc = torch.zeros(self.split_size, 128, device=device)
        notes_enc.index_add_(1, self.npitches[mask], note_mask.float())
        # Don't multi-count notes.
        notes_enc[notes_enc > 1] = 1

        beats_enc = torch.zeros(self.split_size, device=device, dtype=torch.long)
        beats_enc[self.bstarts[self.bgroups == grp_idx]] = 1
        beats_enc[self.dstarts[self.dgroups == grp_idx]] = 2
        return notes_enc, beats_enc

    def _group_onsets(self, onsets: Tensor):
        group_t = self.split_size * self.grid_len
        groups, onsets = onsets // group_t, onsets % group_t
        start_idxs = torch.floor(onsets / self.grid_len).long()
        return groups, start_idxs, onsets


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
