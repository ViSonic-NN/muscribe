import math
from pathlib import Path

import pytorch_lightning as pl
import torch
from torch import Tensor, nn
from torch.nn import functional as f
from torch.utils.data import DataLoader

from . import midi_dataset
from .midi_dataset import TimeSeqMIDIDataset


class BeatPredictorPL(pl.LightningModule):
    def __init__(
        self,
        dataset_dir: Path | str,
        n_epochs: int,
        lr: float = 0.2,
        batch_size: int = 4,
    ):
        super().__init__()
        self.dataset_dir = Path(dataset_dir)
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.model = BeatTransformer(128, 8, 256, 0.1, 4)
        self.save_hyperparameters()
        self._dataset_kwargs = {
            "f_pickle": self.dataset_dir / "features.pkl",
            "seq_len_sec": 10,
            "intv_size": 0.02,
            "annot_kinds": ["beats", "downbeats"],
        }
        self._loader_kwargs = {
            "batch_size": batch_size,
            "num_workers": 0,
            "collate_fn": midi_dataset.collate_fn,
        }

    def run_on_midi_data(self, midi_data: Tensor) -> Tensor:
        dkw = self._dataset_kwargs
        seq_n_samples = int(dkw["seq_len_sec"] / dkw["intv_size"])
        notes = midi_dataset.time_encode_notes(midi_data, dkw["intv_size"])
        note_chunks = midi_dataset.split_and_pad(notes, seq_n_samples, dim=0)
        activations = self.forward(note_chunks).flatten(0, 1)
        return activations[: notes.shape[0]]  # Remove padding

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def configure_optimizers(self):
        from torch.optim import lr_scheduler as sched

        optim = torch.optim.SGD(self.model.parameters(), self.lr, momentum=0.9)
        steps_per_epoch = len(self.train_dataloader())
        tmax = self.n_epochs * steps_per_epoch
        scheduler = {
            "scheduler": sched.CosineAnnealingLR(optim, T_max=tmax),
            "interval": "step",
        }
        return [optim], [scheduler]

    def _step(self, batch, is_train: bool):
        notes, (gt_beats, gt_downbeats) = batch
        pred_beats, pred_downbeats = self(notes).permute(2, 0, 1)
        prefix = "train" if is_train else "val"
        bloss, brecall = self._metrics(pred_beats, gt_beats, f"{prefix}/beat/")
        dbloss, _ = self._metrics(pred_downbeats, gt_downbeats, f"{prefix}/downbeat/")
        return bloss + dbloss if is_train else brecall

    def training_step(self, batch, _):
        return self._step(batch, is_train=True)

    def validation_step(self, batch, _):
        return self._step(batch, is_train=False)

    def train_dataloader(self):
        self.train_dataset = TimeSeqMIDIDataset(
            self.dataset_dir, split="train", **self._dataset_kwargs
        )
        return DataLoader(self.train_dataset, shuffle=True, **self._loader_kwargs)

    def val_dataloader(self):
        self.val_dataset = TimeSeqMIDIDataset(
            self.dataset_dir, split="validation", **self._dataset_kwargs
        )
        return DataLoader(self.val_dataset, shuffle=False, **self._loader_kwargs)

    def _metrics(self, pred_probs: Tensor, gt_beats: Tensor, prefix: str):
        n_beats, total = gt_beats.nonzero().numel(), gt_beats.numel()
        class0, class1 = n_beats / total, (total - n_beats) / total
        weights = class0 + gt_beats * (class1 - class0)
        ce_loss = f.binary_cross_entropy_with_logits(pred_probs, gt_beats.float(), weights)
        pred_correct = pred_probs.round() == gt_beats
        weighted_acc = (pred_correct.float() * weights).sum() / weights.sum()
        recall = torch.sum(pred_correct * gt_beats) / gt_beats.sum()
        self.log(f"{prefix}accuracy", weighted_acc)
        self.log(f"{prefix}recall", recall)
        self.log(f"{prefix}loss", ce_loss)
        return ce_loss, recall


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[: x.shape[1]].unsqueeze(0)  # type: ignore
        return self.dropout(x)


class BeatTransformer(nn.Module):
    def __init__(self, d_model, nhead, d_hid, dropout, nlayers):
        super(BeatTransformer, self).__init__()
        # d_model is the model's input and output dimensionality
        self.embedding = nn.Linear(128, d_model, bias=False)
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, d_hid, dropout, batch_first=True
        )
        self.tf_encoder = nn.TransformerEncoder(enc_layer, nlayers)
        self.beat_cls = nn.Linear(d_model, 2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.pos_encoding(self.embedding(x))
        x2 = self.tf_encoder(x)
        return torch.sigmoid(self.beat_cls(x2))
