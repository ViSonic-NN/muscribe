import math
from pathlib import Path

import pytorch_lightning as pl
import torch
from torch import Tensor, nn
from torch.nn import functional as f
from torch.utils.data import DataLoader

from .midi_dataset import TimeSeqMIDIDataset, collate_fn


class BeatPredictorPL(pl.LightningModule):
    def __init__(
        self,
        dataset_dir: Path | str,
        n_epochs: int,
        lr: float = 0.1,
        batch_size: int = 4,
    ):
        super().__init__()
        self.dataset_dir = Path(dataset_dir)
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.model = BeatTransformer(128, 8, 256, 0, 4)
        self.save_hyperparameters()
        self._dataset_kwargs = {
            "f_pickle": self.dataset_dir / "features.pkl",
            "seq_len_sec": 30,
            "intv_size": 0.1,
            "annot_kinds": ["beats"],
        }
        self._loader_kwargs = {
            "batch_size": batch_size,
            "num_workers": 0,
            "collate_fn": collate_fn,
        }

    def forward(self, x: Tensor):
        return self.model(x).squeeze(-1)

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

    def training_step(self, batch, _):
        notes, (beats,) = batch
        beats_pred = self(notes)
        loss = self.ce_loss(beats_pred, beats)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        notes, (beats,) = batch
        beats_pred = self(notes)
        loss = self.ce_loss(beats_pred, beats)
        accuracy = self.accuracy(beats_pred, beats)
        self.log("val/loss", loss)
        self.log("val/accuracy", accuracy, prog_bar=True)
        return accuracy

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

    def accuracy(self, beats_pred, beats: Tensor):
        n_beats, total = beats.nonzero().numel(), beats.numel()
        class0, class1 = n_beats / total, (total - n_beats) / total
        pred_correct = beats_pred.round() == beats
        weights = class0 + beats * (class1 - class0)
        return (pred_correct.float() * weights).sum() / weights.sum()

    def ce_loss(self, beats_pred, beats: Tensor):
        n_beats, total = beats.nonzero().numel(), beats.numel()
        class0, class1 = n_beats / total, (total - n_beats) / total
        weights = class0 + beats * (class1 - class0)
        return f.binary_cross_entropy(beats_pred, beats.float(), weights)


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
        x = x + self.pe[:x.shape[1]].unsqueeze(0)  # type: ignore
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
        self.beats_head = nn.Linear(d_model, 1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.pos_encoding(self.embedding(x))
        x2 = self.tf_encoder(x)
        beats = torch.sigmoid(self.beats_head(x2))
        return beats
