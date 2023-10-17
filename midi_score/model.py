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
        seq_len_sec: int = 10,
        grid_len: float = 0.02,
        **dataset_kwargs,
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
            "annot_kinds": ["beats"],
            "seq_len_sec": seq_len_sec,
            "grid_len": grid_len,
            **dataset_kwargs
        }
        self._loader_kwargs = {
            "batch_size": batch_size,
            "num_workers": 0,
            "collate_fn": midi_dataset.collate_fn,
        }

    @torch.no_grad()
    def run_on_midi_data(self, midi_data: Tensor) -> Tensor:
        dkw = self._dataset_kwargs
        seq_n_samples = int(dkw["seq_len_sec"] / dkw["intv_size"])
        notes = midi_dataset.time_encode_notes(midi_data, dkw["intv_size"])
        note_chunks = midi_dataset.split_and_pad(notes, seq_n_samples, dim=0)
        activations = self.forward(note_chunks).flatten(0, 1)
        activations = activations[: notes.shape[0]]  # Remove padding
        return activations.detach().cpu()

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
        notes, (gt_beats,) = batch
        pred_probs = self(notes)
        prefix = "train/" if is_train else "val/"
        loss, recall = self._metrics(pred_probs, gt_beats, prefix)
        return loss if is_train else recall

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
        cls_weights = gt_beats.numel() / gt_beats.flatten().bincount() / 3  # 3 classes
        ce_loss = f.cross_entropy(pred_probs.transpose(1, 2), gt_beats, cls_weights)
        gt_beat_ws = cls_weights[gt_beats]
        pred_correct = (pred_probs.argmax(dim=-1) == gt_beats).float()
        weighted_acc = (pred_correct * gt_beat_ws).sum() / gt_beat_ws.sum()
        # Set non-beat predictions to have 0 weight, and the exact same formula
        # computes recall.
        gt_beat_ws[gt_beats == 0] = 0
        recall = (pred_correct * gt_beat_ws).sum() / gt_beat_ws.sum()
        self.log(f"{prefix}loss", ce_loss)
        self.log(f"{prefix}accuracy", weighted_acc)
        self.log(f"{prefix}recall", recall)
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
        self.beat_cls = nn.Linear(d_model, 3)

    def forward(self, x: Tensor) -> Tensor:
        x = self.pos_encoding(self.embedding(x))
        x2 = self.tf_encoder(x)
        return self.beat_cls(x2)
