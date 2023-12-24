import logging
import math
from pathlib import Path

import pytorch_lightning as pl
import torch
from torch import Tensor, nn
from torch.nn import functional as f
from torch.utils.data import DataLoader

from . import midi_dataset
from .midi_dataset import TimeSeqMIDIDataset

logger = logging.getLogger(__name__)


class BeatPredictorPL(pl.LightningModule):
    def __init__(
        self,
        dataset_dir: Path | str,
        lr: float = 1e-3,
        batch_size: int = 64,
        seq_len_sec: int = 30,
        grid_len: float = 0.02,
        **dataset_kwargs,
    ):
        super().__init__()
        self.dataset_dir = Path(dataset_dir)
        self.lr = lr
        self.batch_size = batch_size
        self.model = BeatCRNN(32, 256, 2, 0.1)
        self.save_hyperparameters()
        self.grid_len = grid_len
        self.seq_n_samples = int(seq_len_sec / self.grid_len)
        self._dataset_kwargs = {
            "f_pickle": self.dataset_dir / "features.pkl",
            "seq_len_sec": seq_len_sec,
            "grid_len": grid_len,
            **dataset_kwargs,
        }
        self._loader_kwargs: dict = {"batch_size": batch_size, "num_workers": 0}

    @torch.no_grad()
    def run_on_midi_data(self, midi_data: Tensor) -> Tensor:
        note_grid, max_len = midi_dataset.TimeEncoder.get_whole_encoding(
            midi_data, self.grid_len, self.seq_n_samples
        )
        activs = torch.softmax(self.forward(note_grid).flatten(0, 1), dim=-1)
        # Remove padding and non-beat-class probs
        return activs[:max_len].detach().cpu()

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def configure_optimizers(self):
        from torch.optim import lr_scheduler as sched

        optim = torch.optim.AdamW(self.model.parameters(), self.lr, weight_decay=0.01)
        scheduler = {
            "scheduler": sched.CosineAnnealingLR(optim, T_max=self.num_training_steps),
            "interval": "step",
        }
        # scheduler = sched.ReduceLROnPlateau(optim, patience=10, factor=0.25, min_lr=1e-6)
        # return {"optimizer": optim, "lr_scheduler": scheduler, "monitor": "val/recall"}
        return [optim], [scheduler]

    def training_step(self, batch, _):
        notes, gt_beats = batch
        pred_probs = self(notes)
        return self._metrics(pred_probs, gt_beats, True)

    def validation_step(self, batch, _):
        notes, gt_beats = batch
        pred_probs = self(notes)
        return self._metrics(pred_probs, gt_beats, False)

    def train_dataloader(self):
        device = self.device
        train_dataset = TimeSeqMIDIDataset(
            self.dataset_dir, split="train", **self._dataset_kwargs, device=device
        )
        logger.info("Train dataset size: %d", len(train_dataset))
        return DataLoader(train_dataset, shuffle=True, **self._loader_kwargs)

    def val_dataloader(self):
        device = self.device
        val_dataset = TimeSeqMIDIDataset(
            self.dataset_dir, split="validation", **self._dataset_kwargs, device=device
        )
        logger.info("Val dataset size: %d", len(val_dataset))
        return DataLoader(val_dataset, shuffle=False, **self._loader_kwargs)

    def _metrics(self, pred_probs: Tensor, gt_beats: Tensor, is_train: bool):
        cls_weights = torch.tensor([1, 10, 20]).float().cuda()
        ce_loss = f.cross_entropy(pred_probs.transpose(1, 2), gt_beats, cls_weights)

        predicted = pred_probs.argmax(dim=-1)
        gt_beats_, pred_beats_ = gt_beats != 0, predicted != 0
        b_recall = (gt_beats_ & pred_beats_).sum() / gt_beats_.sum()
        b_prec = (gt_beats_ & pred_beats_).sum() / pred_beats_.sum()
        b_f1 = 2 * b_prec * b_recall / (b_prec + b_recall)
        downbeats = gt_beats == 2
        db_recall = (downbeats & (predicted == 2)).sum() / downbeats.sum()

        prefix = "train/" if is_train else "val/"
        sync_dist = not is_train
        self.log(f"{prefix}loss", ce_loss, sync_dist=sync_dist, prog_bar=is_train)
        self.log(f"{prefix}b_prec", b_prec, sync_dist=sync_dist)
        self.log(f"{prefix}b_recall", b_recall, sync_dist=sync_dist)
        self.log(f"{prefix}b_f1", b_f1, sync_dist=sync_dist)
        self.log(f"{prefix}db_recall", db_recall, sync_dist=sync_dist)
        return ce_loss if is_train else b_recall

    @property
    def example_input_array(self):
        return torch.rand(1, self.seq_n_samples, 128)

    @property
    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        if self.trainer.max_steps > 0:
            return self.trainer.max_steps
        assert (max_epochs := self.trainer.max_epochs) is not None
        limit_batches = self.trainer.limit_train_batches
        batches = len(self.train_dataloader())
        if isinstance(limit_batches, int):
            batches = min(batches, limit_batches)
        else:
            batches = int(limit_batches * batches)
        num_devices = self.trainer.num_devices
        effective_accum = self.trainer.accumulate_grad_batches * num_devices
        return (batches // effective_accum) * max_epochs


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
        x = x + self.pe[: x.shape[1]].unsqueeze(0)
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


class BeatCRNN(nn.Module):
    def __init__(self, d_model: int, d_hid: int, n_rnn_layers: int, dropout: float):
        super().__init__()
        self.embedding = nn.Linear(128, d_model)
        self.convs = nn.Sequential(
            *self._make_block(d_model, d_hid // 4, 9, dropout),
            *self._make_block(d_hid // 4, d_hid // 2, 9, dropout),
            *self._make_block(d_hid // 2, d_hid, 9, dropout),
        )
        self.gru = nn.LSTM(
            input_size=d_hid,
            hidden_size=d_hid // 2,
            num_layers=n_rnn_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )
        self.beat_cls = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(d_hid, 3))

    def forward(self, x: Tensor):
        # x: [batch_size, seq_len, 128]
        x = self.embedding(x)
        # x: [batch_size, seq_len, d_model]
        x = self.convs(x.transpose(1, 2)).transpose(1, 2)
        # x: [batch_size, seq_len, d_hid]
        gru_beat, _ = self.gru(x)
        # gru_beat: [batch_size, seq_len, d_hid]
        return self.beat_cls(gru_beat)
        # return: [batch_size, seq_len, 3]

    @staticmethod
    def _make_block(in_feats: int, out_feats: int, kernel_size: int, dropout: float):
        return [
            nn.Conv1d(
                in_channels=in_feats,
                out_channels=in_feats,
                groups=in_feats,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.BatchNorm1d(in_feats),
            nn.ReLU6(),
            nn.Conv1d(
                in_channels=in_feats,
                out_channels=out_feats,
                kernel_size=1,
                padding=0,
            ),
            nn.BatchNorm1d(out_feats),
            nn.ReLU6(),
            nn.Dropout(p=dropout),
        ]
