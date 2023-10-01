import math
from pathlib import Path

import pytorch_lightning as pl
import torch
from torch import Tensor, nn
from torch.nn import functional as f
from torch.utils.data import DataLoader

from .midi_dataset import TimeSeqMIDIDataset, collate_fn


class BeatPredictorPL(pl.LightningModule):
    def __init__(self, dataset_dir: Path | str, lr: float = 0.01, batch_size: int = 4):
        super().__init__()
        self.dataset_dir = Path(dataset_dir)
        self.model = BeatTransformer(128, 8, 256, 0.1, 4)
        self.lr = lr
        self.batch_size = batch_size
        self.save_hyperparameters()
        self._dataset_kwargs = {
            "f_pickle": self.dataset_dir / "features.pkl",
            "seq_len_sec": 10,
            "intv_size": 0.05,
            "annot_kinds": ["beats"],
        }
        self._loader_kwargs = {
            "batch_size": batch_size,
            "num_workers": 0,
            "collate_fn": collate_fn,
        }

    def forward(self, x: Tensor):
        return self.model(x).transpose(1, 2)

    def configure_optimizers(self):
        return torch.optim.SGD(self.model.parameters(), self.lr, 0.9)

    def training_step(self, batch, _):
        notes, (beats,) = batch
        beats_pred = self(notes)
        loss = f.cross_entropy(beats_pred, beats)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        notes, (beats,) = batch
        beats_pred = self(notes)
        loss = f.cross_entropy(beats_pred, beats)
        accuracy = torch.sum(beats_pred.argmax(dim=1) == beats) / beats.numel()
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
            self.dataset_dir, split="val", **self._dataset_kwargs
        )
        return DataLoader(
            self.val_dataset, shuffle=False, **self._loader_kwargs
        )


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[: x.size(0)]  # type: ignore
        return self.dropout(x)


class BeatTransformer(nn.Module):
    def __init__(self, d_model, nhead, d_hid, dropout, nlayers):
        super(BeatTransformer, self).__init__()
        # d_model is the model's input and output dimensionality
        self.embedding = nn.Linear(128, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model, nhead, d_hid, dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)

        self.beats_head = nn.Linear(d_model, 2)
        # self.downbeats_head = nn.Linear(d_model, 1)
        # self.time_signatures_head = nn.Linear(d_model, 2) # one for numerator, one for denominator
        # self.key_signatures_head = nn.Linear(d_model, num_keys)
        # self.onsets_musical_head = nn.Linear(d_model, 1)
        # self.note_value_head = nn.Linear(d_model, 1)
        # self.hands_head = nn.Linear(d_model, 1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x.permute(1, 0, 2)).permute(1, 0, 2)
        x += self.positional_encoding(x)
        x = self.transformer_encoder(x)
        beats = self.beats_head(x)
        # beats = torch.softmax(self.beats_head(x), dim=-1)
        # downbeats = torch.sigmoid(self.downbeats_head(x))
        # time_signatures = self.time_signatures_head(x)
        # key_signatures = self.key_signatures_head(x)
        # onsets_musical = self.onsets_musical_head(x)
        # note_value = self.note_value_head(x)
        # hands = torch.sigmoid(self.hands_head(x))

        return beats  # , downbeats, time_signatures, key_signatures, onsets_musical, note_value, hands
