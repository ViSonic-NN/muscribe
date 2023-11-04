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
import madmom.features
from pytorch_lightning.utilities import grad_norm



logger = logging.getLogger(__name__)


class BeatPredictorPL(pl.LightningModule):
    def __init__(
        self,
        dataset_dir: Path | str,
        lr: float = 1e-4,
        batch_size: int = 7,
        seq_len_sec: int = 150,
        grid_len: float = 0.02,
        **dataset_kwargs,
    ):
        super().__init__()
        self.dataset_dir = Path(dataset_dir)
        self.lr = lr
        self.batch_size = batch_size
        self.model = BeatCRNN(256, 128, 4, 0.1)
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

        optim = torch.optim.AdamW(self.model.parameters(), self.lr, weight_decay=0.00001)
        #optim = torch.optim.SGD(self.model.parameters(), self.lr, momentum=0.9)
        scheduler = {
            "scheduler": sched.CosineAnnealingLR(optim, T_max=self.num_training_steps),
            "interval": "step",
        }
        # scheduler = sched.ReduceLROnPlateau(optim, patience=10, factor=0.25, min_lr=1e-6)
        # return {"optimizer": optim, "lr_scheduler": scheduler, "monitor": "val/recall"}
        return [optim], [scheduler]

    def training_step(self, batch, _):
        notes, gt_beats, _, _ = batch
        assert torch.isnan(notes).any() == False, "Assert Failed: NaN in input"
        pred_probs = self(notes)
        return self._metrics(pred_probs, gt_beats, True)

    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        # example to inspect gradient information in tensorboard
        if self.trainer.global_step % 25 == 0:  # don't make the tf file huge
            for k, v in self.named_parameters():
                self.logger.experiment.add_histogram(
                    tag=k, values=v.grad, global_step=self.trainer.global_step
                )


    def validation_step(self, batch, _):
        notes, gt_beats, _, _ = batch
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
        # cls_weights = torch.tensor([1, 10, 20]).float().cuda()
        # Convert gt_beats into two binary labels
        binary_label_1 = (gt_beats == 1).float()
        binary_label_2 = (gt_beats == 2).float()

        # check for nan
        assert torch.count_nonzero(gt_beats != gt_beats) == 0

        # No need to compute binary_label_2 as it's redundant (1 - binary_label_0 - binary_label_1)
        # Predicted labels based on logits
        predicted_1 = torch.sigmoid(pred_probs[:,:,1]) > 0.5
        predicted_2 = torch.sigmoid(pred_probs[:,:,2]) > 0.5

        # Identify TP, TN, FP, FN for the two binary labels
        TP_1 = (binary_label_1 * predicted_1.float()).float()
        TN_1 = ((1 - binary_label_1) * (1 - predicted_1.float())).float()

        TP_2 = (binary_label_2 * predicted_2.float()).float()
        TN_2 = ((1 - binary_label_2) * (1 - predicted_2.float())).float()

        # Assign weights
        weight_TP = 20.0
        weight_TN = 1.0

        weights_1 = TP_1 * (weight_TP - 1) + TN_1 * (weight_TN - 1) + 1
        weights_2 = TP_2 * (weight_TP - 1) + TN_2 * (weight_TN - 1) + 1

        # Compute binary cross-entropy for each binary label
        bce_loss_1 = f.binary_cross_entropy_with_logits(pred_probs[:,:,1], binary_label_1, weight = weights_1)
        bce_loss_2 = f.binary_cross_entropy_with_logits(pred_probs[:,:,2], binary_label_2, weight = weights_2)

        # Sum the two binary cross-entropies
        ce_loss = bce_loss_1 + bce_loss_2 +   2*(torch.count_nonzero(binary_label_1) + torch.count_nonzero(binary_label_2))/(gt_beats.shape[1]*gt_beats.shape[0])

        # Checking for nan again, sigh

        if (ce_loss != ce_loss):
            print(gt_beats)
            print(pred_probs)

        assert ce_loss == ce_loss

        predicted = pred_probs.argmax(dim=-1)
        gt_beats_, pred_beats_ = gt_beats != 0, predicted != 0
        b_recall = (gt_beats_ & pred_beats_).sum() / gt_beats_.sum()
        b_prec = (gt_beats_ & pred_beats_).sum() / pred_beats_.sum()
        b_f1 = 2 * b_prec * b_recall / (b_prec + b_recall)
        downbeats = gt_beats == 2
        db_recall = (downbeats & (predicted == 2)).sum() / downbeats.sum()

        prefix = "train/" if is_train else "val/"
        sync_dist = not is_train
        self.log(f"{prefix}loss", ce_loss, sync_dist = sync_dist, prog_bar=is_train)
        self.log(f"{prefix}b_prec", b_prec, sync_dist = sync_dist)
        self.log(f"{prefix}b_recall", b_recall, sync_dist = sync_dist)
        self.log(f"{prefix}b_f1", b_f1, sync_dist = sync_dist)
        self.log(f"{prefix}db_recall", db_recall, sync_dist = sync_dist)
        return ce_loss if is_train else b_recall



    @property
    def example_input_array(self):
        return torch.rand(1, self.seq_n_samples, 128)
    
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

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class BeatCRNN(nn.Module):
    def __init__(self, d_model: int, d_hid: int, n_rnn_layers: int, dropout: float):
        super().__init__()
        self.embedding = nn.Linear(128, d_model)
        self.lazy_norm = nn.LazyBatchNorm1d()
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
        # self.beat_cls = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(d_hid, 3))
        # self.beat_cls = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(d_hid, 3))
    def forward(self, x: Tensor):
        assert torch.isnan(x).any() == False, "Assert Failed: NaN before embedding"
        assert torch.count_nonzero(x) > 0, "Assert Failed: input Tensor is all zero"
        # x: [batch_size, seq_len, 128]
        x1 = self.embedding(x)
        if(torch.isnan(x1).any()):
            print(x)
            #print(x1)
        assert torch.isnan(x1).any() == False, "Assert Failed: NaN after embedding"

        x2 = self.lazy_norm(x1)
        # x: [batch_size, seq_len, d_model]
        assert torch.isnan(x2).any() == False, "Assert Failed: NaN after lazy norm"
        x3 = self.convs(x2.transpose(1, 2)).transpose(1, 2)
        # x: [batch_size, seq_len, d_hid]
        assert torch.isnan(x3).any() == False, "Assert Failed: NaN after conv"
        gru_beat, _ = self.gru(x3)
        assert torch.isnan(gru_beat).any() == False, "Assert Failed: NaN after gru"
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
            nn.LeakyReLU (),
            nn.Conv1d(
                in_channels=in_feats,
                out_channels=out_feats,
                kernel_size=1,
                padding=0,
            ),
            nn.BatchNorm1d(out_feats),
            nn.LeakyReLU (),
            nn.Dropout(p=dropout),
        ]
