import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import LogmelFilterBank, Spectrogram

from .. import params


def init_layer(layer):
    """Initialize a Linear or Convolutional layer."""
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.0)


def init_bn(bn):
    """Initialize a Batchnorm layer."""
    bn.bias.data.fill_(0.0)
    bn.weight.data.fill_(1.0)


def init_gru(rnn):
    """Initialize a GRU layer."""

    def _concat_init(tensor, init_funcs):
        (length, fan_out) = tensor.shape
        fan_in = length // len(init_funcs)

        for i, init_func in enumerate(init_funcs):
            init_func(tensor[i * fan_in : (i + 1) * fan_in, :])

    def _inner_uniform(tensor):
        fan_in = nn.init._calculate_correct_fan(tensor, "fan_in")
        nn.init.uniform_(tensor, -math.sqrt(3 / fan_in), math.sqrt(3 / fan_in))

    for i in range(rnn.num_layers):
        _concat_init(
            getattr(rnn, "weight_ih_l{}".format(i)),
            [_inner_uniform, _inner_uniform, _inner_uniform],
        )
        torch.nn.init.constant_(getattr(rnn, "bias_ih_l{}".format(i)), 0)

        _concat_init(
            getattr(rnn, "weight_hh_l{}".format(i)),
            [_inner_uniform, _inner_uniform, nn.init.orthogonal_],
        )
        torch.nn.init.constant_(getattr(rnn, "bias_hh_l{}".format(i)), 0)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, momentum: float):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels, momentum)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum)
        self.pool = nn.AvgPool2d(kernel_size=(1, 2))
        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input):
        """
        Args:
          input: (batch_size, in_channels, time_steps, freq_bins)
        Outputs:
          output: (batch_size, out_channels, classes_num)
        """

        x = F.relu_(self.bn1(self.conv1(input)))
        x = F.relu_(self.bn2(self.conv2(x)))
        return self.pool(x)


class AcousticModelCRnn8Dropout(nn.Module):
    def __init__(self, classes_num, midfeat, momentum):
        super(AcousticModelCRnn8Dropout, self).__init__()

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=48, momentum=momentum)
        self.conv_block2 = ConvBlock(in_channels=48, out_channels=64, momentum=momentum)
        self.conv_block3 = ConvBlock(in_channels=64, out_channels=96, momentum=momentum)
        self.conv_block4 = ConvBlock(
            in_channels=96, out_channels=128, momentum=momentum
        )

        self.fc5 = nn.Linear(midfeat, 768, bias=False)
        self.bn5 = nn.BatchNorm1d(768, momentum=momentum)

        self.gru = nn.GRU(
            input_size=768,
            hidden_size=256,
            num_layers=2,
            bias=True,
            batch_first=True,
            dropout=0.0,
            bidirectional=True,
        )

        self.fc = nn.Linear(512, classes_num, bias=True)

        self.init_weight()

    def init_weight(self):
        init_layer(self.fc5)
        init_bn(self.bn5)
        init_gru(self.gru)
        init_layer(self.fc)

    def forward(self, input):
        """
        Args:
          input: (batch_size, channels_num, time_steps, freq_bins)
        Outputs:
          output: (batch_size, time_steps, classes_num)
        """

        x = self.conv_block1(input)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x)
        x = F.dropout(x, p=0.2, training=self.training)

        x = x.transpose(1, 2).flatten(2)
        x = F.relu(self.bn5(self.fc5(x).transpose(1, 2)).transpose(1, 2))
        x = F.dropout(x, p=0.5, training=self.training, inplace=True)

        (x, _) = self.gru(x)
        x = F.dropout(x, p=0.5, training=self.training, inplace=False)
        output = torch.sigmoid(self.fc(x))
        return output


class RegressOnsetOffsetFrameVelocityCRNN(nn.Module):
    def __init__(self, frames_per_second: int, classes_num: int):
        super(RegressOnsetOffsetFrameVelocityCRNN, self).__init__()
        (
            self.spectrogram_extractor,
            self.logmel_extractor,
            self.bn0,
        ) = make_extraction_layers(frames_per_second)

        midfeat = 1792
        momentum = 0.01
        self.frame_model = AcousticModelCRnn8Dropout(classes_num, midfeat, momentum)
        self.reg_onset_model = AcousticModelCRnn8Dropout(classes_num, midfeat, momentum)
        self.reg_offset_model = AcousticModelCRnn8Dropout(
            classes_num, midfeat, momentum
        )
        self.velocity_model = AcousticModelCRnn8Dropout(classes_num, midfeat, momentum)

        self.reg_onset_gru = nn.GRU(
            input_size=88 * 2,
            hidden_size=256,
            num_layers=1,
            bias=True,
            batch_first=True,
            dropout=0.0,
            bidirectional=True,
        )
        self.reg_onset_fc = nn.Linear(512, classes_num, bias=True)

        self.frame_gru = nn.GRU(
            input_size=88 * 3,
            hidden_size=256,
            num_layers=1,
            bias=True,
            batch_first=True,
            dropout=0.0,
            bidirectional=True,
        )
        self.frame_fc = nn.Linear(512, classes_num, bias=True)

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_gru(self.reg_onset_gru)
        init_gru(self.frame_gru)
        init_layer(self.reg_onset_fc)
        init_layer(self.frame_fc)

    def forward(self, input):
        """
        Args:
          input: (batch_size, data_length)
        Outputs:
          output_dict: dict, {
            'reg_onset_output': (batch_size, time_steps, classes_num),
            'reg_offset_output': (batch_size, time_steps, classes_num),
            'frame_output': (batch_size, time_steps, classes_num),
            'velocity_output': (batch_size, time_steps, classes_num)
          }
        """
        # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(self.spectrogram_extractor(input))
        # BN over the freq_bins dim
        x = self.bn0(x.transpose(1, 3)).transpose(1, 3)

        # frame_output, reg_onset_output, reg_offset_output, velocity_output:
        # (batch_size, time_steps, classes_num)
        frame_output = self.frame_model(x)
        reg_onset_output = self.reg_onset_model(x)
        reg_offset_output = self.reg_offset_model(x)
        velocity_output = self.velocity_model(x)

        # Use velocities to condition onset regression
        x = torch.cat(
            (reg_onset_output, (reg_onset_output**0.5) * velocity_output.detach()),
            dim=2,
        )
        (x, _) = self.reg_onset_gru(x)
        x = F.dropout(x, p=0.5, training=self.training, inplace=False)
        # reg_onset_output: (batch_size, time_steps, classes_num)
        reg_onset_output = torch.sigmoid(self.reg_onset_fc(x))

        # Use onsets and offsets to condition frame-wise classification
        x = torch.cat(
            (frame_output, reg_onset_output.detach(), reg_offset_output.detach()), dim=2
        )
        (x, _) = self.frame_gru(x)
        x = F.dropout(x, p=0.5, training=self.training, inplace=False)
        # frame_output: (batch_size, time_steps, classes_num)
        frame_output = torch.sigmoid(self.frame_fc(x))
        return {
            "reg_onset_output": reg_onset_output,
            "reg_offset_output": reg_offset_output,
            "frame_output": frame_output,
            "velocity_output": velocity_output,
        }


class RegressPedalCRNN(nn.Module):
    def __init__(self, frames_per_second: int):
        super(RegressPedalCRNN, self).__init__()
        # Spectrogram extractor
        (
            self.spectrogram_extractor,
            self.logmel_extractor,
            self.bn0,
        ) = make_extraction_layers(frames_per_second)

        midfeat = 1792
        momentum = 0.01
        self.reg_pedal_onset_model = AcousticModelCRnn8Dropout(1, midfeat, momentum)
        self.reg_pedal_offset_model = AcousticModelCRnn8Dropout(1, midfeat, momentum)
        self.reg_pedal_frame_model = AcousticModelCRnn8Dropout(1, midfeat, momentum)

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)

    def forward(self, input):
        """
        Args:
          input: (batch_size, data_length)
        Outputs:
          output_dict: dict, {
            'reg_onset_output': (batch_size, time_steps, classes_num),
            'reg_offset_output': (batch_size, time_steps, classes_num),
            'frame_output': (batch_size, time_steps, classes_num),
            'velocity_output': (batch_size, time_steps, classes_num)
          }
        """
        # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(self.spectrogram_extractor(input))
        # BN over the freq_bins dim
        x = self.bn0(x.transpose(1, 3)).transpose(1, 3)

        # reg_pedal_onset_output, reg_pedal_offset_output, pedal_frame_output:
        # (batch_size, time_steps, classes_num)
        reg_pedal_onset_output = self.reg_pedal_onset_model(x)
        reg_pedal_offset_output = self.reg_pedal_offset_model(x)
        pedal_frame_output = self.reg_pedal_frame_model(x)
        return {
            "reg_pedal_onset_output": reg_pedal_onset_output,
            "reg_pedal_offset_output": reg_pedal_offset_output,
            "pedal_frame_output": pedal_frame_output,
        }


# This model is not trained, but is combined from the trained note and pedal models.
class NotePedal(nn.Module):
    def __init__(self, frames_per_second, classes_num):
        """The combination of note and pedal model."""
        super(NotePedal, self).__init__()

        self.note_model = RegressOnsetOffsetFrameVelocityCRNN(
            frames_per_second, classes_num
        )
        self.pedal_model = RegressPedalCRNN(frames_per_second)

    def load_state_dict(self, m, strict=False):
        self.note_model.load_state_dict(m["note_model"], strict=strict)
        self.pedal_model.load_state_dict(m["pedal_model"], strict=strict)

    def forward(self, input):
        return {**self.note_model(input), **self.pedal_model(input)}


def make_extraction_layers(fps: int):
    window_size = 2048
    hop_size = params.sample_rate // fps
    mel_bins = 229
    fmin = 30
    fmax = params.sample_rate // 2

    window = "hann"
    center = True
    pad_mode = "reflect"
    ref = 1.0
    amin = 1e-10
    top_db = None

    # Spectrogram extractor
    spectrogram_extractor = Spectrogram(
        n_fft=window_size,
        hop_length=hop_size,
        win_length=window_size,
        window=window,
        center=center,
        pad_mode=pad_mode,
        freeze_parameters=True,
    )
    # Logmel feature extractor
    logmel_extractor = LogmelFilterBank(
        sr=params.sample_rate,
        n_fft=window_size,
        n_mels=mel_bins,
        fmin=fmin,
        fmax=fmax,
        ref=ref,
        amin=amin,
        top_db=top_db,  # type: ignore
        freeze_parameters=True,
    )
    bn0 = nn.BatchNorm2d(mel_bins, 0.01)
    return spectrogram_extractor, logmel_extractor, bn0
