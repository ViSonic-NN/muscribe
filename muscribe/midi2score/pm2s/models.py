import torch.nn as nn

from ...constants import NATURAL_KEY_ORDER
from .blocks import ConvBlock, GRUBlock, LinearOutput
from .utils import encode_note_sequence, get_in_features

# Ignore minor keys in key signature prediction!
KEY_VOCAB_SIZE = len(NATURAL_KEY_ORDER) // 2


class RNNKeySignatureModel(nn.Module):
    def __init__(self, hidden_size=512):
        super().__init__()
        in_features = get_in_features()
        self.convs = ConvBlock(in_features=in_features)
        self.gru = GRUBlock(in_features=hidden_size)
        self.out = LinearOutput(
            in_features=hidden_size,
            out_features=KEY_VOCAB_SIZE,
            activation_type="softmax",
        )

    def forward(self, x):
        # x: (batch_size, seq_len, len(features)==4)
        x = encode_note_sequence(x)
        x = self.convs(x)  # (batch_size, seq_len, hidden_size)
        x = self.gru(x)  # (batch_size, seq_len, hidden_size)
        y = self.out(x)  # (batch_size, seq_len, KEY_VOCAB_SIZE)
        y = y.transpose(1, 2)  # (batch_size, KEY_VOCAB_SIZE, seq_len)
        return y


class RNNHandPartModel(nn.Module):
    def __init__(self, hidden_size=512):
        super().__init__()
        in_features = get_in_features()
        self.convs = ConvBlock(in_features=in_features)
        self.gru = GRUBlock(in_features=hidden_size)
        self.out = LinearOutput(
            in_features=hidden_size, out_features=1, activation_type="sigmoid"
        )

    def forward(self, x):
        # x: (batch_size, seq_len, len(features)==4)
        x = encode_note_sequence(x)
        x = self.convs(x)  # (batch_size, seq_len, hidden_size)
        x = self.gru(x)  # (batch_size, seq_len, hidden_size)
        y = self.out(x)  # (batch_size, seq_len, 1)
        y = y.squeeze(-1)  # (batch_size, seq_len)
        return y
