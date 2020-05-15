import torch.nn as nn
import torch.nn.functional as F
import torch


class Encoder(nn.Module):
    def __init__(self, in_size, out_size, num_layers, dropout=0.3, bi=True):
        super(Encoder, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.lstm = nn.LSTM(in_size, out_size, num_layers=num_layers, dropout=dropout,
                            bidirectional=bi)

    def forward(self, x):
        h, _ = self.lstm(x)
        return x
