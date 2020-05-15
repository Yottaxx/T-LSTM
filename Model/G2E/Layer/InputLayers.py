import torch.nn as nn
import torch.nn.functional as F
import torch


class InputLayers(nn.Module):
    def __init__(self, in_size, out_size, num_layers=1, dropout=0.3, bi=True):
        super(InputLayers, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.lstm = nn.LSTM(in_size, in_size, num_layers=num_layers, batch_first=True,dropout=0,
                            bidirectional=bi)
        if bi:
            self.linear = nn.Linear(in_size * 2, out_size)
        else:
            self.linear = nn.Linear(in_size * 1, out_size)
        self.dropout=nn.Dropout(dropout)
    def forward(self, x):
        x, _ = self.lstm(x)
        return self.dropout(self.linear(x))
