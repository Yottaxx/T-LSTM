import torch.nn as nn
import torch.nn.functional as F
import torch



class InputLayers(nn.Module):
    def __init__(self, in_size, out_size, num_layers, dropout=0.3, bi=False):
        super(InputLayers, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.lstm = nn.LSTM(in_size, out_size, num_layers=num_layers, dropout=dropout,
                            bidirectional=bi)
        if bi:
            self.linear = nn.Linear(in_size * 2, out_size)
        else:
            self.linear = nn.Linear(in_size * 3, out_size)

    def forward(self, x):
        emb = x
        h, _ = self.lstm(x)
        x = torch.cat((emb, h), -1)
        return self.linear(x)


