import torch.nn as nn
import torch.nn.functional as F
import torch


class PreLayers(nn.Module):
    def __init__(self, in_size, hidden_size,out__size, dropout=0.3):
        super(PreLayers, self).__init__()
        self.in_size = in_size
        self.out__size = out__size
        self.hidden_size = hidden_size
        self.linear_1 = nn.Linear(in_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, out__size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # batch*len*size
        x = self.linear_2(self.dropout(F.relu(self.linear_1(x))))
        return x
