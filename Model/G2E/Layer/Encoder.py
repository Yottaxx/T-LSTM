import torch.nn as nn
import torch.nn.functional as F
import torch
from Model.G2E.gcn import GCN


class Encoder(nn.Module):
    def __init__(self, in_size, g_out_size,h_out_size , dropout=0.3, num_layers=1, bidirectional=True):
        super(Encoder, self).__init__()
        self.g_out_size = g_out_size
        self.gcn = GCN(in_size, g_out_size, dropout)
        self.lstm = nn.LSTM(g_out_size, h_out_size, num_layers=num_layers, batch_first=True,bidirectional=bidirectional)
        self.dropout=nn.Dropout(dropout)
    def forward(self, x, adj):
        x = self.gcn(x, adj)
        x, state = self.lstm(x)
        return self.dropout(x), state
