import torch.nn as nn
import torch.nn.functional as F
import torch
import torch
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data


# graph functional
class EncodeLayer(nn.Module):
    def __init__(self, in_size, hidden_size, num_layers=2, dropout=0.3, bi=True):
        super(EncodeLayer, self).__init__()
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.nun_layers = num_layers
        self.bilstm = nn.LSTM(in_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout,
                              bidirectional=bi)

    def forward(self, x):
        output, _ = self.bilstm(x)
        return output


