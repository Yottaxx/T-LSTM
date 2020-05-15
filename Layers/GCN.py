import torch.nn as nn
import torch.nn.functional as F

from Layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(in_size, hidden_size)
        self.gc2 = GraphConvolution(hidden_size, out_size)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x
