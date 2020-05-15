import torch.nn as nn
import torch.nn.functional as F
import torch
from Layers import GraphConvolution, SentenceMatrixLayer, EncodeLayer
from Attention import MultiHeadedAttention


class ExgLayer(nn.Module):
    def __init__(self, x_size, h_size, g_size, out_size):
        super(ExgLayer, self).__init__()
        self.h_size = h_size
        self.g_size = g_size
        self.out_size = out_size
        self.x_size = x_size
        self.linear_x2 = nn.Linear(x_size, out_size)
        self.linear_h2 = nn.Linear(h_size, out_size)
        self.linear_g2 = nn.Linear(g_size, out_size)

    def forward(self,  x,h, g):
        return self.linear_x2(x) + self.linear_h2(h) + self.linear_g2(g)
