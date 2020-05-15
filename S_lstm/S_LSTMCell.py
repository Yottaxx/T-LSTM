import torch.nn as nn
import torch
from Layers import GraphConvolution, SentenceMatrixLayer, EncodeLayer
from Attention import MultiHeadedAttention
from S_lstm.ExgLayer import ExgLayer
import torch.nn.functional as F


class S_LSTMCell(nn.Module):
    def __init__(self, x_size, h_size, g_size, out_size, bi=True, dropout=0.3):
        super(S_LSTMCell, self).__init__()
        self.x_size = x_size
        self.g_size = g_size
        self.out_size = out_size
        self.h_size = h_size
        self.dropout = nn.Dropout(dropout)
        self.ILayer = ExgLayer(x_size=x_size, h_size=h_size, g_size=g_size, out_size=out_size)
        self.FLayer = ExgLayer(x_size=x_size, h_size=h_size, g_size=g_size, out_size=out_size)
        self.OLayer = ExgLayer(x_size=x_size, h_size=h_size, g_size=g_size, out_size=out_size)
        self.ULayer = ExgLayer(x_size=x_size, h_size=h_size, g_size=g_size, out_size=out_size)

    def forward(self, x, h=None, c=None, g=None):
        # batch_size * in_size
        if h is None:
            h = torch.zeros(x.shape[0], self.h_size).to(x.device)
        if g is None:
            g = torch.zeros(x.shape[0], self.g_size).to(x.device)
        I_L = torch.sigmoid(self.ILayer(x, h, g))
        F_L = torch.sigmoid(self.ILayer(x, h, g))
        O_L = torch.sigmoid(self.ILayer(x, h, g))
        U_L = torch.tanh(self.ULayer(x, h, g))

        if c is None:
            c = torch.zeros_like(I_L).to(x.device)

        C_L = c * F_L + U_L * I_L
        New_H = O_L * torch.tanh(C_L)
        return New_H, New_H,C_L
