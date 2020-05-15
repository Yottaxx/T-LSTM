import torch.nn as nn
import torch
from Attention import MultiHeadedAttention
from S_lstm.ExgLayer import ExgLayer
import torch.nn.functional as F

from S_lstm.GraphConvolution import GraphConvolution
from S_lstm.S_LSTMCell import S_LSTMCell
from S_lstm.S_LSTMLayer import S_LSTMLayer


class S_LSTM(nn.Module):
    def __init__(self, x_size, g_size, out_size):
        super(S_LSTM, self).__init__()
        self.x_size = x_size
        self.g_size = g_size
        self.out_size = out_size
        self.BSLSTM = S_LSTMLayer(x_size=x_size, g_size=g_size, out_size=out_size)
        self.gcn = GraphConvolution(x_size, g_size)

    def forward(self, x, trigger, adj):
        residual = x
        x = self.gcn(x, adj)
        one_hot = F.one_hot(torch.arange(0, trigger.max() + 1), x.shape[1]).to(trigger.device)
        trigger = one_hot[trigger].unsqueeze(-1)
        trigger = trigger.expand(trigger.shape[0], trigger.shape[1], x.shape[-1]).bool()
        g = x.masked_select(trigger).reshape(x.shape[0], x.shape[-1])
        x, _, _ = self.BSLSTM(residual,g)

        return x

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# model = S_LSTMLayer(x_size=10, g_size=30, out_size=10).cuda()
# x = torch.rand(2, 5, 10).to(device)
# model(x)
