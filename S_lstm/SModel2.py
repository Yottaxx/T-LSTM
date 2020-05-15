import torch.nn as nn
import torch
from Attention import MultiHeadedAttention
from S_lstm.ExgLayer import ExgLayer
import torch.nn.functional as F

from S_lstm.GraphConvolution import GraphConvolution
from S_lstm.S_LSTM import S_LSTM
from S_lstm.S_LSTMCell import S_LSTMCell
from S_lstm.S_LSTMLayer import S_LSTMLayer


class SModel(nn.Module):
    def __init__(self, vocab_len, in_size=300, g_size=100, linear_h_size=300, out_size=1, dropout=0.3):
        super(SModel, self).__init__()
        self.in_size = in_size
        self.g_size = g_size
        self.out_size = out_size
        self.linear_h_size = linear_h_size
        self.embedding = nn.Embedding(vocab_len, in_size,padding_idx=1)
        self.S_LSTM = S_LSTM(x_size=in_size, g_size=g_size, out_size=in_size)
        self.li = nn.Linear(in_size*2,in_size)
        self.S_LSTM2 = S_LSTM(x_size=in_size, g_size=g_size, out_size=in_size)
        self.dropout = nn.Dropout(dropout)
        self.outlinear1 = nn.Linear(in_size * 2, linear_h_size)
        self.outlinear2 = nn.Linear(linear_h_size, out_size)

    def forward(self, x, trigger, adj):
        x = self.embedding(x)
        x = self.S_LSTM(x, trigger, adj)
        x= F.relu(self.li(x))
        x = self.S_LSTM2(x, trigger, adj)
        x =  F.relu(self.dropout(x))

        x = self.outlinear1(x)
        x =  F.relu(x)
        x = self.outlinear2(x)
        one_hot = F.one_hot(torch.arange(0, trigger.max() + 1), x.shape[1]).to(trigger.device)
        trigger = one_hot[trigger].unsqueeze(-1)
        trigger = trigger.expand(trigger.shape[0], trigger.shape[1], x.shape[-1]).bool()
        # print("----------trigger----------------")
        # print(trigger)
        trigger_x = x.masked_select(trigger).reshape(x.shape[0], 1, x.shape[-1])
        return trigger_x.squeeze()

