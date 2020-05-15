import torch.nn as nn
import torch
from Attention import MultiHeadedAttention
from S_lstm.ExgLayer import ExgLayer
import torch.nn.functional as F

from S_lstm.GraphConvolution import GraphConvolution
from S_lstm.S_LSTM import S_LSTM
from S_lstm.S_LSTMCell import S_LSTMCell
from S_lstm.S_LSTMLayer import S_LSTMLayer


class SSkipModel(nn.Module):
    def __init__(self,bert_size=3072,in_size=512, g_size=300, linear_h_size=512, out_size=1, dropout=0.1):
        super(SSkipModel, self).__init__()
        self.in_size = in_size
        self.g_size = g_size
        self.out_size = out_size
        self.linear_h_size = linear_h_size
        self.lstm = nn.LSTM(bert_size,int(in_size/2),batch_first=True,bidirectional=True,num_layers=1)
        self.S_LSTM = nn.LSTM(in_size,in_size,batch_first=True,bidirectional=True,num_layers=1)
        self.li = nn.Linear(in_size*2,in_size)
        self.S_LSTM2 = nn.LSTM(in_size,in_size,batch_first=True,bidirectional=True,num_layers=1)
        self.dropout = nn.Dropout(dropout)
        self.outlinear1 = nn.Linear(in_size * 2, linear_h_size)
        self.outlinear2 = nn.Linear(linear_h_size, out_size)

    def forward(self, x,  adj,trigger,mask):
        x,_ = self.lstm(x)
        x,_ = self.S_LSTM(x)
        x= torch.relu(self.li(x))
        x,_= self.S_LSTM2(x)
        x = torch.relu(self.dropout(x))
        one_hot = F.one_hot(torch.arange(0, trigger.max() + 1), x.shape[1]).to(trigger.device)
        trigger = one_hot[trigger].unsqueeze(-1)
        trigger = trigger.expand(trigger.shape[0], trigger.shape[1], x.shape[-1]).bool()
        # print("----------trigger----------------")
        # print(trigger)
        trigger_x = x.masked_select(trigger).reshape(x.shape[0], 1, x.shape[-1])
        x = self.outlinear1(trigger_x)
        x = torch.relu(x)
        x = self.outlinear2(x)
        return x.squeeze()

