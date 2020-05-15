import torch.nn as nn
import torch
import torch.nn.functional as F

from Layers import ReSentenceMatrixLayer, SentenceMatrixLayer, GCN
from Model.G2E.attention import MultiHeadedAttention
from Model.G2E.gcn.GraphConvolution import GraphConvolution
from Model.Relstm.ReLSTM import ReLSTM
from Model.Relstm.ReLSTMLayer import ReLSTMLayer
from .Layer import Decoder, Encoder, InputLayers, PreLayers


class GraphStatelstm(nn.Module):
    def __init__(self, vocab_len, in_size=300, lstm_h_size=1200, g_out_size=300, at_size=1200, l_h_size=300, out_size=1,
                 dropout=0.3, bi=True, Ec_layer=2):
        super(GraphStatelstm, self).__init__()
        self.ELayer_nums = Ec_layer
        self.embedding = nn.Embedding(vocab_len, in_size)
        self.dropout = nn.Dropout(dropout)
        # self.gcn=GCN(in_size,in_size*2,1,dropout)
        # self.pregcn = GraphConvolution(in_size,in_size)
        self.lstm = nn.LSTM(in_size, in_size, num_layers=2, batch_first=True, bidirectional=True,dropout=dropout)
        # self.lstm3 = nn.LSTM(in_size, int(in_size / 2), num_layers=1, batch_first=True, bidirectional=True)
        # self.context_emb = SentenceMatrixLayer(in_size*2)
        # self.gcn = GraphConvolution(in_size*2,in_size)
        # self.dropout = nn.Dropout(dropout)
        # self.linear=nn.Linear(in_size,in_size*2)
        self.PreLayer = PreLayers(in_size * 2, in_size, out_size)
        self.dropout2 =nn.Dropout(dropout)
        self.dropout3=nn.Dropout(dropout)
    def forward(self, x, adj,trigger,mask=None):
        x = self.embedding(x)
        # for i in range(self.ELayer_nums):
        #     x, _ = self.EncoderList[i](x, adj)
        x,_ = self.lstm(x)

        x = self.PreLayer(x)
        one_hot = F.one_hot(torch.arange(0, trigger.max() + 1), x.shape[1]).to(trigger.device)
        trigger = one_hot[trigger].unsqueeze(-1)
        trigger = trigger.expand(trigger.shape[0], trigger.shape[1], x.shape[-1]).bool()
        # print("----------trigger----------------")
        # print(trigger)
        trigger_x = x.masked_select(trigger).reshape(x.shape[0], 1, x.shape[-1])
        return trigger_x.squeeze()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GraphState2eep(10).to(device)
    x = torch.rand(4, 5, 10).to(device)
    adj = torch.ones(4, 5, 5).to(device)
    trigger = torch.tensor([1, 2, 3, 4]).to(device)
    mask = torch.ones(4, 5).to(device)
    out = model(x, trigger, adj, mask)
    print(out.shape)
