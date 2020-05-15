import torch.nn as nn
import torch
from Layers import GraphConvolution, SentenceMatrixLayer, EncodeLayer
from Attention import MultiHeadedAttention
from S_lstm.ExgLayer import ExgLayer
import torch.nn.functional as F

from S_lstm.S_LSTMCell import S_LSTMCell


class S_LSTMLayer(nn.Module):
    def __init__(self, x_size, g_size,out_size):
        super(S_LSTMLayer, self).__init__()
        self.x_size = x_size
        self.g_size = g_size
        self.out_size = out_size
        self.h_size = x_size
        self.Cell = S_LSTMCell(x_size=x_size, h_size=self.h_size, g_size=g_size, out_size=out_size)
        self.ReverseCell = S_LSTMCell(x_size=x_size, h_size=self.h_size, g_size=g_size, out_size=out_size)

    def forward(self, x, g=None):
        # batch * leng  *size
        # print("x",x.shape)
        re_x = self.reverse(x, 1)
        # print("rec",re_x.shape)
        h = torch.zeros(x.shape[0], self.h_size).to(x.device)
        if g is None:
            g = torch.zeros(x.shape[0], self.g_size).to(x.device)
        c = torch.zeros(x.shape[0], self.out_size).to(x.device)
        o = []

        re_h = h.clone()
        re_c = c.clone()
        re_o = []
        # print("in")
        # print(h.shape)
        # print(x.shape)
        # print(c.shape)
        # print(g.shape)
        for i in range(x.shape[1]):
            temp_o, h, c = self.Cell(x[:, i, :], h, c, g)
            # print("cycling")
            # print(temp_o.shape)
            # print(h.shape)
            # print(c.shape)
            o.append(temp_o)

        for i in range(re_x.shape[1]):
            temp_re_o, re_h, re_c = self.Cell(re_x[:, i, :], re_h, re_c, g)
            # print("recycling")
            # print(temp_re_o.shape)
            # print(re_h.shape)
            # print(re_c.shape)
            re_o.append(temp_re_o)

        re_o = torch.stack(re_o, 1)
        re_o =self.reverse(re_o,1)
        o = torch.stack(o, 1)
        out = torch.cat((o, re_o), -1)
        h_state = torch.cat((h, re_h), -1)
        c_state = torch.cat((c, re_c), -1)
        # print("out")
        # print(out.shape)
        # print(h_state.shape)
        # print(c_state.shape)
        return out, h_state, c_state

    def reverse(self, x, dim):
        ind = [i for i in range(x.size(dim) - 1, -1, -1)]
        ind = torch.tensor(ind, dtype=torch.long, device=x.device)
        reverse = torch.index_select(x, dim, ind)
        return reverse


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# model = S_LSTMLayer(x_size=10, g_size=30, out_size=10).cuda()
# x = torch.rand(2, 5, 10).to(device)
# model(x)
