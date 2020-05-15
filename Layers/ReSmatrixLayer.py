import torch.nn as nn
import torch.nn.functional as F
import torch
import torch
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data


# graph functional
class ReSentenceMatrixLayer(nn.Module):
    def __init__(self, in_size, out_size=1):
        super(ReSentenceMatrixLayer, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.a_Asem=nn.Parameter(torch.tensor(0.0))
        self.linear = nn.Linear(in_size * 2, out_size)

    def forward(self, x, adj):
        # x batch*node*emb
        # adj batch*node*node

        # adj is dense batch*node*node*(2*emb)
        # 2*emb for cat xi,xj
        # new_adj = adj.unsqueeze(-1)
        # new_adj = new_adj.expand(new_adj.shape[0], new_adj.shape[1], new_adj.shape[2], x.shape[-1] * 2)

        # xi batch*n*1*emb expand  dim 1 decide x[n]
        xi = x.unsqueeze(-2)
        xi = xi.expand(xi.shape[0], xi.shape[1], xi.shape[1], xi.shape[-1])
        # xj  #xi batch*1*n*emb   dim 2 decide x[n]
        xj = x.unsqueeze(1)
        xj = xj.expand(xj.shape[0], xj.shape[2], xj.shape[2], xj.shape[-1])

        # cat [xi,xj]
        xij = torch.cat((xi, xj), -1)

        #here for rezero have a try
        A_esm = (torch.sigmoid(self.linear(xij).squeeze()))+self.a_Asem*adj.to_dense()

        return A_esm

##test
# edge_index = torch.tensor([[0, 1, 1, 2],
#                            [1, 0, 2, 1]], dtype=torch.long)
# x = torch.rand((3, 100))
# tri = torch.rand((1, 72))
# data = Data(x=x, edge_index=edge_index)
# device = torch.device('cuda')
# data = data.to(device)
# tri = tri.to(device)
# model = FRGN(100, 1)
# model.cuda()
# test = model(data)
# print(test)
