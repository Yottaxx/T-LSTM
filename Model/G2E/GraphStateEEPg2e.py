import torch.nn as nn
import torch

from Layers import GraphConvolution
from .Layer import Decoder, Encoder, InputLayers, PreLayers
import torch.nn.functional as F

class GraphState2eepg2e(nn.Module):
    def __init__(self,vocab_len ,in_size, lstm_h_size=10, g_out_size=15,l_h_size=25, out_size=1,
                 dropout=0.3,
                 bi=True, Ec_layer=2):
        super(GraphState2eepg2e, self).__init__()
        self.ELayer_nums = Ec_layer
        self.embedding = nn.Embedding(vocab_len, in_size)
        self.gcn = GraphConvolution(in_size, in_size)
        self.ContextEmb = InputLayers(in_size, lstm_h_size)
        self.EncoderList = nn.ModuleList(
            Encoder(lstm_h_size, g_out_size, int(lstm_h_size/2), dropout=dropout, bidirectional=bi) for _ in
            range(Ec_layer))

        self.PreLayer = PreLayers(lstm_h_size , l_h_size, out_size)

    def forward(self, x, adj, trigger, mask=None):
        x = self.embedding(x)
        x = self.ContextEmb(x)
        for i in range(self.ELayer_nums):
            x, _ = self.EncoderList[i](x, adj)
        one_hot = F.one_hot(torch.arange(0, trigger.max() + 1), x.shape[1]).to(trigger.device)
        trigger = one_hot[trigger].unsqueeze(-1)
        trigger = trigger.expand(trigger.shape[0], trigger.shape[1], x.shape[-1]).bool()
        trigger_x = x.masked_select(trigger).reshape(x.shape[0], 1, x.shape[-1])
        x = self.PreLayer(trigger_x)
        return x.squeeze()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GraphState2eep(10).to(device)
    x = torch.rand(4, 5, 10).to(device)
    adj = torch.ones(4, 5, 5).to(device)
    trigger = torch.tensor([1, 2, 3, 4]).to(device)
    mask = torch.ones(4, 5).to(device)
    out = model(x, trigger, adj, mask)
    print(out.shape)
