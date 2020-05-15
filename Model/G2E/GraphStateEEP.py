import torch.nn as nn
import torch
from .Layer import Decoder, Encoder, InputLayers, PreLayers


class GraphState2eep(nn.Module):
    def __init__(self, in_size, lstm_h_size=10, g_out_size=15,l_h_size=25, out_size=1,
                 dropout=0.3,
                 bi=True, Ec_layer=4):
        super(GraphState2eep, self).__init__()
        self.ELayer_nums = Ec_layer
        self.ContextEmb = InputLayers(in_size, lstm_h_size)
        self.EncoderList = nn.ModuleList(
            Encoder(lstm_h_size, g_out_size, lstm_h_size, dropout=dropout, bidirectional=bi) for _ in
            range(Ec_layer))
        self.Decoder = Decoder(lstm_h_size * 2, lstm_h_size)
        self.PreLayer = PreLayers(lstm_h_size * 2 + lstm_h_size, l_h_size, out_size)

    def forward(self, x, adj, trigger, mask=None):
        x = self.ContextEmb(x)
        h = torch.tensor([0.0])
        for i in range(self.ELayer_nums):
            h, _ = self.EncoderList[i](x, adj)
        x = self.Decoder(h, x, trigger, mask)
        x = self.PreLayer(x)
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
