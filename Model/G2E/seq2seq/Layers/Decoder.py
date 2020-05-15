import torch.nn as nn
import torch.nn.functional as F
import torch


class Decoder(nn.Module):
    def __init__(self, in_size, out_size, num_layers, dropout=0.3, bi=True):
        super(Decoder, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.linear_a = nn.Linear(in_size*3, out_size)
        self.linear_s = nn.Linear(in_size*2, out_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h_state, x, trigger, mask=None):
        # batch*len*size
        a = torch.cat((h_state, x), -1)

        one_hot = F.one_hot(torch.arange(0, trigger.max() + 1), a.shape[1]).to(trigger.device)
        trigger = one_hot[trigger].unsqueeze(-1)
        trigger = trigger.expand(trigger.shape[0], trigger.shape[1], a.shape[-1]).bool()
        trigger_a = a.masked_select(trigger).reshape(a.shape[0], 1, a.shape[-1])

        s = h_state.sum(dim=1)
        # print("s")
        # print(s.shape)
        # print(a.shape)
        et = F.tanh(self.linear_a(a) + self.linear_s(s).unsqueeze(1))
        if mask is not None:
            mask = mask.squeeze().unsqueeze(-1).expand((mask.shape[0], mask.shape[-1], et.shape[-1]))
            et = et.masked_fill(mask == 0, -1e9)
        attn = self.dropout(F.softmax(et, dim=-1))
        print(trigger_a.shape)
        print(et.shape)
        out = (trigger_a * attn).sum(dim=1)
        return out


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Decoder(10, 10*3, 1).to(device)
x = torch.rand(4, 5, 10).to(device)
h = torch.rand(4, 5, 20).to(device)
trigger = torch.tensor([1, 2, 3, 4]).to(device)
mask = torch.ones(4, 5).to(device)
out = model(h, x, trigger, mask)
print(out.shape)