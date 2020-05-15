import torch.nn as nn
import torch.nn.functional as F
import torch


class Decoder(nn.Module):
    def __init__(self, h_size, x_size, dropout=0.3):
        super(Decoder, self).__init__()
        self.h_size = h_size
        self.x_size = x_size
        self.linear_a = nn.Linear(h_size + x_size, h_size + x_size)
        self.linear_s = nn.Linear(h_size, h_size + x_size)
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
        et = torch.tanh(self.linear_a(a) + self.linear_s(s).unsqueeze(1))
        if mask is not None:
            mask = mask.squeeze().unsqueeze(-1).expand((mask.shape[0], mask.shape[-1], et.shape[-1]))
            et = et.masked_fill(mask == 0, -1e9)
        attn = self.dropout(F.softmax(et, dim=-1))
        out = (trigger_a * attn).sum(dim=1)
        return out


