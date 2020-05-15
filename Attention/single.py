import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention --baseline version"""

    def __init__(self, dropout=0.3):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn
