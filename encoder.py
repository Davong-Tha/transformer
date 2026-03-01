import torch
import torch.nn as nn
import torch.nn.functional as F
from multi_headed_attention import MultiHeadedAttention

class Encoder(nn.Module):
    def __init__(self, dv, dk, num_heads, d_head, dropout=0.1):
        super(Encoder, self).__init__()
        self.d_model = num_heads * d_head
        self.multihead_attn = MultiHeadedAttention(dv=dv, dk=dk, num_heads=num_heads, d_head=d_head)
        self.norm1 = nn.LayerNorm(self.d_model)
        self.linear = nn.Sequential(
            nn.Linear(self.d_model, 4*self.d_model),
            nn.GELU(),
            nn.Linear(4*self.d_model, self.d_model),
        )
        self.norm2 = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, padding_mask=None):
        attn_out = self.multihead_attn(x, x, x, padding_mask=padding_mask, id='e') # (B, T, d_model)
        attn_out = self.dropout(attn_out)
        norm1_out = self.norm1(attn_out + x)  # Residual connection and layer normalization
        linear_out = self.linear(norm1_out)
        linear_out = self.dropout(linear_out)
        out = self.norm2(linear_out + norm1_out)  # Another residual connection and layer normalization
        return out