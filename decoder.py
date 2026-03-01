import torch
import torch.nn as nn
from multi_headed_attention import MultiHeadedAttention

class Decoder(nn.Module):
    def __init__(self, dv, dk, num_heads, d_head, dropout=0.1):
        super(Decoder, self).__init__()
        self.d_model = num_heads * d_head
        self.multihead_attn1 = MultiHeadedAttention(dv=dv, dk=dk, num_heads=num_heads, d_head=d_head)
        self.multihead_attn2 = MultiHeadedAttention(dv=dv, dk=dk, num_heads=num_heads, d_head=d_head)
        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)
        self.norm3 = nn.LayerNorm(self.d_model)
        self.linear = nn.Sequential(
            nn.Linear(self.d_model, 4*self.d_model),
            nn.GELU(),
            nn.Linear(4*self.d_model, self.d_model),
        )
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, encoder_outputs, self_att_mask=None, cross_att_mask=None):
        attn1_out = self.multihead_attn1(x, x, x, padding_mask=self_att_mask, casual_mask=True, id='d self') # (B, T, d_model)
        attn1_out = self.dropout(attn1_out)
        norm1_out = self.norm1(attn1_out + x)  # (B, T, d_model) Residual connection and layer normalization
        attn2_out = self.multihead_attn2(encoder_outputs, encoder_outputs, norm1_out, padding_mask=cross_att_mask, id='d cross') # (B, T, d_model)
        attn2_out = self.dropout(attn2_out)
        norm2_out = self.norm2(attn2_out + norm1_out)  # (B, T, d_model) Residual connection and layer normalization
        
        linear_out = self.linear(norm2_out) # (B, T, d_model)
        linear_out = self.dropout(linear_out)
        out = self.norm3(linear_out + norm2_out)  # (B, T, d_model) Another residual connection and layer normalization
        return out