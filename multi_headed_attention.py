import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadedAttention(nn.Module):
    def __init__(self, dv, dk, num_heads, d_head):
        super(MultiHeadedAttention, self).__init__()
        self.dv = dv
        self.dk = dk
        self.d_head = d_head
        self.d_model = num_heads * d_head
        self.num_heads = num_heads
        self.Wv = nn.Linear(self.d_model, self.dv * self.num_heads)
        self.Wk = nn.Linear(self.d_model, self.dk * self.num_heads)
        self.Wq = nn.Linear(self.d_model, self.dk * self.num_heads)

        self.Wo = nn.Linear(self.dv * self.num_heads, self.d_model)


    def forward(self, V, K, Q, padding_mask=None, casual_mask=False, id=None):
        # print("MultiHeadedAttention input shapes:", V.shape, K.shape, Q.shape)
        # print("W shapes:", self.Wv.weight.shape, self.Wk.weight.shape, self.Wq.weight.shape)
        V = self.Wv(V) # (B, T, dv) -> (B, T, d_head * num_heads)
        K = self.Wk(K) # (B, T, dk) -> (B, T, d_head * num_heads)
        Q = self.Wq(Q) # (B, T, dk) -> (B, T, d_head * num_heads)
        # need to transpose K in such a way it pairs with Q
        V = V.view(V.shape[0], V.shape[1], self.num_heads, self.dv).transpose(1, 2) # (B, num_heads, T, d_head)
        K = K.view(K.shape[0], K.shape[1], self.num_heads, self.dk).transpose(1, 2) # (B, num_heads, T, d_head)
        Q = Q.view(Q.shape[0], Q.shape[1], self.num_heads, self.dk).transpose(1, 2) # (B, num_heads, T, d_head)
        QK = torch.matmul(Q, K.transpose(2, 3))/ torch.sqrt(torch.tensor(self.dk, device=Q.device, dtype=Q.dtype)) # (B, num_heads, T, T) or (B, num_heads, T-1, T-1)
        if padding_mask is not None:
            padding_mask = padding_mask[:, None, None, :]
            QK = QK.masked_fill(padding_mask == 0, float('-inf'))
        # print(f"attention shape {QK.shape}")
        # print(f"padding mask shape {padding_mask.shape}")

        if casual_mask:
            # print('Using casual mask', id)
            mask = torch.triu(torch.ones(QK.shape[-2], QK.shape[-1], device=QK.device, dtype=torch.bool), diagonal=1) # Upper triangular matrix
            QK = QK.masked_fill(mask, float('-inf'))
        QK = F.softmax(QK, dim=-1)
        QKV = torch.matmul(QK, V) # (B, num_heads, T, d_head)
        QKV = QKV.transpose(1,2) # (B, T, num_heads, d_head)
        

        QKV = QKV.reshape(QKV.shape[0], QKV.shape[1], self.dv * self.num_heads) # (B, T, d_model)
        output = self.Wo(QKV) # (B, T, d_model) -> (B, T, d_model)
        return output