from tokenizers import Tokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F

from encoder import Encoder
from decoder import Decoder
from util import SinusoidalPositionalEncoding


class Transformer(nn.Module):
    def __init__(self, dv, dk, num_heads, d_head, num_encoder_layers, num_decoder_layers, output_dim, dropout=0.1):
        super(Transformer, self).__init__()
        self.d_model = num_heads * d_head
        self.encoder = nn.ModuleList([Encoder(dv, dk, num_heads, d_head, dropout=dropout) for _ in range(num_encoder_layers)])
        self.decoder = nn.ModuleList([Decoder(dv, dk, num_heads, d_head, dropout=dropout) for _ in range(num_decoder_layers)])
        self.linear = nn.Linear(self.d_model, output_dim)

    def forward(self, data):
        # Forward pass through encoder
        x = data['x']  # (B, T, d_model)
        y = data['y']  # (B, T, d_model)
        x_mask = data.get('x_mask', None)  # (B, T)
        y_mask = data.get('y_mask', None)  # (B, T)
        encoder_outputs = None
        enc = x
        for layer in self.encoder:
            enc = layer(enc, x_mask)

        encoder_outputs = enc # (B, T, d_model)
        dec = y
        # print("Decoder input shape:", dec.shape)
        for layer in self.decoder:
            dec = layer(dec, encoder_outputs, self_att_mask=y_mask, cross_att_mask=x_mask) # (B, T, d_model)
       
        out = self.linear(dec) # (B, T, output_dim)
        # out = F.softmax(x, dim=-1) # (B, T, output_dim)
        # print("Decoder output shape:", out.shape)
        return out  # return only the part corresponding to y (B, T, output_dim)
    
    # auto regressive prediction
    @torch.no_grad()
    def predict(self, data, token_emb, pos_emb, training=True, max_len=200):
        # x: (B, T, d_model)
        bos_id = 1
        x = data['x']
        x_mask = data['x_mask']
        enc_in= token_emb(x) # (B, T, d_model)
        enc_in = pos_emb(enc_in)   # (B, T, d_model)
        for layer in self.encoder:
            enc_in = layer(enc_in, x_mask)
        encoder_outputs = enc_in # (B, T, d_model)
       
        dec_token = torch.tensor([[bos_id]], device=x.device) # (B, 1) start with <BOS> token
      
        dec_in = token_emb(dec_token) # (B, 1, d_model)
       
        dec_in = pos_emb(dec_in)   # (B, 1, d_model)
       
        
        for i in range(max_len):
            dec_out = dec_in
            for layer in self.decoder:
                dec_out= layer(dec_out, encoder_outputs, cross_att_mask=x_mask) # (B, T, d_model)
            out = self.linear(dec_out) # (B, T, output_dim)
            prob = torch.softmax(out/0.8, dim=-1)  # (B, T, output_dim)
             # temperature
            next_token =  torch.multinomial(prob[0, -1], num_samples=1)  # (B, 1)
            # next_token = torch.argmax(prob[:, -1, :], dim=-1, keepdim=True)
            
            # print('dec_token shape before appending next token:', dec_token.shape)
            dec_token = torch.cat([dec_token, next_token.unsqueeze(0)], dim=1)  # (B, T+1)
            # print('dec token shape at step', i, ':', dec_token.shape)
            # print(f'dec token {dec_token}')
            # dec_in = torch.tensor(dec_token, device=x.device).unsqueeze(0)  # (B, T+1)
            dec_in = token_emb(dec_token) # (B, T+1, d_model)
            dec_in = pos_emb(dec_in)   # (B, T+1, d_model)
            if next_token.item() == 2:
                break
            
        return dec_token

class Transformer_model(nn.Module):
    def __init__(self, num_heads, d_head, dk, dv, vocab_size, dropout=0.1, pad_id=None):
        super().__init__()
        self.d_model = num_heads * d_head
        self.token_emb = nn.Embedding(vocab_size, self.d_model)
        self.pos_emb = SinusoidalPositionalEncoding(self.d_model)
        self.transformer = Transformer(dk, dv, num_heads, d_head, num_encoder_layers=3, num_decoder_layers=3, output_dim=vocab_size, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.pad_id = pad_id
    def forward(self, data):
        # x: (B, T)
        x = data['x']  
        y = data['y']
        x_mask = (x != self.pad_id) # (B, T) mask for padding tokens
        y_mask = (y != self.pad_id) # (B, T) mask for
         # (B, T-1)
        xin = self.token_emb(x) # (B, T, d_model)
        # print(f"before position embedding {xin.shape}")
        xin = self.pos_emb(xin)   # (B, T, d_model)
        
        yin = self.token_emb(y) # (B, T, d_model)
        yin = self.pos_emb(yin)   # (B, T, d_model)
        
        xin = self.dropout(xin)
        yin = self.dropout(yin)
        
       
        out = self.transformer({'x': xin, 'y': yin, 'x_mask': x_mask, 'y_mask': y_mask}) # (B, T, vocab_size)
        return out

    @torch.no_grad()
    def predict(self, x, max_new_tokens=100, temp=0.8, top_k=None):
        self.eval()

        
        x_mask = (x != self.pad_id)
      
        logits = self.transformer.predict({'x': x, 'x_mask': x_mask}, self.token_emb, self.pos_emb, training=False)  # (B, T+max_new_tokens, vocab_size)
            

        return logits  # return token ids (B, T+max_new_tokens)

