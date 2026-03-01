import torch
import torch.nn as nn
import math
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
import os 
from tokenizers import Tokenizer

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model) # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # (max_len, 1)
        deno = 10000 ** (2 * torch.arange(0, d_model, 2).float() / d_model) # (d_model/2,) -> broadcast to (1, d_model/2)
        pe[:, 0::2] = torch.sin(position / deno)
        pe[:, 1::2] = torch.cos(position / deno)
        self.pe = pe.unsqueeze(0) # (1, max_len, d_model)

    def forward(self, x):
        # x: (B, T, d_model)
        T = x.size(1)
        pe = self.pe[:, :T].to(device=x.device, dtype=x.dtype)
        return x + pe


    import torch
from torch.utils.data import Dataset

class CharDataset(Dataset):
    def __init__(self, text: str, block_size: int, stride: int = 256,
                 stoi=None, itos=None, add_unk=True):
        self.block_size = block_size
        self.stride = stride

        # Build vocab only if not provided (do this ONLY on train)
        if stoi is None or itos is None:
            chars = sorted(list(set(text)))
            if add_unk and "<unk>" not in chars:
                chars = ["<unk>"] + chars
            self.stoi = {ch: i for i, ch in enumerate(chars)}
            self.itos = {i: ch for ch, i in self.stoi.items()}
        else:
            self.stoi = stoi
            self.itos = itos

        self.vocab_size = len(self.stoi)
        self.unk_id = self.stoi.get("<unk>", None)
        self.stoi['<BOS>'] =100
        self.itos[100] = '<BOS>'

        # Encode corpus using shared vocab
        ids = []
        for c in text:
            if c in self.stoi:
                ids.append(self.stoi[c])
            else:
                if self.unk_id is None:
                    raise ValueError(f"Found OOV char {repr(c)} but no <unk> in vocab")
                ids.append(self.unk_id)

        self.data = torch.tensor(ids, dtype=torch.long)

     # total tokens needed per sample
        self.window = block_size * 2

        # maximum valid starting index
        self.max_start = len(self.data) - self.window

        if self.max_start <= 0:
            raise ValueError("Dataset too small for given block_size")

    def __len__(self):
        return (self.max_start // self.stride) + 1

    def __getitem__(self, idx):
        start = idx * self.stride
        x = self.data[start : start + self.block_size]
        y = self.data[start + self.block_size : start + self.block_size + 32]
        y = torch.concat([torch.tensor([1]),y])  # Append a special token (e.g., padding)
        return x, y

    def token_to_id(self, token: str) -> int:
        out = []
        for c in token:
            out.append(self.stoi.get(c))
        return torch.tensor(out, dtype=torch.long)
    def id_to_token(self, idx_list) -> str:
        out = []
        for i in idx_list:
            out.append(self.itos.get(i))
        return ''.join(out)

def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_ddp():
    dist.destroy_process_group()
    


    
import torch

class subWordDataset(torch.utils.data.Dataset):
    def __init__(self, token_ids, block_size, stride, pred_len=16, bos_id=100):
        self.data = torch.tensor(token_ids, dtype=torch.long)
        self.block_size = block_size
        self.stride = stride
        self.pred_len = pred_len
        self.bos_id = bos_id

        # tokens needed from start: block_size (x) + pred_len (future y)
        self.window = block_size + pred_len
        self.max_start = len(self.data) - self.window

        if self.max_start < 0:
            raise ValueError(
                f"Dataset too small: need at least {self.window} tokens, got {len(self.data)}"
            )

        # create once (avoids allocating every __getitem__)
        self._bos = torch.tensor([bos_id], dtype=torch.long)

    def __len__(self):
        return (self.max_start // self.stride) + 1

    def __getitem__(self, idx):
        start = idx * self.stride
        x = self.data[start : start + self.block_size]  # (block_size,)

        y_future = self.data[start + self.block_size : start + self.block_size + self.pred_len]  # (pred_len,)
        y = torch.cat([self._bos, y_future], dim=0)  # (pred_len+1,)

        return x, y



class SingleCharDataSet(Dataset):
    def __init__(self, pairs, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = pairs
        self.max_len = max_len
        self.ignore_index = -100
        
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        x_text, y_text = self.data[index]

        x = self.tokenizer.encode(x_text).ids
        tgt = self.tokenizer.encode(y_text).ids

        pad = self.tokenizer.token_to_id('<PAD>')
        bos = self.tokenizer.token_to_id('<BOS>')
        eos = self.tokenizer.token_to_id('<EOS>') # optional

        # ---- encoder input (pad/truncate to max_len) ----
        x = x[:self.max_len]
        x = x + [pad] * (self.max_len - len(x))

        # ---- target ids (optionally add EOS), then clamp to max_len ----
        # print(eos)
        if eos is not None:
            # print('adding eos', eos)
            tgt = tgt[:self.max_len - 1] + [eos]
        else:
            tgt = tgt[:self.max_len]

        # labels: tgt + ignore_index padding
        labels = tgt + [self.ignore_index] * (self.max_len - len(tgt))

        # decoder input: BOS + tgt[:-1] + PAD
        dec_in = [bos] + tgt[:-1]
        dec_in = dec_in + [pad] * (self.max_len - len(dec_in))

        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(dec_in, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )


class SimpleTokenizer:
    def __init__(self, text):
        self.vocab = {ch: i for i, ch in enumerate(sorted(set(text)))}
        self.inv_vocab = {i: ch for ch, i in self.vocab.items()}
        self.vocab_size = len(self.vocab)
        self.pad_id = self.vocab_size
        self.bos_id = self.vocab_size + 1
        self.eos_id = self.vocab_size + 2
        self.vocab['<PAD>'] = self.pad_id
        self.vocab['<BOS>'] = self.bos_id
        self.vocab['<EOS>'] = self.eos_id
        self.inv_vocab[self.pad_id] = '<PAD>'
        self.inv_vocab[self.bos_id] = '<BOS>'
        self.inv_vocab[self.eos_id] = '<EOS>'
        self.vocab_size = len(self.vocab)

    def encode(self, text):
        return [self.vocab.get(ch, self.vocab.get("<unk>")) for ch in text]

    def decode(self, ids):
        text = ''.join([self.inv_vocab.get(i, "<unk>") for i in ids])
        return text
    def token_to_id(self, token):
        return self.vocab[token]
    
    