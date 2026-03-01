from matplotlib import lines
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from transformer import Transformer_model
from util import CharDataset, SimpleTokenizer, SingleCharDataSet, SinusoidalPositionalEncoding, setup_ddp, cleanup_ddp, subWordDataset
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torch

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import random
import numpy as np 

def get_dataLoaders(block_size=512, batch_size=1, stride=256, multiprocessing_distributed=False):
    # 1) Load the file you downloaded from the browser
    with open("./input.txt", "r", encoding="utf-8") as f:
        text = f.read()

    # 2) Train/val split (standard simple split)
    # for faster experimentation, we use only first 10k characters
    # text = text[:5000]
    n = int(0.9 * len(text))
    train_text = text[:n]
    val_text = text[n:]
    

    # 3) Make datasets + loaders
    block_size = block_size   # context length
    batch_size = batch_size

    # 1) build train dataset (vocab built here)
    train_ds = CharDataset(train_text, block_size=block_size, stride=stride, add_unk=True)

    # 2) build val dataset using the SAME vocab
    val_ds = CharDataset(val_text, block_size=block_size, stride=stride,
                        stoi=train_ds.stoi, itos=train_ds.itos, add_unk=True)
    print(train_ds.vocab_size)
    if multiprocessing_distributed:
        train_sampler = DistributedSampler(train_ds, shuffle=True)
        val_sampler = DistributedSampler(val_ds, shuffle=False)

        # print(f"train has {len(train_ds)} samples")

        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler, drop_last=True)
        val_loader   = DataLoader(val_ds, batch_size=batch_size, sampler=val_sampler, drop_last=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=True)
    # print("vocab_size:", train_ds.vocab_size)
    # print('val ds vocab_size:', val_ds.vocab_size)
    # x, y = next(iter(train_loader))
    # print("x shape:", x.shape, "y shape:", y.shape)  # (B, T), (B, T)
    
    return train_loader, val_loader, train_ds.vocab_size, train_ds, val_ds


def getSubwordDataloader(block_size=512, batch_size=64, stride=256, multiprocessing_distributed=False):
    with open("./input.txt", "r", encoding="utf-8") as f:
        text = f.read()

    chunks = text.split('\n\n')  # split by double newlines to get paragraphs
   
    
    tokenizer = Tokenizer.from_file("tokenizer.json")
    train_text = ''
    val_text = ''
    for i, chunk in enumerate(chunks[:-1]): 
        if i % 10 == 0:  
            val_text += chunk + '\n\n'
        else:   
            train_text += chunk + '\n\n'
   

    train_ids = tokenizer.encode(train_text).ids
    val_ids   = tokenizer.encode(val_text).ids

    train_ds = subWordDataset(train_ids, block_size=block_size, stride=stride, bos_id=tokenizer.token_to_id('[BOS]'))
    val_ds = subWordDataset(val_ids, block_size=block_size, stride=stride, bos_id=tokenizer.token_to_id('[BOS]'))
    

    if multiprocessing_distributed:
        train_sampler = DistributedSampler(train_ds, shuffle=True)
        val_sampler = DistributedSampler(val_ds, shuffle=False)

        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler, drop_last=True)
        val_loader   = DataLoader(val_ds, batch_size=batch_size, sampler=val_sampler, drop_last=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loader, val_loader, tokenizer.get_vocab_size(), train_ds, val_ds

import re

# 1) "First Citizen: hello"
SPEAKER_INLINE_RE = re.compile(r'^([A-Za-z][A-Za-z \'-]*):\s*(.*)$')

# 2) "GLOUCESTER:" (all caps speaker line, maybe with spaces/apostrophes)
SPEAKER_ALONE_RE = re.compile(r'^([A-Z][A-Z \'-]*):\s*$')

def collapse_dialogue(text, drop_empty=True):
    lines = text.splitlines()

    out = []
    cur_speaker = None
    cur_text = []

    def flush():
        nonlocal cur_speaker, cur_text
        if cur_speaker is None:
            return
        joined = ' '.join(s.strip() for s in cur_text if s.strip())
        joined = re.sub(r'\s+', ' ', joined).strip()
        if joined or not drop_empty:
            out.append(f"{cur_speaker}: {joined}".rstrip())
        cur_speaker, cur_text = None, []

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        m_alone = SPEAKER_ALONE_RE.match(line)
        if m_alone:
            flush()
            cur_speaker = m_alone.group(1).title()  # "GLOUCESTER" -> "Gloucester"
            cur_text = []
            continue

        m_inline = SPEAKER_INLINE_RE.match(line)
        if m_inline:
            flush()
            cur_speaker = m_inline.group(1).strip()
            rest = m_inline.group(2).strip()
            cur_text = [rest] if rest else []
            continue

        if cur_speaker is not None:
            cur_text.append(line)

    flush()
    return out



def get_SimpleDataloader(block_size=512, batch_size=64, stride=256, multiprocessing_distributed=False, num_line=1):
    with open("./input.txt", "r", encoding="utf-8") as f:
        text = f.read()

    text = text.lower()
    tokenizer = SimpleTokenizer(text)
    
    
    lines = [b.strip() for b in text.split('\n\n') if b.strip()]
    
    collapse = []
    for line in lines:
        speaker = line.split('\n')
        if len(speaker) > 1:
            speaker = ' '.join(speaker)
            collapse.append(speaker)
    
   
    
    # text = "\n".join(lines)
    # print(text)
    with open('./out2.txt', 'w') as f:
        for l in collapse:
            f.write(l)
            f.write('\n-------------------------------\n')
    
    # print(text[:1000])
    
    # print(tokenizer.bos_id)
    
    # lines = text.splitlines()
    lines = collapse
    n = int(0.8* len(lines))
    train_lines = [ln.strip() for ln in lines[:n] if ln.strip()]
    val_lines   = [ln.strip() for ln in lines[n:] if ln.strip()]

    train_pairs = pairMaker(train_lines, num_line)
    val_pairs = pairMaker(val_lines, num_line)
    x_train_lens = [len(l[0]) for l in train_pairs]
    x_val_lens = [len(l[0]) for l in val_pairs]
    x_train_lens = np.array(x_train_lens)
    x_val_lens = np.array(x_val_lens)
    p75_train = np.percentile(x_train_lens, [25, 50, 75, 90, 95])
    p75_val = np.percentile(x_val_lens,  [25, 50, 75, 90, 95])
    print(p75_train)
    print(p75_val)
    # print(p75)
    # print(len(val_pairs))
    # print(train_pairs[0][0])
    # print('--------------------')
    # print(train_pairs[0][1])
    


    train_ds = SingleCharDataSet(train_pairs, tokenizer=tokenizer, max_len=256)
    val_ds = SingleCharDataSet(val_pairs, tokenizer=tokenizer, max_len=256)
    
    if multiprocessing_distributed:
        train_sampler = DistributedSampler(train_ds, shuffle=True)
        val_sampler = DistributedSampler(val_ds, shuffle=False)

        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler, drop_last=True)
        val_loader   = DataLoader(val_ds, batch_size=batch_size, sampler=val_sampler, drop_last=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=True)
    
    return train_loader, val_loader, tokenizer.vocab_size, train_ds, val_ds

import random

def pairMaker(lines, n, seed=None):
    if seed is not None:
        random.seed(seed)

    start = 0
    pairs = []
    L = len(lines)

    while True:
        # need at least 1 line for x and 1 line for y
        remaining = L - start
        if remaining <= 1:
            break

        max_x = min(n, remaining - 1)   # ensure y exists
        x_len = random.randint(1, max_x)

        x_lines = lines[start : start + x_len]
        y_line  = lines[start + x_len]

        x = '\n'.join(x_lines)  # or '\n'.join(x_lines) if you want newlines kept
        pairs.append((x, y_line))

        start += x_len + 1

    return pairs


def getTwitterDataloader(batch_size=64, multiprocessing_distributed=False):
    with open("./TwitterLowerAsciiCorpus.txt", "r", encoding="utf-8") as f:
        text = f.read()

    text = text.lower()
    tokenizer = Tokenizer.from_file("tokenizer.json")
    
    
    lines = [b.strip() for b in text.split('\n\n\n') if b.strip()]
    pairs = []
    all = []
    for l in lines:
        tweet = l.split('\n')
        tweet = [_ for _ in tweet if len(_) > 0]
        if len(tweet) > 2:
            for i in range(len(tweet) - 1):
                all.append(tweet[i].strip())
            all.append(tweet[len(tweet) - 1].strip())
           

            # for i in range(1, len(tweet)):
            #     context = tweet[max(0, i-3):i]   # last K turns
            #     target = tweet[i]
            #     x = '\n'.join(context)
            #     pairs.append((x, target))
                
            # for i in range(1, len(tweet)):
            #     context = tweet[max(0, i-2):i]   # last K turns
            #     target = tweet[i]
            #     x = '\n'.join(context)
            #     pairs.append((x, target))
            
            for i in range(1, len(tweet)):
                context = tweet[max(0, i-1):i]   # last K turns
                target = tweet[i]
                x = ''.join(context)
                pairs.append((x, target))
                 
            
        elif len(tweet) == 2:
            pairs.append((tweet[0].strip(), tweet[1].strip()))
            all.append(tweet[0].strip())
            all.append(tweet[1].strip())
    
    with open('./out2.txt', 'w') as f:
        for l in pairs:
            f.write(l[0])
            f.write('\n')
            f.write(l[1])
            f.write('\n')
            # f.write(l[1])
            f.write('-------------------------------\n')
    
   
    print(np.percentile(np.array([len(a[0]) for a in pairs]), [25, 50, 75, 90, 95]))
    random.shuffle(pairs)
    n = int(0.8 * len(pairs))
    
    train_pairs = pairs[:n]
    val_pairs = pairs[n:]
    
    train_ds = SingleCharDataSet(train_pairs, tokenizer=tokenizer, max_len=256)
    val_ds = SingleCharDataSet(val_pairs, tokenizer=tokenizer, max_len=256)
    
    if multiprocessing_distributed:
        train_sampler = DistributedSampler(train_ds, shuffle=True)
        val_sampler = DistributedSampler(val_ds, shuffle=False)

        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler, drop_last=True)
        val_loader   = DataLoader(val_ds, batch_size=batch_size, sampler=val_sampler, drop_last=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=True)
    
    return train_loader, val_loader, tokenizer.get_vocab_size(), train_ds, val_ds
    
    
                
            
        
    

    
if __name__ == "__main__":
    train_loader, val_loader, vocab_size, train_ds, val_ds = getTwitterDataloader()
    # print(f"Vocab size: {vocab_size}")
    # print(f"Train dataset size: {len(train_ds)}")
    # print(f"Val dataset size: {len(val_ds)}")
    print("Sample data:")
    train_sample = next(iter(train_loader))
    print("x:", train_sample[0].shape)  # first sample's x
    print("dec_in:", train_sample[1].shape)  # first sample's decoder input
    print("label ", train_sample[2].shape)
    # print(train_ds.tokenizer.bos_id)
    # print(train_ds.tokenizer.eos_id)
    # print(train_ds.tokenizer.pad_id)
    # seq = train_sample[2][0]
    # eos_pos = (seq == 41).nonzero(as_tuple=True)[0]

    # if len(eos_pos) > 0:
    #     valid_len = eos_pos[0].item() + 1
    # else:
    #     valid_len = len(seq)
    # print(valid_len)
    # print("labels:", train_sample[2][:, :valid_len+1])  # first sample's labels