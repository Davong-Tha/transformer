import math
from dataset import get_SimpleDataloader, get_dataLoaders, getSubwordDataloader, getTwitterDataloader
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from transformer import Transformer_model
from util import CharDataset, SinusoidalPositionalEncoding, setup_ddp, cleanup_ddp
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

import time

import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from tokenizers import Tokenizer

local_rank = setup_ddp()
train_loader, val_loader, vocab_size, train_ds, val_ds = getTwitterDataloader()
tokenizer = Tokenizer.from_file("tokenizer.json")
# if dist.get_rank() == 0:
#     print(train_ds[0][0].shape)
#     print(train_ds[0][1].shape)
#     print(tokenizer.decode(train_ds[0][0].tolist()))
#     print("-----")
#     print(tokenizer.decode(train_ds[0][1].tolist()))
    

device = torch.device(f"cuda:{local_rank}")
# print(vocab_size)
pad_id = train_ds.tokenizer.token_to_id('<PAD>')
print(pad_id)
model = Transformer_model(num_heads=4, d_head=32, dk=32, dv=32, vocab_size=vocab_size, dropout=0.1, pad_id=pad_id)
# model.load_state_dict(torch.load("./runs/20260213-083917/weights/best.pt")['model_state'])

model = model.to(device)
model = DDP(model, device_ids=[local_rank], output_device=local_rank)


num_epochs = 1500
run_timestamp = time.strftime("%Y%m%d-%H%M%S")
os.makedirs(f"runs/{run_timestamp}/weights", exist_ok=True)
writer = SummaryWriter(log_dir=f"runs/{run_timestamp}")
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, betas=(0.9, 0.98), weight_decay=0.01)
scheduler = CosineAnnealingLR(
    optimizer,
    T_max=num_epochs,   # decay over epochs
    eta_min=3e-5      # final LR
)


best_val_loss = float('inf')

def train_step(x, y, ygt, debug=False):
    x = x.to(device)  # (B, T)
    y = y.to(device)  # (B, T)
    ygt = ygt.to(device)  # (B, T)
    # print(y)
    # print('train x shape:', x.shape, 'y shape:', y.shape)
    # print(x[0])
    # return
    logits = model({'x': x, 'y': y})  # (B, T-1, vocab_size)
    # print('ygt shape:', ygt.shape)
    # print('logits shape:', logits.shape)
    criterion = nn.CrossEntropyLoss(label_smoothing=0, ignore_index=-100)
    loss = criterion(
        logits.reshape(-1, vocab_size),
        ygt.reshape(-1)
    )
    
    
    if debug:
        with torch.no_grad():
            logp = F.log_softmax(logits, dim=-1)                  # (B, T-1, V)
            correct_logp = logp.gather(-1, ygt.unsqueeze(-1))   # (B, T-1, 1)
            avg_nll = (-correct_logp).mean().item()
            avg_p = correct_logp.exp().mean().item()

        print("avg NLL:", avg_nll)
        print("avg p(correct):", avg_p)
        print("random baseline p:", 1.0 / vocab_size)
        print("random baseline loss ln(V):", math.log(vocab_size))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def evaluate():
    model.eval()
    total_loss = torch.zeros((), device=device)
    count = torch.zeros((), device=device)
    for x, y, ygt in val_loader:
        x = x.to(device)
        y = y.to(device)
        ygt = ygt.to(device)
        
        count += 1

        logits = model({'x': x, 'y': y})
        # print('logits mean:', logits.mean().item(), 'logits std:', logits.std().item(), 'logits max:', logits.max().item(), 'logits min:', logits.min().item())
        criterion = nn.CrossEntropyLoss(label_smoothing=0, ignore_index=-100)
        loss = criterion(
            logits.reshape(-1, vocab_size),
            ygt.reshape(-1)
        )
       
        total_loss += loss.detach()
    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(count, op=dist.ReduceOp.SUM)

    return (total_loss/count).item()



for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    t0 = time.perf_counter()
    total_loss = torch.zeros((), device=device)
    count = torch.zeros((), device=device)
    for i, (x, y, ygt) in enumerate(train_loader):
        loss = train_step(x, y, ygt)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
        total_loss += loss
        count += 1
        if dist.get_rank() == 0:
            writer.add_scalar("loss/train_step", total_loss/count, epoch * len(train_loader) + i) # local rank 0 only
    
       
        

    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(count, op=dist.ReduceOp.SUM)
    avg_train_loss = (total_loss / count).item()  
    val_loss = evaluate()
    scheduler.step()
                                                  
    

    
    if dist.get_rank() == 0:
        print(f"Epoch {epoch+1} | Train loss: {avg_train_loss:.4f} | Time: {time.perf_counter() - t0:.2f} sec")
        print(f"Epoch {epoch+1} | Val loss: {val_loss:.4f}")
        # writer.add_scalar('loss/train', avg_train_loss, epoch+1)   
        # writer.add_scalar('loss/val', val_loss, epoch+1) 
        writer.add_scalars('loss', {
            'train': avg_train_loss,
            'val': val_loss
        }, epoch+1)
        
        torch.save(
        {
            "epoch": epoch + 1,
            "model_state": model.module.state_dict(),  # IMPORTANT for DDP
            "optimizer_state": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
        },
        f"runs/{run_timestamp}/weights/latest.pt"
        )
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state": model.module.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                },
                f"runs/{run_timestamp}/weights/best.pt"
            )
    
    
       

cleanup_ddp()