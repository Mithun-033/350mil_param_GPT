import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
import time
from torchinfo import summary
import matplotlib.pyplot as plt
import math
import json
from torch.nn import functional as F
from Muon import HybridOptimizer
dev="cuda" if torch.cuda.is_available() else "cpu"

#TODO more assert statements and docs
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(False)
@dataclass
class Config:
    n_embed:int=1024
    cwl:int=1024
    b_size:int=32
    n_head:int=16
    head_size:int=64
    vocab_size:int=50_257
    n_layer:int=20
    grad_steps:int=int(512/b_size)
    lr:float=3e-4
assert Config.n_head*Config.head_size==Config.n_embed

class RotaryEmbedding(nn.Module):
    def __init__(self,dim,base=10000,max_seq_len=1024):
        super().__init__()
        half=dim//2

        freq=torch.arange(half,dtype=torch.float32)
        freq=1.0/(base**(freq/half))

        pos=torch.arange(max_seq_len,dtype=torch.float32)
        freqs=torch.outer(pos,freq)

        cos=freqs.cos()[None,:,None,:]   
        sin=freqs.sin()[None,:,None,:]

        self.register_buffer("cos",cos)
        self.register_buffer("sin",sin)

    def forward(self,q,k):
        B,T,H,D=q.shape
        half=D//2

        cos=self.cos[:,:T].to(q.device)
        sin=self.sin[:,:T].to(q.device)

        q1,q2=q[...,:half],q[...,half:]
        k1,k2=k[...,:half],k[...,half:]

        q=torch.cat([q1*cos-q2*sin,q1*sin+q2*cos],dim=-1)
        k=torch.cat([k1*cos-k2*sin,k1*sin+k2*cos],dim=-1)

        return q,k

class MultiHeadAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.n_head=config.n_head
        self.head_size=config.head_size

        self.qkv=nn.Linear(config.n_embed,3*self.n_head*self.head_size,bias=False)
        self.proj=nn.Linear(self.n_head*self.head_size,config.n_embed)

        self.proj.weight._is_residual_proj = True 

        self.q_norm=nn.RMSNorm(self.head_size,eps=1e-5)
        self.k_norm=nn.RMSNorm(self.head_size,eps=1e-5)

        self.rope=RotaryEmbedding(self.head_size)

    def forward(self,x):
        B,T,C=x.shape
        qkv=self.qkv(x)
        q,k,v=qkv.split(self.n_head*self.head_size,dim=-1)

        q=q.view(B,T,self.n_head,self.head_size)
        k=k.view(B,T,self.n_head,self.head_size)
        v=v.view(B,T,self.n_head,self.head_size)

        q=self.q_norm(q)
        k=self.k_norm(k)

        q,k=self.rope(q,k)

        q=q.transpose(1,2)
        k=k.transpose(1,2)
        v=v.transpose(1,2)

        y=F.scaled_dot_product_attention(q,k,v,is_causal=True)
        Vn=F.normalize(v, dim=-1)

        out=y - (y * Vn).sum(dim=-1, keepdim=True) * Vn
        out=out.transpose(1,2).reshape(B,T,self.n_head*self.head_size)
        return self.proj(out)

class MLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        hidden=int(config.n_embed*4)
        self.up_proj=nn.Linear(config.n_embed,hidden,bias=False)
        self.down_proj=nn.Linear(hidden,config.n_embed,bias=False)
        self.down_proj.weight._is_residual_proj = True

    def forward(self,x):
        x=self.up_proj(x)
        x=F.relu(x).square()
        x=self.down_proj(x)
        return x

class Block(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.PreNorm1=nn.RMSNorm(config.n_embed,eps=1e-5)
        self.attention=MultiHeadAttention(config)
        self.PreNorm2=nn.RMSNorm(config.n_embed,eps=1e-5)
        self.FeedForwardLayer=MLP(config)
        self.res_scale=1/math.sqrt(2*config.n_layer)

    def forward(self,x):
        x=x+self.res_scale*self.attention(self.PreNorm1(x))
        x=x+self.res_scale*self.FeedForwardLayer(self.PreNorm2(x))
        return x

class GPT(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config=config
        self.embed=nn.Embedding(config.vocab_size,config.n_embed)
        self.blocks=nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.final_norm=nn.RMSNorm(config.n_embed,eps=1e-5)
        self.Dense=nn.Linear(config.n_embed,config.vocab_size,bias=False)

        self.apply(self._init_weights)
        self.Dense.weight=self.embed.weight

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, '_is_residual_proj'):
                std = 0.02 / math.sqrt(2 * self.config.n_layer)
            nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self,x):
        x=self.embed(x)
        x=self.blocks(x)
        x=self.final_norm(x)
        return self.Dense(x)

#TODO better generator
def generator(ids,b_size,cwl,device):
    ids=torch.as_tensor(ids,dtype=torch.long)
    step=b_size*cwl

    for i in range(0,len(ids)-step-1,step):
        chunk=ids[i:i+step+1]

        x=chunk[:-1].view(b_size,cwl).to(device,non_blocking=True)
        y=chunk[1:].view(b_size,cwl).to(device,non_blocking=True)

        yield x,y

model=GPT(Config()).to(dev)
model=torch.compile(model)

summary(model)

criterion=nn.CrossEntropyLoss()
optimizer=HybridOptimizer(model.parameters())

total_steps=1_000_000_000//(512*Config.cwl)
epochs=1
warmup_steps=int(total_steps*0.05)

cosine_steps=int(total_steps-warmup_steps)

warmup_scheduler=torch.optim.lr_scheduler.LinearLR(optimizer.adamw,start_factor=0.2,end_factor=1,total_iters=warmup_steps)
cosine_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer.adamw,T_max=cosine_steps,eta_min=Config.lr*0.1)
scheduler=torch.optim.lr_scheduler.SequentialLR(optimizer.adamw,[warmup_scheduler,cosine_scheduler],[warmup_steps])
#TODO try diff scheduler 
warmup_scheduler2=torch.optim.lr_scheduler.LinearLR(optimizer.muon,start_factor=0.2,end_factor=1,total_iters=warmup_steps)
cosine_scheduler2=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer.muon,T_max=cosine_steps,eta_min=Config.lr*0.1)
scheduler2=torch.optim.lr_scheduler.SequentialLR(optimizer.muon,[warmup_scheduler2,cosine_scheduler2],[warmup_steps])

val_ids=np.load("none",mmap_mode="r")
val_loss=[]
train_loss=[]
ids=np.load("none",mmap_mode="r")
steps=0

for i in range(epochs):
    start=time.time()
    loss_accum=0
    optimizer.zero_grad()
    model.train()

    for x,y in generator(ids,Config.b_size,Config.cwl,dev):

        with torch.autocast(device_type="cuda",dtype=torch.bfloat16):
            out=model(x)
            out=out.view(-1,Config.vocab_size)
            y=y.view(-1)
            loss=criterion(out,y)

        loss=loss/Config.grad_steps
        loss_accum+=loss.item()

        loss.backward()
        steps+=1

        if steps%Config.grad_steps==0:
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step()
            scheduler.step()
            scheduler2.step()
            optimizer.zero_grad()

            if steps%(20*Config.grad_steps)==0:
                tokens_processed=20*Config.grad_steps*Config.b_size*Config.cwl
                end=time.time()
                time_taken=end-start
                tps=tokens_processed/time_taken
                print(f"Loss:{loss_accum/20:.5f} | Time:{time_taken:.2f}s | Tokens/s:{tps:.0f}")
                train_loss.append(loss_accum/20)
                loss_accum=0
                start=end

        if steps%(Config.grad_steps*100)==0:
            torch.save({
                "model":model.state_dict(),
                "optimizer":optimizer.state_dict(),
                "scheduler":scheduler.state_dict(),
                "step":steps
            },"bf16-proper-4.pt")

            model.eval()

            loss_accum_val=0
            val_steps=0

            with torch.no_grad():
                for x,y in generator(val_ids,Config.b_size,Config.cwl,dev):

                    with torch.autocast(device_type="cuda",dtype=torch.bfloat16):
                        out=model(x)
                        out=out.view(-1,Config.vocab_size)
                        y=y.view(-1)
                        loss=criterion(out,y)

                    loss_accum_val+=loss.item()
                    val_steps+=1

            val_loss.append(loss_accum_val/val_steps)

            with open("bf16-4.json","w") as f:
                json.dump(val_loss,f)

            print(f"bf16:- Val Loss :{loss_accum_val/val_steps:.5f} Count:{len(val_loss)}")

            model.train()

torch.save(model.state_dict(),"GPT-bf16-3.pt")

plt.plot(val_loss,label="Validation Loss")
plt.plot(train_loss,label="Train Loss")
plt.title("Train vs Validation Loss Curve")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.legend()
plt.savefig("loss_curve_bf16.png")
plt.show()