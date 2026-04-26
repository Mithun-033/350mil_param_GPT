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
    value_embed_rank:int=16
    
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
        
def Value_Embed_Layer(layer_id):
    return Config.n_layer%2==(layer_idx)%2
    
class MultiHeadAttention(nn.Module):
    def __init__(self,config,layer_id):
        super().__init__()
        self.n_head=config.n_head
        self.head_size=config.head_size

        self.qkv=nn.Linear(config.n_embed,3*self.n_head*self.head_size,bias=False)
        self.proj=nn.Linear(self.n_head*self.head_size,config.n_embed)

        self.proj.weight._is_residual_proj = True 

        self.q_norm=nn.RMSNorm(self.head_size,eps=1e-5)
        self.k_norm=nn.RMSNorm(self.head_size,eps=1e-5)

        self.rope=RotaryEmbedding(self.head_size)
        self.value_gate=nn.Linear(config.value_embed_rank,config.n_head) if Value_Embed_Layer(layer_id) else None

    def forward(self,x,ve):
        B,T,C=x.shape
        qkv=self.qkv(x)
        q,k,v=qkv.split(self.n_head*self.head_size,dim=-1)

        q=q.view(B,T,self.n_head,self.head_size)
        k=k.view(B,T,self.n_head,self.head_size)
        v=v.view(B,T,self.n_head,self.head_size)

        q=self.q_norm(q)
        k=self.k_norm(k)
        
        if self.vaue_projection:
            ve=ve.view(B,T,self.n_head,self.head_size)
            gate=4*torch.sigmoid(self.value_gate((x[:,:,:self.config.value_embed_rank]))
            v=v+gate.unqueeze(-1)*ve
            ve=v
        q,k=self.rope(q,k)

        q=q.transpose(1,2)
        k=k.transpose(1,2)
        v=v.transpose(1,2)

        y=F.scaled_dot_product_attention(q,k,v,is_causal=True)
        Vn=F.normalize(v, dim=-1)

        out=y - (y * Vn).sum(dim=-1, keepdim=True)*Vn
        out=out.transpose(1,2).reshape(B,T,self.n_head*self.head_size)
        return [self.proj(out),ve]

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
    def __init__(self,config,layer_id):
        super().__init__()
        self.PreNorm1=nn.RMSNorm(config.n_embed,eps=1e-5)
        self.attention=MultiHeadAttention(config,layer_id)
        self.PreNorm2=nn.RMSNorm(config.n_embed,eps=1e-5)
        self.FeedForwardLayer=MLP(config)
        self.res_scale=1/math.sqrt(2*config.n_layer)

    def forward(self,x,ve):
        logits,ve=self.attention(self.PreNorm1(x),ve)
        x=x+self.res_scale*logits
        x=x+self.res_scale*self.FeedForwardLayer(self.PreNorm2(x))
        return [x,ve]

class GPT(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config=config
        self.embed=nn.Embedding(config.vocab_size,config.n_embed)
        self.blocks=nn.ModuleList([Block(config,layer_id+1) for layer_id in range(config.n_layer)])
        self.final_norm=nn.RMSNorm(config.n_embed,eps=1e-5)
        self.Dense=nn.Linear(config.n_embed,config.vocab_size,bias=False)
        self.ve=torch.zeros(config.b_size,config.cwl,self.n_head,self.head_size)
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
        
        for i in self.blocks:
              x,self.ve=self.blocks[i](x,self.ve)
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


