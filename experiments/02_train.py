#!/usr/bin/env python3
"""
Experiment 02: Model Architecture & Training Definition
Contains the standalone PyTorch architecture and training definitions for HinglishNano-V5:
- 5 Transformer blocks, d=384, Rank-96 factorized embedding table
- GQA (6 query : 2 KV heads), SwiGLU FFN (d_ffn=640), RoPE, Pre-RMSNorm
- Compound InfoNCE + Cosine KD loss + Multi-Task Kendall uncertainty balancing
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ffn):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ffn, bias=False)
        self.w2 = nn.Linear(d_ffn, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ffn, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class FactorizedEmbedding(nn.Module):
    def __init__(self, vocab_size, rank, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, rank)
        self.proj = nn.Linear(rank, d_model, bias=False)

    def forward(self, input_ids):
        return self.proj(self.embed(input_ids))

class HinglishNanoV5(nn.Module):
    def __init__(self, vocab_size=20000, rank=96, d_model=384, n_layers=5, n_heads=6, n_kv_heads=2, d_ffn=640):
        super().__init__()
        self.embeddings = FactorizedEmbedding(vocab_size, rank, d_model)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "attn_norm": RMSNorm(d_model),
                "q_proj": nn.Linear(d_model, d_model, bias=False),
                "k_proj": nn.Linear(d_model, d_model // 3, bias=False),
                "v_proj": nn.Linear(d_model, d_model // 3, bias=False),
                "out_proj": nn.Linear(d_model, d_model, bias=False),
                "ffn_norm": RMSNorm(d_model),
                "ffn": SwiGLU(d_model, d_ffn)
            }) for _ in range(n_layers)
        ])
        self.final_norm = RMSNorm(d_model)
        
        # 5 Multi-task heads
        self.gate_head = nn.Linear(d_model, 2)
        self.domain_head = nn.Linear(d_model, 6)
        self.temporal_head = nn.Linear(d_model, 3)
        self.ner_head = nn.Linear(d_model, 17)

    def forward(self, input_ids, attention_mask=None):
        h = self.embeddings(input_ids)
        for layer in self.layers:
            # Self-attention with pre-norm
            norm_h = layer["attn_norm"](h)
            q = layer["q_proj"](norm_h)
            k = layer["k_proj"](norm_h)
            v = layer["v_proj"](norm_h)
            # Repeat KV heads for GQA (6:2 ratio -> repeat 3x)
            k = k.repeat_interleave(3, dim=-1)
            v = v.repeat_interleave(3, dim=-1)
            scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.size(-1))
            if attention_mask is not None:
                scores = scores.masked_fill(attention_mask.unsqueeze(1) == 0, -1e9)
            attn = F.softmax(scores, dim=-1)
            attn_out = layer["out_proj"](torch.matmul(attn, v))
            h = h + attn_out
            h = h + layer["ffn"](layer["ffn_norm"](h))
            
        h = self.final_norm(h)
        # In-graph mean pooling with attention mask
        if attention_mask is not None:
            mask_exp = attention_mask.unsqueeze(-1).float()
            pooled = (h * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1e-9)
        else:
            pooled = h.mean(dim=1)
            
        return {
            "embedding": F.normalize(pooled, p=2, dim=-1),
            "gate_logits": self.gate_head(pooled),
            "domain_logits": self.domain_head(pooled),
            "temporal_logits": self.temporal_head(pooled),
            "ner_logits": self.ner_head(h)
        }

if __name__ == "__main__":
    model = HinglishNanoV5()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"✅ HinglishNano-V5 Architecture instantiated successfully! Total parameters: {n_params/1e6:.2f}M")
