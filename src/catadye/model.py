"""Small GPT whose MLP is a chain of sub-blocks (CMS-style).

Every arm uses the same chain of `n_sub` sub-blocks, so parameter count and forward FLOPs are
identical across arms; arms differ only in how each sub-block is *updated* (see staggered.py).
Sub-blocks are residual: x = x + mlp_i(ln_i(x)). The paper's chain has no stated residuals; this is
our design choice, kept identical across arms.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    n_layer: int = 4
    n_head: int = 4
    d_model: int = 256
    ctx: int = 256
    n_sub: int = 3          # sub-blocks per MLP chain (CMS levels)
    sub_hidden_mult: float = 4 / 3  # per-sub-block hidden = mult * d_model, so the chain ~ one 4x MLP
    dropout: float = 0.0


class Attention(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        self.n_head = c.n_head
        self.qkv = nn.Linear(c.d_model, 3 * c.d_model, bias=False)
        self.proj = nn.Linear(c.d_model, c.d_model, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).contiguous().view(B, T, C))


class SubMLP(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        h = int(c.d_model * c.sub_hidden_mult)
        self.ln = nn.LayerNorm(c.d_model)
        self.up = nn.Linear(c.d_model, h, bias=False)
        self.down = nn.Linear(h, c.d_model, bias=False)

    def forward(self, x):
        return x + self.down(F.gelu(self.up(self.ln(x))))


class MLPChain(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        self.subs = nn.ModuleList(SubMLP(c) for _ in range(c.n_sub))

    def forward(self, x):
        for s in self.subs:
            x = s(x)
        return x


class Block(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(c.d_model)
        self.attn = Attention(c)
        self.mlp = MLPChain(c)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        # the chain adds its own residuals, so it takes and returns the residual stream
        return self.mlp(x)


class GPT(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        self.c = c
        self.wte = nn.Embedding(c.vocab_size, c.d_model)
        self.wpe = nn.Embedding(c.ctx, c.d_model)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layer))
        self.ln_f = nn.LayerNorm(c.d_model)
        self.lm_head = nn.Linear(c.d_model, c.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight  # tie
        self.apply(self._init)
        for n, p in self.named_parameters():
            if n.endswith("proj.weight") or n.endswith("down.weight"):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * c.n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.wte(idx) + self.wpe(torch.arange(T, device=idx.device))
        for b in self.blocks:
            x = b(x)
        logits = self.lm_head(self.ln_f(x))
        if targets is None:
            return logits
        return logits, F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
