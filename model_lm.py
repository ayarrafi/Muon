"""Small GPT-style causal language model for LLM optimizer comparison.

Architecture: pre-norm transformer (GPT-2 style).
Default config (~10 M params) fits comfortably in 6 GB VRAM.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, context_len: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.context_len = context_len
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, 4 * d_model, bias=False)
        self.fc2 = nn.Linear(4 * d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, context_len: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, context_len)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 384,
        n_heads: int = 6,
        n_layers: int = 6,
        context_len: int = 256,
    ):
        super().__init__()
        self.context_len = context_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(context_len, d_model)
        self.blocks = nn.Sequential(*[Block(d_model, n_heads, context_len) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        x = self.blocks(x)
        return self.lm_head(self.ln_f(x))

    def num_params(self) -> int:
        # exclude tied lm_head weight
        return sum(p.numel() for n, p in self.named_parameters() if "lm_head" not in n)


def split_params_for_muon_gpt(
    model: GPT,
) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    """Split GPT params into (muon_params, adamw_params).

    Muon group: hidden 2D weight matrices (attention qkv/proj, MLP fc1/fc2).
    AdamW group: embeddings, layer norms, lm_head (tied to tok_emb so excluded).
    """
    muon_names = {"qkv.weight", "proj.weight", "fc1.weight", "fc2.weight"}
    muon, adamw = [], []
    seen_ids: set[int] = set()
    for name, p in model.named_parameters():
        if id(p) in seen_ids or not p.requires_grad:
            continue
        seen_ids.add(id(p))
        leaf = name.rsplit(".", 1)[-1] if "." in name else name
        parent_leaf = ".".join(name.rsplit(".", 2)[-2:]) if name.count(".") >= 2 else name
        if parent_leaf in muon_names or leaf in muon_names:
            muon.append(p)
        else:
            adamw.append(p)
    return muon, adamw
