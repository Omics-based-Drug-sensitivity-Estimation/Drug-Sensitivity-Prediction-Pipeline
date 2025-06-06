# DrugEmbedding.py
"""Drug‑level encoders
======================
This module provides two lightweight building blocks:

1. **DrugEmbeddingModel** – a graph Transformer that consumes
   *node feature matrices* (`x`) **plus** adjacency & bond‑type matrices
   and outputs a *token‑level* embedding (CLS + atoms).

2. **ChemEncoder** – a thin convenience wrapper around a pretrained
   ChemBERTa backbone that exposes `hidden_size` and optionally freezes
   the backbone during finetuning.

Both encoders share a simple, self‑contained API and *never* import from
any dataset code, so they can be reused in inference / serving pipelines
without pulling in DeepChem, RDKit, or pandas.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from utils.utils import get_device

__all__: Sequence[str] = (
    "AdjMultiHeadAttention",
    "DrugEmbeddingModel",
    "ChemEncoder",
)

# ---------------------------------------------------------------------
# 1) Graph Transformer (BGD) – token‑level output (CLS first)
# ---------------------------------------------------------------------
class AdjMultiHeadAttention(nn.Module):
    """Multi‑head attention modulated by *adjacency* & *bond type* matrices."""

    def __init__(self, embed_dim: int, num_heads: int, num_bond_types: int):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # 0 → padding (no bond);   1‥num_bond_types → bond category
        self.bond_embed = nn.Embedding(num_bond_types + 1, 1, padding_idx=0)
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor, bond: torch.Tensor) -> torch.Tensor:
        """Shape
            x    : (B, N, H)
            adj  : (B, N, N)  – 0/1
            bond : (B, N, N)  – categorical (0 = pad)
        """
        B, N, H = x.shape
        qkv = (
            self.qkv_proj(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )  # 3,B,h,N,d
        q, k, v = qkv  # each (B,h,N,d)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B,h,N,N)
        attn = attn + adj.unsqueeze(1)  # encourage real edges
        attn = attn + self.bond_embed(bond.long()).squeeze(-1).unsqueeze(1)
        attn = attn.masked_fill(attn == 0, -1e9).softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, N, H)
        return self.out_proj(out)


class TransformerEncoderLayer(nn.Module):
    """Encoder = MH‑Attention + FFN + residuals (+ norm/dropout)."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        num_bond_types: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.attn = AdjMultiHeadAttention(embed_dim, num_heads, num_bond_types)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(ff_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, adj, bond):
        x = self.norm1(x + self.drop(self.attn(x, adj, bond)))
        x = self.norm2(x + self.drop(self.ff(x)))
        return x


class DrugEmbeddingModel(nn.Module):
    """Graph Transformer that returns CLS + per‑atom embeddings."""

    def __init__(
        self,
        in_dim: int,
        embed_dim: int,
        n_heads: int,
        ff_dim: int,
        n_layers: int,
        n_bond_types: int,
        max_nodes: int,
    ) -> None:
        super().__init__()
        self.project = nn.Linear(in_dim, embed_dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, embed_dim))  # learnable CLS
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(embed_dim, n_heads, ff_dim, n_bond_types)
                for _ in range(n_layers)
            ]
        )
        self.max_nodes = max_nodes
        self.device = get_device()
        self.to(self.device)

    # ------------------------------------------------------------
    # forward
    # ------------------------------------------------------------
    def forward(self, x: torch.Tensor, adj: torch.Tensor, bond: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Return **full sequence** embedding; caller may slice `[ :, 0 ]` for CLS."""
        B, N, _ = x.shape  # N ≤ max_nodes
        x = self.project(x)

        # prepend CLS token & pad matrices accordingly
        cls_tok = self.cls.expand(B, -1, -1)
        x = torch.cat([cls_tok, x], dim=1)  # (B, N+1, H)

        pad_fn = lambda m, v=0: F.pad(m, (1, 0, 1, 0), value=v)
        adj, bond = pad_fn(adj, 1), pad_fn(bond)

        for layer in self.layers:
            x = layer(x, adj, bond)
        return x  # (B, N+1, H)

# ---------------------------------------------------------------------
# 2) ChemEncoder (tokenised SMILES → token embeddings)
# ---------------------------------------------------------------------
class ChemEncoder(nn.Module):
    """Wrapper around a HuggingFace backbone that *optionally* freezes weights."""

    def __init__(self, backbone_name: str, *, freeze: bool = True):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name)
        if freeze:
            self.backbone.requires_grad_(False)
        self.hidden_size: int = self.backbone.config.hidden_size

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Return token‑level embeddings (`last_hidden_state`)."""
        return self.backbone(input_ids=ids, attention_mask=mask).last_hidden_state