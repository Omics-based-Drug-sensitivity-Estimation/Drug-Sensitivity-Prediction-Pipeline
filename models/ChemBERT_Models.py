"""
ChemBERT-based PASO variants
===========================

v1 – Token-level Context-Attention fusion (like BGD-v1)  
v2 – Cross-Attention module operating on *mean-pooled* drug embedding  
v3 – CLS-token + Cross-Attention blocks (token-wise)

All models expect **four positional tensors**:

    drug_batch : Dict[str, torch.Tensor]   # keys: "input_ids", "attention_mask"
    gep        : torch.Tensor              # (B, 619)
    cnv        : torch.Tensor
    mut        : torch.Tensor

and return:

    preds, aux_dict   # aux_dict is empty for now
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn

from utils.drug_embedding import ChemEncoder
from utils.layers import ContextAttentionLayer, dense_layer
from utils.cross_attention import CrossAttentionModule
from utils.hyperparams import LOSS_FN_FACTORY
from utils.utils import get_device

__all__ = ["PASO_Chem_v1", "PASO_Chem_v2", "PASO_Chem_v3"]


# ---------------------------------------------------------------------
# helper: tiny dense block for omics → latent
# ---------------------------------------------------------------------
def _omics_dense(in_dim: int, out_dim: int, p: float) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(p))


# ---------------------------------------------------------------------
# v1  – Context-Attention fusion
# ---------------------------------------------------------------------
class PASO_Chem_v1(nn.Module):
    """ChemBERTa + *ContextAttention* (3 omics heads)"""

    def __init__(self, cfg: Dict):
        super().__init__()
        self.cfg = cfg
        self.device = get_device()
        self.loss_fn = LOSS_FN_FACTORY[cfg.get("loss_fn", "mse")]

        # ─ Drug encoder ───────────────────────────────────────────────
        self.encoder = ChemEncoder(
            cfg.get("tokenizer_name", "DeepChem/ChemBERTa-77M-MLM"),
            freeze=not cfg.get("train_backbone", False),
        ).to(self.device)

        H = self.encoder.hidden_size
        L = cfg["smiles_padding_length"]
        P = cfg.get("number_of_pathways", 619)
        d_attn = cfg.get("smiles_attention_size", 64)

        # ─ 3 context-attention heads (GEP / CNV / MUT) ────────────────
        mk_attn = lambda: ContextAttentionLayer(H, L, 1, P, d_attn, temperature=cfg["molecule_temperature"])
        self.ctx_blocks = nn.ModuleList([mk_attn() for _ in range(3)])

        # ─ Dense projection for each omics matrix ─────────────────────
        d_omics = cfg.get("omics_dense_size", 256)
        self.omics_proj = nn.ModuleList([_omics_dense(P, d_omics, cfg["dropout"]) for _ in range(3)])

        # ─ MLP head ───────────────────────────────────────────────────
        self.mlp = nn.Sequential(
            nn.Linear(H * 3 + d_omics * 3, 512),
            nn.ReLU(),
            nn.Dropout(cfg["dropout"]),
            nn.Linear(512, 1),
        )

    # -------------------------------------------------
    def forward(self, drug_batch: Dict[str, torch.Tensor], gep, cnv, mut):
        ids = drug_batch["input_ids"].to(self.device)
        mask = drug_batch["attention_mask"].to(self.device)
        toks = self.encoder(ids, mask)  # (B, L, H)

        omics_list = [gep, cnv, mut]
        feats: List[torch.Tensor] = []

        for ctx, om, proj in zip(self.ctx_blocks, omics_list, self.omics_proj):
            om = om.to(self.device).unsqueeze(-1)
            h, _ = ctx(toks, om)
            feats.append(h)
            feats.append(proj(om.squeeze(-1)))

        y_hat = self.mlp(torch.cat(feats, dim=1)).squeeze(-1)
        return y_hat, {}

    def loss(self, y_hat, y_true):  # training helper
        return self.loss_fn(y_hat, y_true)


# ---------------------------------------------------------------------
# v2  – Cross-Attention module (drug → omics)
# ---------------------------------------------------------------------
class PASO_Chem_v2(nn.Module):
    """ChemBERTa mean-pool + *CrossAttentionModule* fusion"""

    def __init__(self, cfg: Dict):
        super().__init__()
        self.cfg = cfg
        self.device = get_device()
        self.loss_fn = LOSS_FN_FACTORY[cfg.get("loss_fn", "mse")]

        self.encoder = ChemEncoder(
            cfg.get("tokenizer_name", "DeepChem/ChemBERTa-77M-MLM"),
            freeze=not cfg.get("train_backbone", False),
        ).to(self.device)
        H = self.encoder.hidden_size

        mk_cross = lambda: CrossAttentionModule(
            H, cfg.get("num_layers", 2), cfg.get("n_heads", 8), cfg["dropout"]
        )
        self.cross_gep, self.cross_cnv, self.cross_mut = mk_cross(), mk_cross(), mk_cross()

        self.mlp = nn.Sequential(
            nn.Linear(H * 3, 512), nn.ReLU(), nn.Dropout(cfg["dropout"]), nn.Linear(512, 1)
        )

    # -------------------------------------------------
    def forward(self, drug_batch, gep, cnv, mut):
        ids = drug_batch["input_ids"].to(self.device)
        mask = drug_batch["attention_mask"].to(self.device)
        drug_vec = self.encoder(ids, mask).mean(1).unsqueeze(1)  # (B, 1, H)

        gep, cnv, mut = [t.to(self.device) for t in (gep, cnv, mut)]

        g, _ = self.cross_gep(drug_vec, gep)
        c, _ = self.cross_cnv(drug_vec, cnv)
        m, _ = self.cross_mut(drug_vec, mut)

        y_hat = self.mlp(torch.cat([g, c, m], dim=1)).squeeze(-1)
        return y_hat, {}

    def loss(self, y_hat, y_true):
        return self.loss_fn(y_hat, y_true)


# ---------------------------------------------------------------------
# v3  – CLS token + Cross-Attention blocks
# ---------------------------------------------------------------------
class _TokenCrossBlock(nn.Module):
    """Single cross-attention block: omics-vector → token sequence."""

    def __init__(self, omics_dim: int, embed_dim: int, n_heads: int, p: float):
        super().__init__()
        self.q_proj = nn.Linear(omics_dim, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, p, batch_first=True)

    def forward(self, omics_vec, tokens, kpm):
        q = self.q_proj(omics_vec).unsqueeze(1)                 # (B,1,H)
        ctx, w = self.attn(q, tokens, tokens, key_padding_mask=kpm, need_weights=False)
        return ctx.squeeze(1)                                   # (B,H)


class PASO_Chem_v3(nn.Module):
    """CLS-token + token-wise cross attention (3 omics blocks)"""

    def __init__(self, cfg: Dict):
        super().__init__()
        self.cfg = cfg
        self.device = get_device()
        self.loss_fn = LOSS_FN_FACTORY[cfg.get("loss_fn", "mse")]

        # ─ Encoder ────────────────────────────────────────────────────
        self.encoder = ChemEncoder(
            cfg.get("tokenizer_name", "DeepChem/ChemBERTa-77M-MLM"),
            freeze=not cfg.get("train_backbone", False),
        ).to(self.device)
        H = self.encoder.hidden_size

        P = cfg.get("number_of_pathways", 619)
        d_omics = cfg.get("omics_dense_size", 256)
        make_dense = lambda: _omics_dense(P, d_omics, cfg["dropout"])
        self.lin_gep, self.lin_cnv, self.lin_mut = make_dense(), make_dense(), make_dense()

        heads = cfg.get("n_heads", 8)
        mk_blk = lambda: _TokenCrossBlock(d_omics, H, heads, cfg["dropout"])
        self.attn_gep, self.attn_cnv, self.attn_mut = mk_blk(), mk_blk(), mk_blk()

        self.mlp = nn.Sequential(
            nn.Linear(H * 4, 512), nn.ReLU(), nn.Dropout(cfg["dropout"]), nn.Linear(512, 1)
        )

    # -------------------------------------------------
    def forward(self, drug_batch, gep, cnv, mut):
        ids = drug_batch["input_ids"].to(self.device)
        mask = drug_batch["attention_mask"].to(self.device)
        toks = self.encoder(ids, mask)             # (B, L, H)
        cls = toks[:, 0]                           # (B, H)
        kpm = mask == 0                            # padding mask

        gep_ctx = self.attn_gep(self.lin_gep(gep.to(self.device)), toks, kpm)
        cnv_ctx = self.attn_cnv(self.lin_cnv(cnv.to(self.device)), toks, kpm)
        mut_ctx = self.attn_mut(self.lin_mut(mut.to(self.device)), toks, kpm)

        y_hat = self.mlp(torch.cat([cls, gep_ctx, cnv_ctx, mut_ctx], dim=1)).squeeze(-1)
        return y_hat, {}

    def loss(self, y_hat, y_true):
        return self.loss_fn(y_hat, y_true)
