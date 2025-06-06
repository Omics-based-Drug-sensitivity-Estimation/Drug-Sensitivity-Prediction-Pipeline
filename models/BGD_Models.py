"""
Graph-based PASO variants (BGD family)
=====================================

v1  – Drug Graph Transformer  + multi-head Context-Attention fusion  
v2  – Drug Graph Transformer  + Cross-Attention blocks

Both models consume the **BGD mini-batch** produced by
`data.tripleomics_drug_dataset.bgd_collate`:

    batch =
        (x, adj, bond),      # graph tensors  (B, N, ..)
        gep, cnv, mut,       # omics matrices (B, 619)
        ic50                 # targets        (B,)

and return:

    preds, aux_dict
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from utils.drug_embedding import DrugEmbeddingModel
from utils.layers import ContextAttentionLayer, dense_layer
from utils.cross_attention import CrossAttentionModule
from utils.hyperparams import LOSS_FN_FACTORY
from utils.utils import get_device

# ────────────────────────────────────────────────────────────────────
# constants / helpers
# ────────────────────────────────────────────────────────────────────
_BOND_TYPES: List[str] = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]


def _triple_to_list(t: torch.Tensor) -> List[torch.Tensor]:
    """Helper: split tuple of three omics tensors into list → loop-friendly."""
    return [t.unsqueeze(-1) for t in t]  # (B, 619, 1)


# ────────────────────────────────────────────────────────────────────
# v1  –  Context-Attention fusion (mirrors ChemBERTa-v1)
# ────────────────────────────────────────────────────────────────────

class PASO_BGD_v1(nn.Module):
    """Graph Transformer + multi-head ContextAttention fusion."""

    def __init__(self, cfg: Dict):
        super().__init__()
        self.cfg = cfg
        self.device = get_device()
        self.loss_fn = LOSS_FN_FACTORY[cfg.get("loss_fn", "mse")]

        # ―― Drug encoder 세팅 ――――――――――――――――――――――――――――――――――――――――――
        self.drug_encoder = DrugEmbeddingModel(
            in_dim=78,
            embed_dim=cfg["smiles_embedding_size"],  # H
            n_heads=8,
            ff_dim=2048,
            n_layers=6,
            n_bond_types=len(_BOND_TYPES),
            max_nodes=cfg["smiles_padding_length"],  # N
        ).to(self.device)

        H = cfg["smiles_embedding_size"]              # hidden dim
        # CLS 포함해서 총 시퀀스 길이는 N + 1
        L = cfg["smiles_padding_length"] + 1          # (예: max_nodes + 1)
        P = cfg.get("number_of_pathways", 619)        # omics 차원 (예: 619)
        d_attn = cfg.get("smiles_attention_size", 64) # ContextAttention 내부 차원

        # ―― per-omics attention heads 수 가져오기
        n_gep = cfg.get("molecule_gep_heads", [2])[0]
        n_cnv = cfg.get("molecule_cnv_heads", [2])[0]
        n_mut = cfg.get("molecule_mut_heads", [2])[0]

        # ── 각 omics별 ContextAttentionLayer 생성
        #    ContextAttentionLayer(reference_hidden_size=H,
        #                          reference_sequence_length=L,
        #                          context_hidden_size=1,
        #                          context_sequence_length=P,
        #                          attention_size=d_attn)
        mk_ctx = lambda: ContextAttentionLayer(H, L, 1, P, d_attn)
        self.ctx_gep = nn.ModuleList([mk_ctx() for _ in range(n_gep)])
        self.ctx_cnv = nn.ModuleList([mk_ctx() for _ in range(n_cnv)])
        self.ctx_mut = nn.ModuleList([mk_ctx() for _ in range(n_mut)])

        # ―― Omics projection to dense representation (B, P) → (B, d_omics)
        d_omics = cfg.get("omics_dense_size", 128)
        self.lin_gep = nn.Sequential(
            nn.Linear(P, d_omics),
            nn.ReLU(),
            nn.Dropout(cfg["dropout"]),
        )
        self.lin_cnv = nn.Sequential(
            nn.Linear(P, d_omics),
            nn.ReLU(),
            nn.Dropout(cfg["dropout"]),
        )
        self.lin_mut = nn.Sequential(
            nn.Linear(P, d_omics),
            nn.ReLU(),
            nn.Dropout(cfg["dropout"]),
        )

        # ―― MLP head 세팅 ――――――――――――――――――――――――――――――――――――――――――
        #   fusion_dim = (n_gep + n_cnv + n_mut) * H + 3 * d_omics
        fusion_dim = (n_gep + n_cnv + n_mut) * H + 3 * d_omics
        hidden = cfg.get("stacked_dense_hidden_sizes", [1024, 512])
        layers: List[nn.Module] = []
        prev = fusion_dim
        for h in hidden:
            layers += [
                nn.Linear(prev, h),
                nn.ReLU(),
                nn.Dropout(cfg["dropout"]),
            ]
            prev = h
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(prev, 1)

        self.to(self.device)

    def forward(self, drug_tuple, gep, cnv, mut):
        """
        Parameters
        ----------
        drug_tuple : (x, adj, bond)   # from bgd_collate
        gep, cnv, mut : [B, P]        # triple-omics
        """
        # 1) 입력 텐서 device 이동
        x, adj, bond = drug_tuple
        x = x.to(self.device)       # (B, N, in_dim)
        adj = adj.to(self.device)   # (B, N, N)
        bond = bond.to(self.device) # (B, N, N)
        gep = gep.to(self.device).float().unsqueeze(-1)  # → (B, P, 1)
        cnv = cnv.to(self.device).float().unsqueeze(-1)  # → (B, P, 1)
        mut = mut.to(self.device).float().unsqueeze(-1)  # → (B, P, 1)

        # 2) Drug 임베딩 (CLS + atom embeddings)
        tok_full = self.drug_encoder(x, adj, bond)  # (B, N+1, H)

        feats: List[torch.Tensor] = []

        # 3) GEP heads: reference = tok_full (B, L, H), context = gep (B, P, 1)
        for attn in self.ctx_gep:
            h, _ = attn(tok_full, gep)  # → h: (B, H)
            feats.append(h)

        # 4) CNV heads
        for attn in self.ctx_cnv:
            h, _ = attn(tok_full, cnv)  # → h: (B, H)
            feats.append(h)

        # 5) MUT heads
        for attn in self.ctx_mut:
            h, _ = attn(tok_full, mut)  # → h: (B, H)
            feats.append(h)

        # 6) Dense omics 표현 (B, P) → (B, d_omics)
        #    원래 gep, cnv, mut은 (B, P, 1) 형태이므로, 채널 차원 제거
        gep_flat = gep.squeeze(-1)  # → (B, P)
        cnv_flat = cnv.squeeze(-1)  # → (B, P)
        mut_flat = mut.squeeze(-1)  # → (B, P)

        dense_gep = self.lin_gep(gep_flat)  # (B, d_omics)
        dense_cnv = self.lin_cnv(cnv_flat)  # (B, d_omics)
        dense_mut = self.lin_mut(mut_flat)  # (B, d_omics)

        feats += [dense_gep, dense_cnv, dense_mut]  # (B, d_omics) 세 개 추가

        # 7) 모든 feature를 dim=1 기준으로 합치기
        #    feats 요소별 shape:
        #     - 각 attention head h: (B, H)
        #     - dense_gep, etc.: (B, d_omics)
        #    torch.cat(feats, dim=1) → (B, (n_heads_total*H + 3*d_omics))
        fusion = torch.cat(feats, dim=1)  # (B, fusion_dim)

        # 8) MLP → 최종 예측값
        y_hat = self.out(self.mlp(fusion)).squeeze(-1)  # (B,)
        return y_hat, {}  # aux dict은 현재 비워둡니다

    def loss(self, y_hat, y_true):
        return self.loss_fn(y_hat, y_true)


# ────────────────────────────────────────────────────────────────────
# v2  – Cross-Attention fusion (mirrors ChemBERTa-v2)
# ────────────────────────────────────────────────────────────────────
class PASO_BGD_v2(nn.Module):
    """Graph Transformer + *CrossAttentionModule* fusion (3 omics blocks)."""

    def __init__(self, cfg: Dict):
        super().__init__()
        self.cfg = cfg
        self.device = get_device()
        self.loss_fn = LOSS_FN_FACTORY[cfg.get("loss_fn", "mse")]

        # ―― Drug encoder ――――――――――――――――――――――――――――――――――――――――――
        self.drug_encoder = DrugEmbeddingModel(
            in_dim=78,
            embed_dim=cfg["smiles_embedding_size"],
            n_heads=cfg.get("n_heads", 8),
            ff_dim=2048,
            n_layers=cfg.get("num_layers", 2),
            n_bond_types=len(_BOND_TYPES),
            max_nodes=cfg["smiles_padding_length"],
        ).to(self.device)

        H = cfg["smiles_embedding_size"]
        mk_cross = lambda: CrossAttentionModule(
            H,
            num_layers=cfg.get("num_layers", 2),
            num_heads=cfg.get("n_heads", 8),
            dropout=cfg["dropout"],
        )
        self.cross_gep = mk_cross()
        self.cross_cnv = mk_cross()
        self.cross_mut = mk_cross()

        self.mlp = nn.Sequential(
            nn.Linear(H * 3, 256), nn.ReLU(), nn.Dropout(cfg["dropout"]), nn.Linear(256, 1)
        )

    # ———————————————————————————— forward ————————————————————————————
    def forward(self, drug_tuple, gep, cnv, mut):
        """
        Parameters
        ----------
        drug_tuple : (x, adj, bond)   # from bgd_collate
        gep, cnv, mut : [B, 619]      # triple-omics
        """
        x, adj, bond = drug_tuple
        x, adj, bond = x.to(self.device), adj.to(self.device), bond.to(self.device)
        gep, cnv, mut = [t.to(self.device).float() for t in (gep, cnv, mut)]

        tok = self.drug_encoder(x, adj, bond)          # CLS + atom embeddings
        g, _ = self.cross_gep(tok, gep)
        c, _ = self.cross_cnv(tok, cnv)
        m, _ = self.cross_mut(tok, mut)
        y_hat = self.mlp(torch.cat([g, c, m], dim=1)).squeeze(-1)
        return y_hat, {}

    def loss(self, y_hat, y_true):
        return self.loss_fn(y_hat, y_true)
