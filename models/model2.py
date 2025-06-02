import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Dict, Tuple, List

# 프로젝트 유틸

from utils.utils import get_device, get_log_molar
from utils.hyperparams import LOSS_FN_FACTORY, ACTIVATION_FN_FACTORY

# 이미 encoder.py 안에 구현해 둔 클래스들을 재사용합니다
from yj.encoder import ChemEncoder

# --------------------------------------------------------------------------- #
#  Attention 블록
# --------------------------------------------------------------------------- #
class CrossAttentionBlock(nn.Module):
    """
    • omics 벡터(1×omics_dim)에 SMILES 토큰 임베딩(L×H)을 어텐션.
    • 출력:  (context      : [B, H],
             attn_weights : [B, L])   # 분석/시각화용
    """
    def __init__(self, omics_dim: int, embed_dim: int,
                 n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.q_proj = nn.Linear(omics_dim, embed_dim)
        self.attn   = nn.MultiheadAttention(
            embed_dim, n_heads, dropout=dropout,
            batch_first=True, bias=True
        )

    def forward(self,
                omics_vec: torch.Tensor,
                token_emb: torch.Tensor,
                key_padding_mask: torch.Tensor):
        # Q: [B,1,H],   K/V: [B,L,H]
        q = self.q_proj(omics_vec).unsqueeze(1)
        ctx, w = self.attn(q, token_emb, token_emb,
                           key_padding_mask=key_padding_mask,
                           need_weights=True,
                           average_attn_weights=False)
        return ctx.squeeze(1), w.squeeze(1)            # [B,H], [B,L]


# --------------------------------------------------------------------------- #
#  메인 모델
# --------------------------------------------------------------------------- #
class PASO_GEP_CNV_MUT(nn.Module):
    """
    (1) ChemBERTa-smiles backbone
    (2) GEP / CNV / MUT 벡터를 각각 cross-attention 으로 융합
    (3) MLP → IC50 예측
    """
    def __init__(self, params: Dict):
        super().__init__()
        self.params = params
        self.device = get_device()

        # 1) ──────────────────────────────────────────────────────────── #
        self.chem_encoder = ChemEncoder(
            params.get("tokenizer_name", "DeepChem/ChemBERTa-77M-MLM"),
            freeze=not params.get("train_backbone", False)
        )
        H = self.chem_encoder.hidden_size                # ≒ 384

        # 2) ──────────────────────────────────────────────────────────── #
        n_pw   = params.get("number_of_pathways", 619)
        dense  = params.get("omics_dense_size", 256)
        drop_p = params.get("dropout", 0.5)
        act    = nn.ReLU()

        self.gep_lin = nn.Sequential(nn.Linear(n_pw, dense), act, nn.Dropout(drop_p))
        self.cnv_lin = nn.Sequential(nn.Linear(n_pw, dense), act, nn.Dropout(drop_p))
        self.mut_lin = nn.Sequential(nn.Linear(n_pw, dense), act, nn.Dropout(drop_p))

        heads = params.get("n_heads", 8)
        self.attn_gep = CrossAttentionBlock(dense, H, heads, drop_p)
        self.attn_cnv = CrossAttentionBlock(dense, H, heads, drop_p)
        self.attn_mut = CrossAttentionBlock(dense, H, heads, drop_p)

        # 3) ──────────────────────────────────────────────────────────── #
        mlp_layers: List[nn.Module] = []
        in_dim = H * 4                       # CLS + 3×cross-ctx
        for h in params.get("stacked_dense_hidden_sizes", [1024, 512]):
            mlp_layers += [nn.Linear(in_dim, h), act, nn.Dropout(drop_p)]
            in_dim = h
        self.mlp  = nn.Sequential(*mlp_layers)
        self.head = nn.Linear(in_dim, 1)

        # 4) ──────────────────────────────────────────────────────────── #
        self.loss_fn = LOSS_FN_FACTORY[params.get("loss_fn", "mse")]
        pp = params.get("drug_sensitivity_processing_parameters", {}).get("parameters", {})
        self.ic50_min = pp.get("min", -8.66)
        self.ic50_max = pp.get("max", 13.11)

        self.to(self.device)

    # ------------------------------------------------------------------ #
    def forward(self,
                drug_tokens: Dict[str, torch.Tensor],
                gep: torch.Tensor,
                cnv: torch.Tensor,
                mut: torch.Tensor):
        """
        Parameters
        ----------
        drug_tokens : dict  {input_ids, attention_mask}
        gep / cnv / mut :  [B, pathways]
        Returns
        -------
        y_hat      : [B]  (raw IC50, log10[μM] 스케일 아님)
        pred_dict  : dict {"log_micromolar_IC50": [B]}
        """
        ids   = drug_tokens["input_ids"].to(self.device)
        mask  = drug_tokens["attention_mask"].to(self.device)
        kpm   = (mask == 0)                      # padding positions

        tok_emb = self.chem_encoder(ids, mask)   # [B,L,H]
        cls_tok = tok_emb[:, 0]                  # [B,H]

        gep_ctx, _ = self.attn_gep(self.gep_lin(gep.to(self.device)), tok_emb, kpm)
        cnv_ctx, _ = self.attn_cnv(self.cnv_lin(cnv.to(self.device)), tok_emb, kpm)
        mut_ctx, _ = self.attn_mut(self.mut_lin(mut.to(self.device)), tok_emb, kpm)

        fuse = torch.cat([cls_tok, gep_ctx, cnv_ctx, mut_ctx], dim=1)
        y_hat = self.head(self.mlp(fuse)).squeeze(-1)        # [B]

        log_pred = get_log_molar(y_hat,
                                 ic50_min=self.ic50_min,
                                 ic50_max=self.ic50_max)

        return y_hat, {"log_micromolar_IC50": log_pred.detach()}

    # ------------------------------------------------------------------ #
    # helper wrappers
    # ------------------------------------------------------------------ #
    def loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Loss wrapper so 학습코드에서 model.loss(...) 로 호출 가능."""
        return self.loss_fn(pred.view_as(target), target)

    def save(self, filepath: str | os.PathLike) -> None:
        torch.save(self.state_dict(), filepath)

    @classmethod
    def load(cls, filepath: str | os.PathLike, params: Dict):
        model = cls(params)
        model.load_state_dict(torch.load(filepath, map_location="cpu"))
        return model