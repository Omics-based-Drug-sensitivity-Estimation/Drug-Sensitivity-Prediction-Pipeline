
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
from typing import Tuple, Dict
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
"""ChemBERTa‑Omics dataset
==========================
A drop‑in replacement for the original ``ChemBERTaOmicsDataset`` that:

*   **Auto‑cleans cell‑line names** (strip → upper‑case) so mismatched IDs like
    ``22Rv1``/``22RV1`` no longer raise ``KeyError``.
*   **Gracefully handles missing cell lines** – missing rows are *dropped* with
    a warning instead of crashing.  Set ``raise_missing=True`` if you prefer
    the old behaviour.
*   Keeps the **public API identical** – same init signature and **identical
    output dict** from ``__getitem__`` – so you do **not** need to change any
    training code.

The only new *optional* parameter is ``case`` ("upper" or "lower").  Default
behaves exactly like before ("upper").
"""

from pathlib import Path
from typing import Dict, List, Tuple, Union
import os

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _standardize(df: pd.DataFrame) -> pd.DataFrame:
    """Z‑score standardisation (μ=0, σ=1)."""
    return (df - df.mean()) / (df.std(ddof=0) + 1e-9)


def _clean_index(idx: pd.Index, case: str) -> pd.Index:
    idx = idx.astype(str).str.strip()
    if case == "upper":
        idx = idx.str.upper()
    elif case == "lower":
        idx = idx.str.lower()
    return idx


# ---------------------------------------------------------------------------
# Main dataset
# ---------------------------------------------------------------------------

class ChemBERTaOmicsDataset(Dataset):
    """Tokenised SMILES + triple‑omics → IC50.

    *Input / output tensors remain *identical* to the original implementation.*
    """

    def __init__(
        self,
        drug_sensitivity_csv: Union[str, Path],
        smiles_csv: Union[str, Path],
        gep_csv: Union[str, Path],
        cnv_csv: Union[str, Path],
        mut_csv: Union[str, Path],
        tokenizer_name: str = "DeepChem/ChemBERTa-77M-MLM",
        max_len: int = 256,
        gep_standardize: bool = True,
        cnv_standardize: bool = True,
        mut_standardize: bool = True,
        *,
        case: str = "upper",          # NEW – case normalisation ("upper"/"lower")
        drop_missing: bool = True,      # NEW – drop rows whose cell‑line omics are missing
        raise_missing: bool = False,    #   └─ if True, override drop_missing and raise
    ) -> None:
        super().__init__()

        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

        # ------------------------------------------------------------------
        # 1) Drug sensitivity + SMILES merge
        # ------------------------------------------------------------------
        sens = pd.read_csv(drug_sensitivity_csv)  # columns: drug, cell_line, IC50
        sens["cell_line"] = _clean_index(sens["cell_line"], case)

        smiles_df = pd.read_csv(smiles_csv)
        smiles_df.columns = (
            smiles_df.columns.str.strip().str.replace("'|\"", "", regex=True)
        )
        for cand in ["drug", "DRUG_NAME", "Drug", "DRUG"]:
            if cand in smiles_df.columns:
                smiles_df = smiles_df.rename(columns={cand: "drug"})
                break
        else:
            raise KeyError("smiles_csv 파일에서 약물명을 나타내는 칼럼을 찾지 못했습니다.")

        sens = sens.merge(smiles_df, on="drug", how="left", validate="m:1")
        if sens["SMILES"].isna().any():
            miss = sens[sens["SMILES"].isna()]["drug"].unique()
            raise ValueError(f"SMILES not found for drugs: {miss[:10]} …")

        # ------------------------------------------------------------------
        # 2) Omics matrices
        # ------------------------------------------------------------------
        gep = pd.read_csv(gep_csv, index_col=0)
        cnv = pd.read_csv(cnv_csv, index_col=0)
        mut = pd.read_csv(mut_csv, index_col=0)

        for df in (gep, cnv, mut):
            df.index = _clean_index(df.index, case)

        if gep_standardize:
            gep = _standardize(gep)
        if cnv_standardize:
            cnv = _standardize(cnv)
        if mut_standardize:
            mut = _standardize(mut)

        # ------------------------------------------------------------------
        # 3) Build row list – drop or raise if cell line missing
        # ------------------------------------------------------------------
        cell2idx = {cl: i for i, cl in enumerate(gep.index)}
        rows: List[Tuple[str, int, float]] = []
        missing: set[str] = set()

        for _, r in sens.iterrows():
            cl = r["cell_line"]
            if cl not in cell2idx:
                missing.add(cl)
                continue
            rows.append((r["SMILES"], cell2idx[cl], r["IC50"]))

        if missing:
            msg = f"{len(missing)} cell lines not found in omics: {sorted(missing)[:10]} …"
            if raise_missing:
                raise KeyError(msg)
            if drop_missing:
                print("⚠️", msg, "→ 해당 샘플은 제외하고 계속 진행합니다.")
            else:
                # keep but will KeyError later – mirror old behaviour
                for cl in missing:
                    rows.append((None, None, None))  # placeholder to provoke crash later

        assert rows, "No training samples after filtering – check data files."
        self.rows = rows

        # ------------------------------------------------------------------
        # 4) Cache numpy + tokenise
        # ------------------------------------------------------------------
        self.gep_arr = gep.to_numpy(dtype="float32")
        self.cnv_arr = cnv.to_numpy(dtype="float32")
        self.mut_arr = mut.to_numpy(dtype="float32")

        smiles_list = [r[0] for r in rows]
        self.input_ids, self.attn_masks = self._pretokenize(smiles_list)

    # --------------------------------------------------------
    # Private helpers
    # --------------------------------------------------------
    def _pretokenize(self, smiles_list: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        enc = self.tokenizer(
            smiles_list,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
            add_special_tokens=True,
        )
        return enc["input_ids"], enc["attention_mask"]

    # --------------------------------------------------------
    # PyTorch Dataset API
    # --------------------------------------------------------
    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        smiles, cell_idx, ic50 = self.rows[idx]
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attn_masks[idx],
            "gep": torch.from_numpy(self.gep_arr[cell_idx]),
            "cnv": torch.from_numpy(self.cnv_arr[cell_idx]),
            "mut": torch.from_numpy(self.mut_arr[cell_idx]),
            "ic50": torch.tensor(ic50, dtype=torch.float32),
        }

# -----------------------------------------------------------------
# collate_fn  –  (동일 shape이라 그저 stack)
# -----------------------------------------------------------------
def chemberta_collate(batch):
    out = {}
    for k in batch[0]:
        out[k] = torch.stack([b[k] for b in batch])
    return out

# Imports & helper functions
import random, math, json, logging, shutil, gc, os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.cuda.amp import GradScaler, autocast
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr
from tqdm.auto import tqdm

# ------------------------------------------------------------
# Model blocks (unchanged in spirit, but hidden size read dynamically)
# ------------------------------------------------------------
class ChemEncoder(nn.Module):
    """Token-level encoder (no pooling)."""
    def __init__(self, model_name: str, freeze: bool = True):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        if freeze:
            self.backbone.requires_grad_(False)
        self.hidden_size = self.backbone.config.hidden_size  # e.g. 384

    def forward(self, ids, mask):
        return self.backbone(input_ids=ids, attention_mask=mask).last_hidden_state

class CrossAttentionBlock(nn.Module):
    def __init__(self, omics_dim: int, embed_dim: int, n_heads: int = 8, p: float = 0.1):
        super().__init__()
        self.q_proj = nn.Linear(omics_dim, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=p, batch_first=True)

    def forward(self, omics_vec, token_emb, kpm):
        q = self.q_proj(omics_vec).unsqueeze(1)        # [B,1,H]
        ctx, w = self.attn(q, token_emb, token_emb, key_padding_mask=kpm,
                           need_weights=True, average_attn_weights=False)
        return ctx.squeeze(1), w.squeeze(1)            # [B,H], [B,L]

class PASO_CrossAttention(nn.Module):
    def __init__(
        self,
        model_name: str,
        n_pathways: int = 619,
        omics_dense: int = 256,
        hidden_sizes: Tuple[int] = (1024, 512),
        dropout: float = 0.5,
        cross_heads: int = 8,
        train_backbone: bool = False,
    ):
        super().__init__()
        self.device = get_device()
        self.encoder = ChemEncoder(model_name, freeze=not train_backbone)
        H = self.encoder.hidden_size                    # ← dynamic

        act, drop = nn.ReLU(), nn.Dropout(dropout)
        self.gep_lin = nn.Sequential(nn.Linear(n_pathways, omics_dense), act, drop)
        self.cnv_lin = nn.Sequential(nn.Linear(n_pathways, omics_dense), act, drop)
        self.mut_lin = nn.Sequential(nn.Linear(n_pathways, omics_dense), act, drop)

        self.attn_gep = CrossAttentionBlock(omics_dense, H, cross_heads, dropout)
        self.attn_cnv = CrossAttentionBlock(omics_dense, H, cross_heads, dropout)
        self.attn_mut = CrossAttentionBlock(omics_dense, H, cross_heads, dropout)

        mlp_layers = []
        in_dim = H * 4
        for h in hidden_sizes:
            mlp_layers += [nn.Linear(in_dim, h), act, drop]
            in_dim = h
        self.mlp = nn.Sequential(*mlp_layers)
        self.out = nn.Linear(in_dim, 1)
        self.to(self.device)

    def forward(self, batch: Dict[str, torch.Tensor]):
        ids, mask = batch["input_ids"].to(self.device), batch["attention_mask"].to(self.device)
        gep = self.gep_lin(batch["gep"].to(self.device))
        cnv = self.cnv_lin(batch["cnv"].to(self.device))
        mut = self.mut_lin(batch["mut"].to(self.device))

        toks = self.encoder(ids, mask)                 # [B,L,H]
        cls = toks[:, 0]                               # [B,H]
        kpm = (mask == 0)

        gep_ctx, a_gep = self.attn_gep(gep, toks, kpm)
        cnv_ctx, a_cnv = self.attn_cnv(cnv, toks, kpm)
        mut_ctx, a_mut = self.attn_mut(mut, toks, kpm)
        fuse = torch.cat([cls, gep_ctx, cnv_ctx, mut_ctx], dim=1)
        pred = self.out(self.mlp(fuse)).squeeze(-1)

        if not self.training:
            batch["attn_gep"], batch["attn_cnv"], batch["attn_mut"] = (
                a_gep.cpu(), a_cnv.cpu(), a_mut.cpu()
            )
        return pred
    
    
    
from typing import Union, List, Tuple, Dict
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

def _clean_index(idx, case: str):
    # (기존 구현 재사용)
    if case == "upper":
        return idx.str.upper().str.strip()
    else:
        return idx.str.lower().str.strip()

def _standardize(df: pd.DataFrame) -> pd.DataFrame:
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    arr = scaler.fit_transform(df.values)
    return pd.DataFrame(arr, index=df.index, columns=df.columns)

class ChemBERTaOmicsDataset(Dataset):
    """Tokenised SMILES + triple-omics → IC50 (min–max scaled)."""

    def __init__(
        self,
        drug_sensitivity_csv: Union[str, Path],
        smiles_csv: Union[str, Path],
        gep_csv: Union[str, Path],
        cnv_csv: Union[str, Path],
        mut_csv: Union[str, Path],
        tokenizer_name: str = "DeepChem/ChemBERTa-77M-MLM",
        max_len: int = 256,
        gep_standardize: bool = True,
        cnv_standardize: bool = True,
        mut_standardize: bool = True,
        drug_sensitivity_min_max: bool = True,      # ← 추가
        *,
        case: str = "upper",
        drop_missing: bool = True,
        raise_missing: bool = False,
    ) -> None:
        super().__init__()
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

        # ------------------------------------------------------------------
        # 1) Drug sensitivity (+IC50 정규화) + SMILES merge
        # ------------------------------------------------------------------
        sens = pd.read_csv(drug_sensitivity_csv)  # columns: drug, cell_line, IC50

        # IC50 min–max scaling
        if drug_sensitivity_min_max:
            ic50_min = sens["IC50"].min()
            ic50_max = sens["IC50"].max()
            sens["IC50"] = (sens["IC50"] - ic50_min) / (ic50_max - ic50_min)
            self.drug_sensitivity_processing_parameters = {
                "processing": "min_max",
                "parameters": {"min": float(ic50_min), "max": float(ic50_max)},
            }
        else:
            self.drug_sensitivity_processing_parameters = {}

        sens["cell_line"] = _clean_index(sens["cell_line"], case)

        smiles_df = pd.read_csv(smiles_csv)
        smiles_df.columns = (
            smiles_df.columns.str.strip().str.replace("'|\"", "", regex=True)
        )
        for cand in ["drug", "DRUG_NAME", "Drug", "DRUG"]:
            if cand in smiles_df.columns:
                smiles_df = smiles_df.rename(columns={cand: "drug"})
                break
        else:
            raise KeyError("smiles_csv 파일에서 약물명을 나타내는 칼럼을 찾지 못했습니다.")

        sens = sens.merge(smiles_df, on="drug", how="left", validate="m:1")
        if sens["SMILES"].isna().any():
            miss = sens[sens["SMILES"].isna()]["drug"].unique()
            raise ValueError(f"SMILES not found for drugs: {miss[:10]} …")

        # ------------------------------------------------------------------
        # 2) Omics matrices
        # ------------------------------------------------------------------
        gep = pd.read_csv(gep_csv, index_col=0)
        cnv = pd.read_csv(cnv_csv, index_col=0)
        mut = pd.read_csv(mut_csv, index_col=0)
        for df in (gep, cnv, mut):
            df.index = _clean_index(df.index, case)
        if gep_standardize:
            gep = _standardize(gep)
        if cnv_standardize:
            cnv = _standardize(cnv)
        if mut_standardize:
            mut = _standardize(mut)

        # ------------------------------------------------------------------
        # 3) Build row list – drop or raise if cell line missing
        # ------------------------------------------------------------------
        cell2idx = {cl: i for i, cl in enumerate(gep.index)}
        rows: List[Tuple[str, int, float]] = []
        missing: set[str] = set()
        for _, r in sens.iterrows():
            cl = r["cell_line"]
            if cl not in cell2idx:
                missing.add(cl)
                continue
            rows.append((r["SMILES"], cell2idx[cl], r["IC50"]))
        if missing:
            msg = f"{len(missing)} cell lines not found in omics: {sorted(missing)[:10]} …"
            if raise_missing:
                raise KeyError(msg)
            if drop_missing:
                print("⚠️", msg, "→ 해당 샘플은 제외하고 계속 진행합니다.")
        assert rows, "No training samples after filtering – check data files."
        self.rows = rows

        # ------------------------------------------------------------------
        # 4) Cache numpy + tokenise
        # ------------------------------------------------------------------
        self.gep_arr = gep.to_numpy(dtype="float32")
        self.cnv_arr = cnv.to_numpy(dtype="float32")
        self.mut_arr = mut.to_numpy(dtype="float32")
        smiles_list = [r[0] for r in rows]
        self.input_ids, self.attn_masks = self._pretokenize(smiles_list)

    def _pretokenize(self, smiles_list: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        enc = self.tokenizer(
            smiles_list,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
            add_special_tokens=True,
        )
        return enc["input_ids"], enc["attention_mask"]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        smiles, cell_idx, ic50 = self.rows[idx]
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attn_masks[idx],
            "gep": torch.from_numpy(self.gep_arr[cell_idx]),
            "cnv": torch.from_numpy(self.cnv_arr[cell_idx]),
            "mut": torch.from_numpy(self.mut_arr[cell_idx]),
            "ic50": torch.tensor(ic50, dtype=torch.float32),
        }

def chemberta_collate(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}