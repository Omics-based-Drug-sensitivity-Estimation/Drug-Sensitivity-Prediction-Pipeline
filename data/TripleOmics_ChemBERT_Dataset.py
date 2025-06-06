"""tripleomics_chemberta_dataset.py
================================================
ChemBERTa‑Omics Dataset (v2.1)
------------------------------------------------
* PyTorch `Dataset` coupling **ChemBERTa SMILES** with **triple‑omics** (GEP/CNV/MUT) + IC50.
* Robust CSV handling inspired by the user's working `TripleOmics_Drug_dataset`.
* **NEW in v2.1** – gracefully overwrites *pre‑existing* `SMILES` column in `drug_sens_csv` to stop the
  `AttributeError: 'DataFrame' object has no attribute 'str'` that occurs when pandas creates
  duplicate columns during merge.

Quick start
-----------
```python
from tripleomics_chemberta_dataset import ChemBERTaOmicsDataset, chemberta_collate
train_ds = ChemBERTaOmicsDataset(
    "MixedSet_train_Fold0.csv", "CCLE-GDSC-SMILES.csv",
    "GEP.csv", "CNV.csv", "MUT.csv",
    max_len=256, case="upper")
```
`__getitem__` ⇒
```python
{
  "input_ids": Tensor[L],  # SMILES token IDs
  "attention_mask": Tensor[L],
  "gep": Tensor[619],
  "cnv": Tensor[619],
  "mut": Tensor[619],
  "ic50": Tensor[1]
}
```
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional

import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from transformers import AutoTokenizer

__all__ = ["ChemBERTaOmicsDataset", "chemberta_collate"]

log = logging.getLogger(__name__)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------

def _standardise(df: pd.DataFrame) -> pd.DataFrame:
    return (df - df.mean()) / (df.std(ddof=0) + 1e-9)


def _clean_index(idx: pd.Index, case: str) -> pd.Index:
    idx = idx.astype(str).str.strip()
    if case == "upper":
        idx = idx.str.upper()
    elif case == "lower":
        idx = idx.str.lower()
    return idx


# ---------------------------------------------------------------------
# main dataset
# ---------------------------------------------------------------------
class ChemBERTaOmicsDataset(Dataset):
    """ChemBERTa + triple‑omics → IC50 (PyTorch Dataset).

    ‑ Robust CSV handling (header auto‑detect, whitespace trim).
    ‑ Optional omics Z‑score, IC50 min‑max.
    ‑ One‑shot SMILES tokenisation for fast training.
    """

    def __init__(
        self,
        drug_sens_csv: Union[str, Path],
        smiles_csv: Union[str, Path],
        gep_csv: Union[str, Path],
        cnv_csv: Union[str, Path],
        mut_csv: Union[str, Path],
        *,
        tokenizer_name: str = "DeepChem/ChemBERTa-77M-MLM",
        max_len: int = 256,
        case: str = "upper",
        standardise_omics: Optional[Dict[str, bool]] = None,
        minmax_ic50: bool = True,
    ) -> None:
        super().__init__()
        self.max_len = max_len
        self.tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

        # 1) load & clean tables ---------------------------------------
        sens = self._load_sens(drug_sens_csv, case)
        smiles_df = self._load_smiles(smiles_csv)

        # if sens already has SMILES → drop
        if "SMILES" in sens.columns:
            log.warning("Found 'SMILES' in drug_sens_csv – dropping & replacing with canonical column from smiles_csv.")
            sens = sens.drop(columns=["SMILES"])

        # merge (m:1)
        sens = sens.merge(smiles_df, on="drug", how="left", validate="m:1")

        # ── handle duplicate column names (Edge‑case) ─────────────────--
        if sens.columns.duplicated().any():
            dups = sens.columns[sens.columns.duplicated()].unique()
            log.warning("Duplicate columns after merge: %s – keeping first occurrence.", dups.tolist())
            sens = sens.loc[:, ~sens.columns.duplicated()].copy()

        # tidy SMILES
        sens["SMILES"] = sens["SMILES"].astype(str).str.strip()
        bad = sens["SMILES"].isin(["", "nan", "NaN", "None"])
        if bad.any():
            raise ValueError(f"Empty SMILES for {bad.sum()} rows – fix input CSVs.")

        # IC50 scaling
        if minmax_ic50:
            lo, hi = sens.IC50.min(), sens.IC50.max()
            sens.IC50 = (sens.IC50 - lo) / (hi - lo)
            self.ic50_min, self.ic50_max = float(lo), float(hi)
        else:
            self.ic50_min = self.ic50_max = None

        # 2) omics ------------------------------------------------
        # standardise_omics:  
        #   - None  → default {"gep": True, "cnv": True, "mut": True}
        #   - bool  → apply same flag to all three (True/False)
        #   - dict  → per‑omics overrides, e.g. {"cnv": False}
        std = {"gep": True, "cnv": True, "mut": True}
        if isinstance(standardise_omics, bool):
            std = {k: standardise_omics for k in std}
        elif isinstance(standardise_omics, dict):
            std.update(standardise_omics)

        gep = self._load_omics(gep_csv, case, std["gep"])
        cnv = self._load_omics(cnv_csv, case, std["cnv"])
        mut = self._load_omics(mut_csv, case, std["mut"])(mut_csv, case, std["mut"])

        # 3) build rows
        cell2i = {cl: i for i, cl in enumerate(gep.index)}
        self.rows = [
            (row.SMILES, cell2i[row.cell_line], float(row.IC50))
            for _, row in sens.iterrows() if row.cell_line in cell2i
        ]
        if not self.rows:
            raise RuntimeError("No samples after matching cell lines → omics.")

        self.gep_arr = gep.to_numpy(dtype="float32")
        self.cnv_arr = cnv.to_numpy(dtype="float32")
        self.mut_arr = mut.to_numpy(dtype="float32")

        # 4) tokenise once
        enc = self.tok(
            [s for s, _, _ in self.rows],
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
            add_special_tokens=True,
        )
        self.input_ids = enc["input_ids"]
        self.attn_masks = enc["attention_mask"]

    # helper loaders ----------------------------------------------------
    @staticmethod
    def _load_smiles(path: str | Path) -> pd.DataFrame:
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        if set(df.columns) == {0, 1}:
            df.columns = ["drug", "SMILES"]
        else:
            for cand in ("drug", "Drug", "DRUG", "DRUG_NAME"):
                if cand in df.columns:
                    df = df.rename(columns={cand: "drug"})
                    break
            else:
                raise KeyError("[SMILES CSV] drug column not found")
            non_drug = [c for c in df.columns if c != "drug"]
            df = df.rename(columns={non_drug[0]: "SMILES"})
        return df[["drug", "SMILES"]]

    @staticmethod
    def _load_sens(path: str | Path, case: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        for cand in ("drug", "Drug", "DRUG", "DRUG_NAME"):
            if cand in df.columns and cand != "drug":
                df = df.rename(columns={cand: "drug"})
        for cand in ("cell_line", "CELL_LINE", "Cell_line", "CELL"):
            if cand in df.columns and cand != "cell_line":
                df = df.rename(columns={cand: "cell_line"})
        df["cell_line"] = _clean_index(df["cell_line"], case)
        required = {"drug", "cell_line", "IC50"}
        if not required.issubset(df.columns):
            raise KeyError(f"drug_sens_csv missing columns: {required - set(df.columns)}")
        return df[list(required)]

    @staticmethod
    def _load_omics(path: str | Path, case: str, do_std: bool) -> pd.DataFrame:
        df = pd.read_csv(path, index_col=0)
        df.index = _clean_index(df.index, case)
        return _standardise(df) if do_std else df

    # protocol ----------------------------------------------------------
    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int):
        sm, ci, ic = self.rows[idx]
        return {
            "input_ids":      self.input_ids[idx],
            "attention_mask": self.attn_masks[idx],
            "gep":            torch.from_numpy(self.gep_arr[ci]),
            "cnv":            torch.from_numpy(self.cnv_arr[ci]),
            "mut":            torch.from_numpy(self.mut_arr[ci]),
            "ic50":           torch.tensor(ic, dtype=torch.float32),
        }

def chemberta_collate(batch: List[Dict[str, torch.Tensor]]):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}