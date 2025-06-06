

"""tripleomics_bgd_dataset.py
================================================
Graph‑based Drug Embedding + Triple‑Omics Dataset  **v4.0**
----------------------------------------------------------
### What’s fixed
1. **SMILES column auto‑detect:** first column whose name contains "SMILE" (case‑insensitive). If none, fallback to the first *non‑drug* column. → `I‑BRD9` 누락 문제 해결.
2. **Drug key normalisation** shared by *both* CSVs via `_norm_drug()`.
3. **Mapping duplicates** logged & first instance kept.
4. Complete `__getitem__` + `bgd_collate` restored.

This file should now run end‑to‑end without `KeyError: SMILES not found`.
"""

from __future__ import annotations

import logging, re
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional

import deepchem as dc
import pandas as pd
import torch
from rdkit import Chem
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

__all__ = ["TripleOmics_Drug_dataset", "bgd_collate"]

log = logging.getLogger(__name__)
dc_logger = logging.getLogger("deepchem")
dc_logger.setLevel(logging.ERROR)

# ───────────────────── constants ────────────────────────
BOND_TYPES = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]
BTYPE2IDX = {b: i + 1 for i, b in enumerate(BOND_TYPES)}
MAX_NODES = 128

# ───────────────────── drug‑key normaliser ───────────────
_dash = re.compile(r"[\u2010-\u2015\u2212]")
_nonvalid = re.compile(r"[^A-Z0-9\-_]")

def _norm_drug(name: str) -> str:
    name = _dash.sub("-", name.upper()).replace(" ", "")
    return _nonvalid.sub("", name)

# ───────────────────── padding helpers ───────────────────

def _pad_vec(x: torch.Tensor, length: int):
    # cast to float32 to avoid dtype mismatch with Linear layers
    x = x.float()
    y = torch.zeros(length, *x.shape[1:], dtype=torch.float32)
    y[: x.shape[0]] = x
    return y


def _pad_mat(m: torch.Tensor, size: int):
    y = torch.zeros(size, size, dtype=m.dtype)
    n = m.shape[0]
    y[:n, :n] = m
    return y

# ───────────────────── graph helpers ─────────────────────

def _adj_from_list(adj_list: List[List[int]], n: int):
    adj = torch.eye(n)
    for u, nbr in enumerate(adj_list):
        for v in nbr:
            adj[u, v] = adj[v, u] = 1.0
    return adj


def _bond_matrix(mol: Chem.Mol, n: int):
    mat = torch.zeros(n, n)
    for b in mol.GetBonds():
        u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        mat[u, v] = mat[v, u] = BTYPE2IDX.get(str(b.GetBondType()), 0)
    return mat

# ───────────────────── dataset ───────────────────────────
class TripleOmics_Drug_dataset(Dataset):
    """Graph drug embedding + triple‑omics → IC50."""

    def __init__(
        self,
        drug_sens_csv: Union[str, Path],
        smiles_csv: Union[str, Path],
        gep_csv: Union[str, Path],
        cnv_csv: Union[str, Path],
        mut_csv: Union[str, Path],
        *,
        standardise_omics: Optional[Union[bool, Dict[str, bool]]] = True,
        minmax_ic50: bool = True,
        cols: Tuple[str, str, str] = ("drug", "cell_line", "IC50"),
    ):
        super().__init__()
        d_col, c_col, ic_col = cols

        # 1) drug‑sensitivity -------------------------------------------
        sens = pd.read_csv(drug_sens_csv)
        sens.columns = sens.columns.str.strip()
        sens[d_col] = sens[d_col].astype(str).map(_norm_drug)
        sens[c_col] = sens[c_col].astype(str).str.strip().str.upper()
        if minmax_ic50:
            lo, hi = sens[ic_col].min(), sens[ic_col].max()
            sens[ic_col] = (sens[ic_col] - lo) / (hi - lo + 1e-9)
        self.sens = sens
        self.drug_col, self.cell_col, self.ic50_col = d_col, c_col, ic_col

        # 2) SMILES ------------------------------------------------------
        smi_df = pd.read_csv(smiles_csv)
        smi_df.columns = smi_df.columns.str.strip()
        drug_col = next((c for c in ("drug", "Drug", "DRUG", "DRUG_NAME") if c in smi_df.columns), smi_df.columns[0])
        smi_df = smi_df.rename(columns={drug_col: "drug"})
        # pick SMILES column ↴
        smiles_col = next((c for c in smi_df.columns if "smile" in c.lower()), None)
        if smiles_col is None:
            smiles_col = [c for c in smi_df.columns if c != "drug"][0]
        smi_df = smi_df.rename(columns={smiles_col: "SMILES"})
        smi_df["drug"] = smi_df["drug"].astype(str).map(_norm_drug)

        _drug2smi: Dict[str, str] = {}
        for d, s in zip(smi_df.drug, smi_df.SMILES):
            if d in _drug2smi:
                log.warning("Duplicate drug '%s' in SMILES table – keeping first entry.", d)
                continue
            _drug2smi[d] = s
        self._drug2smi = _drug2smi

        # 3) omics -------------------------------------------------------
        std = {"gep": True, "cnv": True, "mut": True}
        if isinstance(standardise_omics, bool):
            std = {k: standardise_omics for k in std}
        elif isinstance(standardise_omics, dict):
            std.update(standardise_omics)
        self.gep = self._load_omics(gep_csv, std["gep"])
        self.cnv = self._load_omics(cnv_csv, std["cnv"])
        self.mut = self._load_omics(mut_csv, std["mut"])
        self._cell2idx = {cl: i for i, cl in enumerate(self.gep.index)}

        # 4) featuriser + cache -----------------------------------------
        self._featuriser = dc.feat.graph_features.ConvMolFeaturizer(use_chirality=True)
        self._cache: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    # helpers -----------------------------------------------------------
    def _load_omics(self, path: str | Path, do_std: bool):
        df = pd.read_csv(path, index_col=0)
        df.index = df.index.astype(str).str.strip().str.upper()
        if do_std:
            df.iloc[:] = StandardScaler().fit_transform(df)
        return df

    def _encode(self, d_key: str):
        if d_key in self._cache:
            return self._cache[d_key]
        smi = self._drug2smi.get(d_key)
        if smi is None:
            raise KeyError(f"SMILES not found for drug '{d_key}'. SMILES keys sample: {list(self._drug2smi)[:5]} …")
        mol = Chem.MolFromSmiles(smi)
        molgraph = self._featuriser.featurize([mol])[0]
        feats = torch.from_numpy(molgraph.atom_features).float()
        n = feats.shape[0]
        adj = _adj_from_list(molgraph.canon_adj_list, n)
        bond = _bond_matrix(mol, n)
        self._cache[d_key] = (feats, adj, bond)
        return feats, adj, bond

    # Dataset ----------------------------------------------------------
    def __len__(self):
        return len(self.sens)

    def __getitem__(self, idx):
        row = self.sens.iloc[idx]
        d_key = row[self.drug_col]
        cell = row[self.cell_col]
        feats, adj, bond = self._encode(d_key)
        feats = _pad_vec(feats, MAX_NODES)
        adj = _pad_mat(adj, MAX_NODES)
        bond = _pad_mat(bond, MAX_NODES)
        ci = self._cell2idx.get(cell)
        if ci is None:
            raise KeyError(f"Cell-line '{cell}' missing in omics index.")
        gep = torch.tensor(self.gep.iloc[ci].values, dtype=torch.float32)
        cnv = torch.tensor(self.cnv.iloc[ci].values, dtype=torch.float32)
        mut = torch.tensor(self.mut.iloc[ci].values, dtype=torch.float32)
        ic50 = torch.tensor(row[self.ic50_col], dtype=torch.float32)
        return (feats, adj, bond), gep, cnv, mut, ic50

# ───────────────────── collate fn ─────────────────────

def bgd_collate(batch: List):
    drug_t, gep, cnv, mut, ic50 = zip(*batch)
    x, adj, bond = zip(*drug_t)
    return (
        torch.stack(x),
        torch.stack(adj),
        torch.stack(bond),
    ), torch.stack(gep), torch.stack(cnv), torch.stack(mut), torch.stack(ic50)
