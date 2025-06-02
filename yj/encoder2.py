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