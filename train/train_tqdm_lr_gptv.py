"""Training script for PASO_GEP_CNV_MUT
================================================
Key features
------------
1. **ReduceLROnPlateau** scheduler (val‑loss monitor)
2. **Early stopping** with configurable patience
3. **Attention heat‑map export** every *N* epochs (`attn_viz_interval`)
4. Self‑contained config via `build_cfg()`
5. Case‑insensitive optimiser lookup
"""
import sys
import os
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from yj.encoder import ChemBERTaOmicsDataset, chemberta_collate
from models.model import PASO_GEP_CNV_MUT
from utils.hyperparams import OPTIMIZER_FACTORY
from utils.loss_functions import pearsonr, r2_score
from utils.utils import get_device, get_log_molar

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def ensure_dir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)


class EarlyStopping:
    """Simple early‑stopping utility."""

    def __init__(self, patience: int = 20, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best: float | None = None
        self.counter = 0

    def step(self, value: float) -> bool:
        if self.best is None or value < self.best - self.min_delta:
            self.best = value
            self.counter = 0
            return False  # keep training
        self.counter += 1
        return self.counter >= self.patience


# -----------------------------------------------------------------------------
# Attention visualisation
# -----------------------------------------------------------------------------

def _heatmap(tensor: torch.Tensor, title: str, dest: Path):
    plt.figure(figsize=(10, 6))
    sns.heatmap(tensor, cmap="viridis")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(dest)
    plt.close()


def save_attention(pred_dict: Dict[str, Any], out_dir: Path, epoch: int, batch_idx: int = 0):
    ensure_dir(out_dir)
    for key, val in pred_dict.items():
        if not key.endswith("_attn"):
            continue
        attn = val[batch_idx].detach().cpu()
        _heatmap(attn, f"{key} | epoch {epoch}", out_dir / f"{key}_epoch{epoch}.png")


# -----------------------------------------------------------------------------
# Epoch loops
# -----------------------------------------------------------------------------

def train_epoch(model, loader, optim, device, ic50_max: float, ic50_min: float) -> float:
    model.train()
    total = 0.0
    pbar = tqdm(loader, desc="Train", leave=False)
    for batch in pbar:
        optim.zero_grad()
        drug = {k: v.to(device) for k, v in batch.items() if k in {"input_ids", "attention_mask"}}
        y_hat, _ = model(
            drug,
            batch["gep"].to(device),
            batch["cnv"].to(device),
            batch["mut"].to(device),
        )
        loss = model.loss(y_hat, batch["ic50"].to(device))
        loss.backward()
        clip_grad_norm_(model.parameters(), 2.0)
        optim.step()
        total += loss.item()
        pbar.set_postfix({"loss": f"{total/(pbar.n+1):.4f}"})
    pbar.close()
    return total / len(loader)


def eval_epoch(model, loader, device, ic50_max: float, ic50_min: float):
    model.eval()
    loss_sum = 0.0
    preds, labels = [], []
    with torch.no_grad():
        pbar = tqdm(loader, desc="Val", leave=False)
        for batch in pbar:
            drug = {k: v.to(device) for k, v in batch.items() if k in {"input_ids", "attention_mask"}}
            y_hat, p_dict = model(
                drug,
                batch["gep"].to(device),
                batch["cnv"].to(device),
                batch["mut"].to(device),
            )
            loss = model.loss(y_hat, batch["ic50"].to(device))
            loss_sum += loss.item()
            if p_dict.get("log_micromolar_IC50") is not None:
                preds.append(p_dict["log_micromolar_IC50"].cpu())
            labels.append(get_log_molar(batch["ic50"], ic50_max=ic50_max, ic50_min=ic50_min).cpu())
            pbar.set_postfix({"loss": f"{loss_sum/(pbar.n+1):.4f}"})
        pbar.close()
    preds_cat = torch.cat(preds)
    labels_cat = torch.cat(labels)
    pear = pearsonr(preds_cat, labels_cat)
    rmse = torch.sqrt(torch.mean((preds_cat - labels_cat) ** 2))
    r2 = r2_score(preds_cat, labels_cat)
    return loss_sum / len(loader), pear, rmse, r2, preds_cat, labels_cat


# -----------------------------------------------------------------------------
# Train routine
# -----------------------------------------------------------------------------

def train(cfg: Dict[str, Any]):
    torch.backends.cudnn.benchmark = True
    device = get_device()
    logger.info("Device: %s", device)

    with open(cfg["gene_filepath"], "rb") as f:
        pathways: List[str] = pickle.load(f)

    folds = cfg["params"].get("fold", 1)
    logger.info("%d‑fold CV", folds)

    for fold in range(folds):
        logger.info("=== Fold %d / %d ===", fold + 1, folds)
        fold_dir = Path(cfg["model_path"]) / cfg["training_name"] / f"Fold{fold+1}"
        ensure_dir(fold_dir / "weights"); ensure_dir(fold_dir / "attention")

        # datasets
        tr_csv = f"{cfg['drug_sensitivity_filepath']}train_Fold{fold}.csv"
        te_csv = f"{cfg['drug_sensitivity_filepath']}test_Fold{fold}.csv"
        ds_kwargs = dict(
            smiles_csv=cfg["smiles_filepath"],
            gep_csv=cfg["gep_filepath"],
            cnv_csv=cfg["cnv_filepath"],
            mut_csv=cfg["mut_filepath"],
            tokenizer_name="DeepChem/ChemBERTa-77M-MLM",
            max_len=cfg["params"].get("smiles_padding_length", 128),
        )
        train_ds = ChemBERTaOmicsDataset(drug_sensitivity_csv=tr_csv, **ds_kwargs)
        val_ds   = ChemBERTaOmicsDataset(drug_sensitivity_csv=te_csv, **ds_kwargs)

        loader_kwargs = dict(batch_size=cfg["params"]["batch_size"], num_workers=cfg["params"].get("num_workers", 4), collate_fn=chemberta_collate)
        tr_loader = torch.utils.data.DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kwargs)
        val_loader = torch.utils.data.DataLoader(val_ds, shuffle=False, drop_last=False, **loader_kwargs)
        logger.info("Data: %d train | %d val", len(train_ds), len(val_ds))

        # model
        params = cfg["params"].copy(); params.update(number_of_genes=len(pathways))
        model = PASO_GEP_CNV_MUT(params).to(device)

        opt_name = params.get("optimizer", "adam").lower()
        if opt_name not in OPTIMIZER_FACTORY:
            raise KeyError(f"Unsupported optimizer {opt_name}")
        optimizer = OPTIMIZER_FACTORY[opt_name](model.parameters(), lr=params.get("lr", 1e-3))
        scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=params.get("lr_scheduler_patience", 10), factor=0.5, verbose=True)
        early_stop = EarlyStopping(params.get("early_stopping_patience", 20))

        ic50_max = params["drug_sensitivity_processing_parameters"]["parameters"]["max"]
        ic50_min = params["drug_sensitivity_processing_parameters"]["parameters"]["min"]
        attn_interval = params.get("attn_viz_interval", 10)

        best_val = float("inf")
        for epoch in range(1, params["epochs"] + 1):
            logger.info("Epoch %03d / %03d", epoch, params["epochs"])
            tr_loss = train_epoch(model, tr_loader, optimizer, device, ic50_max, ic50_min)
            val_loss, pear, rmse, r2, _, _ = eval_epoch(model, val_loader, device, ic50_max, ic50_min)
            logger.info("Train %.4f | Val %.4f | Pearson %.4f | RMSE %.4f | R2 %.4f", tr_loss, val_loss, pear, rmse, r2)

            scheduler.step(val_loss)

            # --- save best & last ----
            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), fold_dir / "weights" / "best_val.pt")
            torch.save(model.state_dict(), fold_dir / "weights" / "last_epoch.pt")

            # --- save attention ---
            if epoch % attn_interval == 0:
                with torch.no_grad():
                    sample = next(iter(val_loader))
                    drug = {k: v.to(device) for k, v in sample.items() if k in {"input_ids", "attention_mask"}}
                    _, p_dict = model(drug, sample["gep"].to(device), sample["cnv"].to(device), sample["mut"].to(device))
                save_attention(p_dict, fold_dir / "attention", epoch)

            # --- early stop ---
            if early_stop.step(val_loss):
                logger.info("Early stopping at epoch %d", epoch)
                break
                        
# ----------------------------------------------------------------------------- #
# Config helper & entry-point
# ----------------------------------------------------------------------------- #
def build_cfg() -> Dict[str, Any]:
    return { ... }   # 기존 딕셔너리 그대로 넣기

if __name__ == "__main__":
    train(build_cfg())