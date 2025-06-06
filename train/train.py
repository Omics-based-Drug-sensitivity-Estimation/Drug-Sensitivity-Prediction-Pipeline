#!/usr/bin/env python
# ─────────────────────────────────────────────────────────────────────
# train.py
# ─────────────────────────────────────────────────────────────────────
"""
Entry-point to train any PASO variant.

Usage
-----
$ python -m train.train --model_version v4 --config_path configs/params.json
"""

from __future__ import annotations

import argparse, json, logging, random, shutil, time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

# project imports
from utils.utils import get_device, set_seed
from data.TripleOmics_ChemBERT_Dataset import (
    ChemBERTaOmicsDataset,
    chemberta_collate,
)
from data.TripleOmics_BGD_Dataset import (
    TripleOmics_Drug_dataset,
    bgd_collate,
)
from models.model import MODEL_FACTORY
from utils.hyperparams import OPTIMIZER_FACTORY, SCHEDULER_FACTORY
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from tqdm import tqdm
# ─────────────────────────────────────────────────────────────────────
# CLI & helpers
# ─────────────────────────────────────────────────────────────────────
LOG = logging.getLogger("train")

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train PASO models (ChemBERTa / BGD back-ends)."
    )
    p.add_argument("--model_version", required=True, choices=["v1", "v2", "v3", "v4", "v5"])
    p.add_argument("--config_path", required=True, help="Path to JSON config.")
    p.add_argument("--run_name", default="run", help="Folder inside results/ to save checkpoints.")
    return p.parse_args()


def _load_cfg(path: str | Path) -> Dict:
    with open(path) as j:
        return json.load(j)


def _dataloaders(cfg: Dict, version: str) -> Tuple[DataLoader, DataLoader]:
    """Return (train_loader, test_loader) for the requested model family."""
    if version in {"v1", "v2", "v3"}:          # Chem-based
        ds_cls, collate_fn = ChemBERTaOmicsDataset, chemberta_collate
    else:                                      # Graph-based (BGD)
        ds_cls, collate_fn = TripleOmics_Drug_dataset, bgd_collate

    train_ds = ds_cls(**cfg["train_dataset_args"])
    test_ds  = ds_cls(**cfg["test_dataset_args"])

    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg["batch_size"], shuffle=False, collate_fn=collate_fn
    )
    return train_loader, test_loader


# ─────────────────────────────────────────────────────────────────────
# tiny trainer (1-file demo) – replace with your advanced trainer later
# ─────────────────────────────────────────────────────────────────────
def _train_one_epoch(model, loader, optim, device):
    model.train()
    total, n = 0.0, 0
    for batch in tqdm(loader, desc="  ▶ Training batches", unit="batch"):
        optim.zero_grad(set_to_none=True)
        # ChemBERTa 계열은 dict, BGD 계열은 tuple
        if isinstance(batch, dict):
            out, _ = model(batch, batch["gep"], batch["cnv"], batch["mut"])
            loss = model.loss(out, batch["ic50"].to(device))
        else:
            (drug, gep, cnv, mut, y) = batch
            out, _ = model(drug, gep, cnv, mut)
            loss = model.loss(out, y.to(device))
        loss.backward()
        optim.step()
        total += loss.item() * len(out)
        n += len(out)
    return total / n



def _evaluate(model, loader, device):
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, dict):
                out, _ = model(batch, batch["gep"], batch["cnv"], batch["mut"])
                loss = model.loss(out, batch["ic50"].to(device))
            else:
                (drug, gep, cnv, mut, y) = batch
                out, _ = model(drug, gep, cnv, mut)
                loss = model.loss(out, y.to(device))
            total += loss.item() * len(out)
            n += len(out)
    return total / n


# ─────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────
def main() -> None:
    args = _parse_args()
    cfg = _load_cfg(args.config_path)
    cfg["model_version"] = args.model_version

    # setup I/O
    out_dir = Path("results") / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # reproducibility
    seed = cfg.get("seed", 42)
    set_seed(seed)

    # device & model
    device = get_device()
    model_cls = MODEL_FACTORY[args.model_version]
    model = model_cls(cfg).to(device)

    # optimiser / scheduler (optuna-ready keys!)
    optim = OPTIMIZER_FACTORY[cfg.get("optimizer", "adam")](model.parameters(), lr=cfg["lr"])
    kw = cfg.get("scheduler_kw", {})           # ← NEW
    sched = SCHEDULER_FACTORY[cfg.get("scheduler", "plateau")](optim, **kw)

    best_val = float("inf")
    for fold in range(cfg.get("folds", 1)):
        LOG.info("─" * 60)
        LOG.info("Fold %d / %d", fold + 1, cfg.get("folds", 1))

        train_loader, val_loader = _dataloaders(cfg, args.model_version)

        for epoch in range(1, cfg.get("epochs", 10) + 1):
            t0 = time.time()
            tr_loss = _train_one_epoch(model, train_loader, optim, device)
            val_loss = _evaluate(model, val_loader, device)
            sched.step(val_loss)

            LOG.info(
                "Epoch %3d │ train %.4f │ val %.4f │ Δt %.1fs",
                epoch,
                tr_loss,
                val_loss,
                time.time() - t0,
            )

            # save best per fold
            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), out_dir / f"best_fold{fold+1}.pt")

    LOG.info("Training finished. Best val loss: %.4f", best_val)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    main()