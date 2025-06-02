import torch
from torch.utils.data import DataLoader
import numpy as np
from collections import defaultdict
from utils.loss_functions import pearsonr, r2_score

# ──────────────────────────────────────────────────────────────────────────────
# Basic numerical sanity checks
# ──────────────────────────────────────────────────────────────────────────────
def assert_finite(tensor, name):
    if not torch.isfinite(tensor).all():
        bad = tensor[~torch.isfinite(tensor)]
        raise RuntimeError(f"{name} contains non-finite values: {bad[:10]}")

# ──────────────────────────────────────────────────────────────────────────────
# Batch-level probe during training / evaluation
# ──────────────────────────────────────────────────────────────────────────────
def probe_batch(model, batch, device, scaler=None):
    """Run one forward pass and return a dict of raw + log-scaled preds/labels."""
    model.eval()
    with torch.no_grad():
        drug_data = {
            "input_ids": batch["input_ids"].to(device),
            "attention_mask": batch["attention_mask"].to(device),
        }
        y_hat, pred_dict = model(
            drug_data,
            batch["gep"].to(device),
            batch["cnv"].to(device),
            batch["mut"].to(device),
        )
        if y_hat.ndim == 2 and y_hat.shape[1] == 1:
            y_hat = y_hat.squeeze(1)
    y_true_raw = batch["ic50"].to(device).float()
    y_pred_raw = y_hat.float()

    if scaler is not None:
        # assume scaler is a callable shared between train and eval
        y_true_log = scaler(y_true_raw)
        y_pred_log = scaler(y_pred_raw)
    else:
        y_true_log = y_true_raw
        y_pred_log = y_pred_raw

    # basic checks
    assert_finite(y_true_raw, "y_true_raw")
    assert_finite(y_pred_raw, "y_pred_raw")

    return {
        "true_raw": y_true_raw.cpu(),
        "pred_raw": y_pred_raw.cpu(),
        "true_log": y_true_log.cpu(),
        "pred_log": y_pred_log.cpu(),
    }

# ──────────────────────────────────────────────────────────────────────────────
# Epoch-level metric tracker
# ──────────────────────────────────────────────────────────────────────────────
class MetricTracker:
    """Accumulates metrics across batches and prints a quick summary."""

    def __init__(self):
        self.buf = defaultdict(list)

    def update(self, d):
        for k, v in d.items():
            self.buf[k].append(v.detach().cpu())

    def summary(self):
        out = {}
        for k, vs in self.buf.items():
            cat = torch.cat(vs)
            out[k] = {
                "mean": float(cat.mean()),
                "std": float(cat.std()),
                "min": float(cat.min()),
                "max": float(cat.max()),
            }
        # compute final metrics in chosen space (here log)
        y_p, y_t = torch.cat(self.buf["pred_log"]), torch.cat(self.buf["true_log"])
        y_p = y_p.reshape(-1)
        y_t = y_t.reshape(-1)
        out["pearson_log"]  = float(pearsonr(y_p, y_t))
        out["rmse_log"]     = float(torch.sqrt(torch.mean((y_p - y_t) ** 2)))
        out["r2_log"]       = float(r2_score(y_p, y_t))

        return out