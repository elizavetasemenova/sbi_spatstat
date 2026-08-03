"""Train the NPE (CNN+MAF), the hand-crafted-summary NPE, and the regressor."""

import json
import time
import numpy as np
import torch

import config as C
from sbilgcp import lgcp
from sbilgcp.npe import (
    NPE, CNNEmbedding, MLPEmbedding, CNNRegressor,
    TrainConfig, train_npe, train_regressor,
)

torch.set_num_threads(4)


def main():
    d = np.load(C.F_TRAIN)
    theta = d["theta"].astype(np.float32)
    y = torch.as_tensor(d["y"], dtype=torch.float32)
    print(f"loaded {theta.shape[0]} training datasets")

    reports = {}

    # ---- Main method: CNN embedding + MAF ------------------------------------
    print("\n=== Training NPE (CNN + MAF) ===")
    torch.manual_seed(0)
    npe = NPE(CNNEmbedding(embedding_dim=48), n_transforms=5, hidden=64, seed=0)
    cfg = TrainConfig(epochs=80, batch_size=256, lr=5e-4, patience=15, log_every=5)
    rep = train_npe(npe, theta, y, cfg)
    torch.save(npe.state_dict(), C.F_NPE)
    reports["npe_cnn"] = {
        "val": rep.best_val, "wall": rep.wall_time, "best_epoch": rep.best_epoch,
        "n_params": sum(p.numel() for p in npe.parameters()),
        "train_curve": rep.train_loss, "val_curve": rep.val_loss,
    }
    print(f"  best val {rep.best_val:.4f}  time {rep.wall_time:.1f}s")

    # ---- Baseline 1: hand-crafted spatial summaries + MAF --------------------
    print("\n=== Training NPE (hand-crafted summaries + MAF) ===")
    feats = lgcp.handcrafted_summaries(y)
    emb = MLPEmbedding(feats.shape[1], embedding_dim=48)
    emb.set_normalizer(feats)
    torch.manual_seed(0)
    npe_hc = NPE(emb, n_transforms=5, hidden=64, seed=0)
    rep_hc = train_npe(npe_hc, theta, feats, cfg)
    torch.save(npe_hc.state_dict(), C.F_NPE_HC)
    reports["npe_handcrafted"] = {
        "val": rep_hc.best_val, "wall": rep_hc.wall_time, "best_epoch": rep_hc.best_epoch,
        "n_summaries": int(feats.shape[1]),
        "train_curve": rep_hc.train_loss, "val_curve": rep_hc.val_loss,
    }
    print(f"  best val {rep_hc.best_val:.4f}  time {rep_hc.wall_time:.1f}s")

    # ---- Baseline 2: point-estimate CNN regressor ---------------------------
    print("\n=== Training CNN regressor (point-estimate baseline) ===")
    torch.manual_seed(0)
    reg = CNNRegressor()
    rep_r = train_regressor(reg, theta, y, TrainConfig(epochs=80, batch_size=256, patience=15))
    torch.save(reg.state_dict(), C.F_REG)
    reports["regressor"] = {
        "val_mse": rep_r.best_val, "wall": rep_r.wall_time,
        "train_curve": rep_r.train_loss, "val_curve": rep_r.val_loss,
    }
    print(f"  best val MSE {rep_r.best_val:.4f}  time {rep_r.wall_time:.1f}s")

    with open(C.F_REPORTS, "w") as f:
        json.dump(reports, f, indent=2)
    print(f"\nsaved reports -> {C.F_REPORTS}")


if __name__ == "__main__":
    main()
