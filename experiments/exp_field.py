"""Amortized reconstruction of the LGCP log-intensity surface.

Trains the field U-Net, then evaluates the posterior over the whole surface:
pointwise coverage, field RMSE, and---crucially---coverage of *aggregate*
functionals (total abundance, regional block counts), comparing the full
low-rank+diagonal posterior against a diagonal-only ablation to show the
low-rank term is what calibrates sums.
"""

import json
import numpy as np
import torch

import config as C
from sbilgcp.lgcp import LGCPSimulator, sample_prior
from sbilgcp import field as F

torch.set_num_threads(4)


def simulate_fields(sim, theta, rng, batch=1024):
    """Batched field simulation (avoids building a huge [N,n,n] covariance)."""
    ys, Zs = [], []
    for s in range(0, theta.shape[0], batch):
        y, Z = sim.simulate(theta[s:s + batch], rng, return_field=True)
        ys.append(y); Zs.append(Z)
    return torch.cat(ys, 0), torch.cat(Zs, 0)


N_TRAIN = 40000
N_TEST = 2000
S = 500
LEVELS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
SEED = 20250

F_RES = C.RESULTS + "/field_results.json"
F_ARR = C.RESULTS + "/field_arrays.npz"
F_MODEL = C.RESULTS + "/field_unet.pt"
F_DATA = C.RESULTS + "/field_test.npz"


def cov_at_levels(true, samp):
    """Coverage of central intervals for a scalar functional. true:[B], samp:[B,S]."""
    out = []
    for lv in LEVELS:
        lo = np.quantile(samp, (1 - lv) / 2, 1); hi = np.quantile(samp, 1 - (1 - lv) / 2, 1)
        out.append(float(((true >= lo) & (true <= hi)).mean()))
    return out


def main():
    sim = LGCPSimulator(G=C.GRID)
    area = sim.area
    print(f"simulating {N_TRAIN} fields ...")
    rng = np.random.default_rng(SEED)
    th = sample_prior(N_TRAIN, rng); y, Z = simulate_fields(sim, th, rng)
    print("training field U-Net ...")
    model = F.FieldUNet(rank=16)
    rep = F.train_field(model, y, Z, epochs=70, batch=128, lr=5e-4, patience=12, log_every=5)
    torch.save(model.state_dict(), F_MODEL)
    print(f"  best val NLL/cell {rep.best_val:.4f}  ({rep.wall/60:.1f} min)")

    rng = np.random.default_rng(SEED + 1)
    th_te = sample_prior(N_TEST, rng); y_te, Z_te = simulate_fields(sim, th_te, rng)
    Zt = Z_te.reshape(N_TEST, -1).numpy()
    np.savez_compressed(F_DATA, y=y_te.numpy(), Z=Z_te.numpy(), theta=th_te)

    model.eval()
    with torch.no_grad():
        mu, d, V = model(y_te)
    mu_n = mu.numpy()
    # field reconstruction quality
    rmse = float(np.sqrt(((mu_n - Zt) ** 2).mean()))
    w = np.exp(Zt); w = w / w.sum()
    rmse_w = float(np.sqrt((w * (mu_n - Zt) ** 2).sum(1).mean() * Zt.shape[1]))
    corr = float(np.corrcoef(mu_n.ravel(), Zt.ravel())[0, 1])

    # samples: full low-rank vs diagonal-only
    s_full = F.sample_field(mu, d, V, S, rng=1).numpy()
    s_diag = F.sample_field(mu, d, torch.zeros_like(V), S, rng=1).numpy()

    # pointwise coverage (full); pixel z-scores
    def pointwise_cov(samp):
        out = []
        for lv in LEVELS:
            lo = np.quantile(samp, (1 - lv) / 2, 1); hi = np.quantile(samp, 1 - (1 - lv) / 2, 1)
            out.append(float(((Zt >= lo) & (Zt <= hi)).mean()))
        return out
    pw_full = pointwise_cov(s_full)
    pmean = s_full.mean(1); pstd = s_full.std(1)
    zpix = ((Zt - pmean) / (pstd + 1e-8)).ravel()

    # functional coverage: total abundance + 4x4 block sums
    Ntrue = (area * np.exp(Zt)).sum(1)
    Nfull = (area * np.exp(s_full)).sum(2); Ndiag = (area * np.exp(s_diag)).sum(2)
    ab_full = cov_at_levels(Ntrue, Nfull); ab_diag = cov_at_levels(Ntrue, Ndiag)

    # regional block sums (aggregate to 4x4 super-cells -> 16 regions)
    G = C.GRID; b = G // 4
    def blocks(arr):  # arr [...,n]
        a = arr.reshape(arr.shape[:-1] + (4, b, 4, b))
        return a.sum(axis=(-3, -1)).reshape(arr.shape[:-1] + (16,))
    Rtrue = blocks(area * np.exp(Zt))
    Rfull = blocks(area * np.exp(s_full)); Rdiag = blocks(area * np.exp(s_diag))
    def region_cov(samp):
        lv = 0.9
        lo = np.quantile(samp, 0.05, 1); hi = np.quantile(samp, 0.95, 1)
        return float(((Rtrue >= lo) & (Rtrue <= hi)).mean())
    reg_full = region_cov(Rfull); reg_diag = region_cov(Rdiag)

    res = {
        "levels": LEVELS, "rmse": rmse, "rmse_intensity_weighted": rmse_w,
        "field_corr": corr, "val_nll": rep.best_val, "train_min": rep.wall / 60,
        "pointwise_cov": pw_full,
        "abundance_cov_full": ab_full, "abundance_cov_diag": ab_diag,
        "region_cov90_full": reg_full, "region_cov90_diag": reg_diag,
        "zpix_std": float(zpix.std()), "zpix_mean": float(zpix.mean()),
    }
    with open(F_RES, "w") as f:
        json.dump(res, f, indent=2)
    # arrays for example-map figure (first 3 test datasets)
    np.savez_compressed(
        F_ARR, y=y_te[:6].numpy(), Ztrue=Z_te[:6].numpy(),
        pmean=pmean[:6], pstd=pstd[:6], zpix=zpix[::37][:5000],
        Ntrue=Ntrue, Nfull_lo=np.quantile(Nfull, 0.05, 1), Nfull_hi=np.quantile(Nfull, 0.95, 1),
    )
    print(f"RMSE {rmse:.3f} (weighted {rmse_w:.3f}), corr {corr:.3f}")
    print(f"pointwise 90% cov {pw_full[LEVELS.index(0.9)]:.3f}")
    print(f"abundance 90% cov: full {ab_full[LEVELS.index(0.9)]:.3f} vs diag {ab_diag[LEVELS.index(0.9)]:.3f}")
    print(f"region 90% cov: full {reg_full:.3f} vs diag {reg_diag:.3f}")
    print(f"pixel z-score std {zpix.std():.3f} (target 1.0)")
    print(f"saved -> {F_RES}")


if __name__ == "__main__":
    main()
