"""Amortized LGCP inference from spatially aggregated counts (change of support)."""

import json
import numpy as np
import torch

import config as C
from sbilgcp import lgcp, aggregation as A, diagnostics as diag
from sbilgcp.npe import TrainConfig, train_npe

torch.set_num_threads(4)
N_TRAIN = 60000
N_TEST = 3000
N_POST = 1000
COARSE = 4                       # 4x4 regions (16-fold aggregation of the 16x16 grid)
LEVELS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
SEED = 71830

F_RES = C.RESULTS + "/agg_results.json"
F_ARR = C.RESULTS + "/agg_arrays.npz"
F_MODEL = C.RESULTS + "/agg_npe.pt"
F_DATA = C.RESULTS + "/agg_test.npz"


def posterior(model, r, n_per=N_POST, batch=256):
    r = torch.as_tensor(r, dtype=torch.float32)
    out = []
    for s in range(0, r.shape[0], batch):
        u = model.sample(r[s:s + batch], n_per=n_per)
        out.append(lgcp.from_unconstrained(u.reshape(-1, 3)).reshape(u.shape[0], n_per, 3).numpy())
    return np.concatenate(out, 0)


def main():
    sim = A.AggregationSimulator(G=C.GRID, C=COARSE)
    print(f"simulating {N_TRAIN} aggregated ({COARSE}x{COARSE}-region) datasets ...")
    rng = np.random.default_rng(SEED)
    theta_tr, r_tr = sim.simulate_dataset(N_TRAIN, rng)
    print("training aggregation NPE ...")
    model = A.AggNPE(C=COARSE, n_transforms=5, hidden=64)
    rep = train_npe(model, theta_tr, r_tr, TrainConfig(epochs=80, batch_size=256, patience=15,
                                                       log_every=5), transform_fn=lgcp.to_unconstrained)
    torch.save(model.state_dict(), F_MODEL)
    print(f"  best val {rep.best_val:.3f} ({rep.wall/60:.1f} min)")

    rng = np.random.default_rng(SEED + 1)
    theta_te, r_te = sim.simulate_dataset(N_TEST, rng)
    np.savez_compressed(F_DATA, theta=theta_te, r=r_te.numpy())
    post = posterior(model, r_te)

    ranks = diag.sbc_ranks(theta_te, post)
    pvals = [diag.rank_uniformity_pvalue(ranks[:, p], N_POST) for p in range(3)]
    cov = diag.credible_coverage(theta_te, post, LEVELS)
    rec = diag.recovery_metrics(theta_te, post)

    # information loss vs fully observed: compare posterior widths & R^2
    full = json.load(open(C.F_CALIB))["npe_cnn"]
    width_agg = post.std(1).mean(0)
    # fully-observed widths from stored eval arrays
    full_arr = dict(np.load(C.RESULTS + "/eval_arrays.npz"))
    width_full = full_arr["npe_post_std"].mean(0)

    res = {
        "coarse": COARSE, "levels": LEVELS,
        "sbc_pvalues": pvals,
        "coverage": cov.tolist(),
        "rmse": rec["rmse"].tolist(), "r2": rec["r2"].tolist(),
        "post_width_agg": width_agg.tolist(),
        "post_width_full": width_full.tolist(),
        "width_ratio": (width_agg / width_full).tolist(),
        "r2_full": full["r2"],
        "val_nll": rep.best_val, "train_min": rep.wall / 60,
    }
    with open(F_RES, "w") as f:
        json.dump(res, f, indent=2)
    np.savez_compressed(F_ARR, theta_te=theta_te, post_mean=post.mean(1),
                        post_std=post.std(1), ranks=ranks.astype(int))
    i90 = LEVELS.index(0.9)
    print(f"SBC p {np.round(pvals,3)}  90% cov {np.round(cov[i90],2)}")
    print(f"R2 agg {np.round(rec['r2'],2)} vs full {np.round(full['r2'],2)}")
    print(f"posterior width ratio (agg/full) {np.round(width_agg/width_full,2)}")
    print(f"saved -> {F_RES}")


if __name__ == "__main__":
    main()
