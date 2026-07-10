"""Calibration, coverage and recovery for NPE and baselines (+ misspecification).

Produces results/calibration.json and results/eval_arrays.npz.
"""

import json
import time
import numpy as np
import torch

import config as C
from sbilgcp import lgcp, diagnostics as diag
from sbilgcp.npe import NPE, CNNEmbedding, MLPEmbedding, CNNRegressor

torch.set_num_threads(4)
LEVELS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]


def npe_posterior(model, y, n_per=C.N_POSTERIOR, batch=256):
    """Return natural-space posterior samples [M, n_per, 3]."""
    y = torch.as_tensor(y, dtype=torch.float32)
    out = []
    for s in range(0, y.shape[0], batch):
        u = model.sample(y[s:s + batch], n_per=n_per)          # [b, n_per, 3]
        nat = lgcp.from_unconstrained(u.reshape(-1, 3)).reshape(u.shape)
        out.append(nat.numpy())
    return np.concatenate(out, 0)


def npe_posterior_feats(model, feats, n_per=C.N_POSTERIOR, batch=256):
    feats = torch.as_tensor(feats, dtype=torch.float32)
    out = []
    for s in range(0, feats.shape[0], batch):
        u = model.sample(feats[s:s + batch], n_per=n_per)
        nat = lgcp.from_unconstrained(u.reshape(-1, 3)).reshape(u.shape)
        out.append(nat.numpy())
    return np.concatenate(out, 0)


def summarize(true_theta, post, L, tag):
    ranks = diag.sbc_ranks(true_theta, post)
    pvals = [diag.rank_uniformity_pvalue(ranks[:, p], L) for p in range(3)]
    cov = diag.credible_coverage(true_theta, post, LEVELS)
    rec = diag.recovery_metrics(true_theta, post)
    print(f"[{tag}] SBC p-values {np.round(pvals,3)}  "
          f"90%cov {np.round(cov[LEVELS.index(0.9)],3)}  RMSE {np.round(rec['rmse'],3)}")
    return {
        "sbc_pvalues": pvals,
        "coverage": cov.tolist(),
        "levels": LEVELS,
        "rmse": rec["rmse"].tolist(),
        "mae": rec["mae"].tolist(),
        "r2": rec["r2"].tolist(),
        "z_mean": rec["z_mean"].tolist(),
        "z_std": rec["z_std"].tolist(),
        "ranks": ranks.astype(int),
    }


def main():
    results = {"levels": LEVELS, "n_posterior": C.N_POSTERIOR}
    arrays = {}

    # ---- load test set ------------------------------------------------------
    d = np.load(C.F_TEST)
    theta_te = d["theta"].astype(np.float32)
    y_te = d["y"].astype(np.float32)
    M = theta_te.shape[0]

    # ---- main NPE -----------------------------------------------------------
    npe = NPE(CNNEmbedding(embedding_dim=48), n_transforms=5, hidden=64)
    npe.load_state_dict(torch.load(C.F_NPE)); npe.eval()
    t0 = time.time()
    post = npe_posterior(npe, y_te)
    t_infer = time.time() - t0
    results["npe_infer_time_total"] = t_infer
    results["npe_infer_time_per_dataset"] = t_infer / M
    print(f"NPE inference for {M} datasets: {t_infer:.2f}s "
          f"({1e3*t_infer/M:.3f} ms/dataset)")
    r = summarize(theta_te, post, C.N_POSTERIOR, "NPE-CNN")
    arrays["npe_ranks"] = r.pop("ranks")
    arrays["npe_post_mean"] = post.mean(1)
    arrays["npe_post_std"] = post.std(1)
    results["npe_cnn"] = r

    # ---- hand-crafted NPE ---------------------------------------------------
    feats = lgcp.handcrafted_summaries(torch.as_tensor(y_te)).numpy()
    train_feats = lgcp.handcrafted_summaries(
        torch.as_tensor(np.load(C.F_TRAIN)["y"][:5000])
    )
    emb = MLPEmbedding(feats.shape[1], embedding_dim=48)
    emb.set_normalizer(train_feats)
    npe_hc = NPE(emb, n_transforms=5, hidden=64)
    npe_hc.load_state_dict(torch.load(C.F_NPE_HC)); npe_hc.eval()
    post_hc = npe_posterior_feats(npe_hc, feats)
    r = summarize(theta_te, post_hc, C.N_POSTERIOR, "NPE-handcrafted")
    arrays["hc_ranks"] = r.pop("ranks")
    results["npe_handcrafted"] = r

    # ---- regressor baseline (Gaussian predictive via residual std) ----------
    reg = CNNRegressor(); reg.load_state_dict(torch.load(C.F_REG)); reg.eval()
    with torch.no_grad():
        # residual std estimated on training data (unconstrained space)
        dtr = np.load(C.F_TRAIN)
        ytr = torch.as_tensor(dtr["y"][:5000], dtype=torch.float32)
        thtr_u = lgcp.to_unconstrained(torch.as_tensor(dtr["theta"][:5000], dtype=torch.float32))
        pred_tr = reg(ytr)
        resid_std = (pred_tr - thtr_u).std(0).numpy()           # [3]
        pred_te_u = reg(torch.as_tensor(y_te, dtype=torch.float32)).numpy()
    rng = np.random.default_rng(0)
    draws_u = pred_te_u[:, None, :] + resid_std[None, None, :] * rng.standard_normal(
        (M, C.N_POSTERIOR, 3)
    )
    post_reg = lgcp.from_unconstrained(
        torch.as_tensor(draws_u.reshape(-1, 3), dtype=torch.float32)
    ).reshape(M, C.N_POSTERIOR, 3).numpy()
    r = summarize(theta_te, post_reg, C.N_POSTERIOR, "Regressor")
    arrays["reg_ranks"] = r.pop("ranks")
    results["regressor"] = r
    results["regressor"]["resid_std_unconstrained"] = resid_std.tolist()

    # ---- misspecification ----------------------------------------------------
    dm = np.load(C.F_MISSPEC)
    results["misspec"] = {}
    for nu in C.NU_MISSPEC:
        th = dm[f"theta_{nu}"].astype(np.float32)
        yy = dm[f"y_{nu}"].astype(np.float32)
        post_m = npe_posterior(npe, yy)
        r = summarize(th, post_m, C.N_POSTERIOR, f"NPE misspec nu={nu}")
        arrays[f"misspec_ranks_{nu}"] = r.pop("ranks")
        results["misspec"][str(nu)] = r

    with open(C.F_CALIB, "w") as f:
        json.dump(results, f, indent=2)
    np.savez_compressed(C.RESULTS + "/eval_arrays.npz", **arrays,
                        theta_te=theta_te)
    print(f"\nsaved -> {C.F_CALIB} and eval_arrays.npz")


if __name__ == "__main__":
    main()
