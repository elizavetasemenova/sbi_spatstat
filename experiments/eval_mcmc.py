"""Compare NPE posteriors against the gold-standard NUTS posteriors."""

import json
import numpy as np
import torch

import config as C
from sbilgcp import lgcp, diagnostics as diag
from sbilgcp.npe import NPE, CNNEmbedding

torch.set_num_threads(4)


def main():
    mc = np.load(C.F_MCMC)
    mc_theta = mc["theta"]              # [K, S, 3] (may contain nan rows)
    idx = mc["idx"]
    walls = mc["walls"]
    rhats = mc["rhats"]

    d = np.load(C.F_TEST)
    y_te = d["y"].astype(np.float32)
    theta_true = d["theta"].astype(np.float32)[idx]

    npe = NPE(CNNEmbedding(embedding_dim=48), n_transforms=5, hidden=64)
    npe.load_state_dict(torch.load(C.F_NPE)); npe.eval()

    K = len(idx)
    c2st_vals, wass, mean_ref, mean_npe, std_ref, std_npe = [], [], [], [], [], []
    examples = {}
    keep = []
    for k in range(K):
        ref_s = mc_theta[k]
        ref_s = ref_s[~np.isnan(ref_s).any(1)]
        if rhats[k].max() > 1.1 or len(ref_s) < 200:
            continue                    # drop non-converged references
        keep.append(k)
        yk = torch.as_tensor(y_te[idx[k]:idx[k] + 1], dtype=torch.float32)
        u = npe.sample(yk, n_per=len(ref_s))
        npe_s = lgcp.from_unconstrained(u.reshape(-1, 3)).numpy()
        c2st_vals.append(diag.c2st(ref_s, npe_s, seed=k))
        wass.append(diag.wasserstein1_marginal(ref_s, npe_s))
        mean_ref.append(ref_s.mean(0)); mean_npe.append(npe_s.mean(0))
        std_ref.append(ref_s.std(0)); std_npe.append(npe_s.std(0))
        if len(examples) < 3:
            examples[str(k)] = {
                "ref": ref_s[:800].tolist(),
                "npe": npe_s[:800].tolist(),
                "true": theta_true[k].tolist(),
            }

    c2st_vals = np.array(c2st_vals)
    wass = np.array(wass)
    mean_ref, mean_npe = np.array(mean_ref), np.array(mean_npe)
    std_ref, std_npe = np.array(std_ref), np.array(std_npe)

    # normalised Wasserstein by prior std for interpretability
    prior_sd = np.array([lgcp.BETA0_SD,
                         (lgcp.SIGMA_HIGH - lgcp.SIGMA_LOW) / np.sqrt(12),
                         (lgcp.ELL_HIGH - lgcp.ELL_LOW) / np.sqrt(12)])

    out = {
        "n_compared": len(keep),
        "c2st_mean": float(c2st_vals.mean()),
        "c2st_median": float(np.median(c2st_vals)),
        "c2st_all": c2st_vals.tolist(),
        "wasserstein_mean": wass.mean(0).tolist(),
        "wasserstein_norm_mean": (wass.mean(0) / prior_sd).tolist(),
        "mcmc_wall_median": float(np.median(walls)),
        "mcmc_wall_mean": float(np.mean(walls)),
        "npe_amortized_per_dataset_ms": None,   # filled from calibration.json
        "post_mean_corr": [float(np.corrcoef(mean_ref[:, p], mean_npe[:, p])[0, 1])
                           for p in range(3)],
        "post_std_ratio_mean": (std_npe / (std_ref + 1e-9)).mean(0).tolist(),
        "examples": examples,
        "mean_ref": mean_ref.tolist(), "mean_npe": mean_npe.tolist(),
        "std_ref": std_ref.tolist(), "std_npe": std_npe.tolist(),
    }
    try:
        with open(C.F_CALIB) as f:
            cal = json.load(f)
        out["npe_amortized_per_dataset_ms"] = 1e3 * cal["npe_infer_time_per_dataset"]
        out["speedup"] = out["mcmc_wall_median"] / cal["npe_infer_time_per_dataset"]
    except Exception:
        pass

    with open(C.F_MCMC_CMP, "w") as f:
        json.dump(out, f, indent=2)
    print(f"compared {len(keep)} datasets")
    print(f"  C2ST mean {out['c2st_mean']:.3f} (0.5=indistinguishable)")
    print(f"  Wasserstein (norm by prior sd) {np.round(out['wasserstein_norm_mean'],3)}")
    print(f"  posterior-mean corr {np.round(out['post_mean_corr'],3)}")
    if "speedup" in out:
        print(f"  speedup vs NUTS: {out['speedup']:.0f}x")
    print(f"saved -> {C.F_MCMC_CMP}")


if __name__ == "__main__":
    main()
