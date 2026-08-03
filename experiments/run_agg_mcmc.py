"""Validate the aggregation NPE against gold-standard NUTS on the aggregated model."""

import json
import numpy as np
import torch

import config as C
from sbilgcp import lgcp, aggregation as A, diagnostics as diag
from sbilgcp import reference_mcmc as ref

K = 15
COARSE = 4
F_DATA = C.RESULTS + "/agg_test.npz"
F_MODEL = C.RESULTS + "/agg_npe.pt"
F_OUT = C.RESULTS + "/agg_mcmc.json"


def main():
    d = np.load(F_DATA)
    theta = d["theta"].astype(np.float32); r = d["r"].astype(np.float32)
    sim = A.AggregationSimulator(G=C.GRID, C=COARSE)
    model = A.AggNPE(C=COARSE, n_transforms=5, hidden=64)
    model.load_state_dict(torch.load(F_MODEL)); model.eval()

    c2st_all, wass, walls, examples = [], [], [], {}
    for k in range(K):
        res = ref.run_nuts_aggregated(r[k], sim.core.coords, sim.area, C.GRID, COARSE,
                                      num_warmup=600, num_samples=500, num_chains=2, seed=500 + k)
        walls.append(res["wall"])
        if max(res["rhat"].values()) > 1.1:
            print(f"  [{k}] skip Rhat {max(res['rhat'].values()):.3f}"); continue
        ref_s = res["theta"]
        with torch.no_grad():
            u = model.sample(torch.as_tensor(r[k:k+1], dtype=torch.float32), n_per=len(ref_s))
            npe_s = lgcp.from_unconstrained(u.reshape(-1, 3)).numpy()
        c2st_all.append(diag.c2st(ref_s, npe_s, seed=k))
        wass.append(diag.wasserstein1_marginal(ref_s, npe_s))
        if len(examples) < 3:
            examples[str(k)] = {"ref": ref_s[:700].tolist(), "npe": npe_s[:700].tolist(),
                                "true": theta[k].tolist()}
        print(f"  [{k}] wall {res['wall']:.0f}s Rhat {max(res['rhat'].values()):.3f} "
              f"C2ST {c2st_all[-1]:.3f}", flush=True)

    prior_sd = np.array([lgcp.BETA0_SD, (lgcp.SIGMA_HIGH - lgcp.SIGMA_LOW) / np.sqrt(12),
                         (lgcp.ELL_HIGH - lgcp.ELL_LOW) / np.sqrt(12)])
    out = {
        "n_compared": len(c2st_all),
        "c2st_mean": float(np.mean(c2st_all)) if c2st_all else None,
        "wass_norm_mean": (np.mean(wass, 0) / prior_sd).tolist() if wass else None,
        "mcmc_wall_median": float(np.median(walls)),
        "examples": examples,
    }
    with open(F_OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"agg NPE vs NUTS over {out['n_compared']}: C2ST {out['c2st_mean']}")
    print(f"saved -> {F_OUT}")


if __name__ == "__main__":
    main()
