"""Validate the design-aware preferential NPE against gold-standard NUTS."""

import json
import numpy as np
import torch

import config as C
from sbilgcp import preferential as P, diagnostics as diag
from sbilgcp import reference_mcmc as ref

K = 12
F_PREF_DATA = C.RESULTS + "/pref_test.npz"
F_PREF_PS = C.RESULTS + "/pref_npe_ps.pt"
F_OUT = C.RESULTS + "/pref_mcmc.json"


def main():
    d = np.load(F_PREF_DATA)
    theta = d["theta"].astype(np.float32); X = d["X"].astype(np.float32)
    sim = P.PreferentialSimulator(G=C.GRID)

    ps = P.NPEk(P.CNN2ch(48), param_dim=4, n_transforms=5, hidden=64)
    ps.load_state_dict(torch.load(F_PREF_PS)); ps.eval()

    c2st_all, wass, rhats = [], [], []
    examples = {}
    for k in range(K):
        obs, r = X[k, 0], X[k, 1]
        y = np.rint(np.expm1(obs)) * (r > 0.5)         # reconstruct integer counts
        res = ref.run_nuts_preferential(r, y, sim.core.coords, sim.area,
                                        num_warmup=600, num_samples=500, num_chains=2,
                                        seed=100 + k)
        rhats.append(max(res["rhat"].values()))
        if rhats[-1] > 1.1:
            print(f"  [{k}] skipped (Rhat {rhats[-1]:.3f})"); continue
        ref_s = res["theta"]
        with torch.no_grad():
            u = ps.sample(torch.as_tensor(X[k:k+1], dtype=torch.float32), n_per=len(ref_s))
            npe_s = P.from_unconstrained(u.reshape(-1, 4)).numpy()
        c2st_all.append(diag.c2st(ref_s, npe_s, seed=k))
        wass.append(diag.wasserstein1_marginal(ref_s, npe_s))
        if len(examples) < 2:
            examples[str(k)] = {"ref": ref_s[:600].tolist(), "npe": npe_s[:600].tolist(),
                                "true": theta[k].tolist()}
        print(f"  [{k}] wall {res['wall']:.0f}s Rhat {rhats[-1]:.3f} C2ST {c2st_all[-1]:.3f}",
              flush=True)

    out = {
        "n_compared": len(c2st_all),
        "c2st_mean": float(np.mean(c2st_all)) if c2st_all else None,
        "c2st_all": [float(x) for x in c2st_all],
        "wasserstein_mean": np.mean(wass, 0).tolist() if wass else None,
        "examples": examples,
    }
    with open(F_OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"design-aware NPE vs NUTS: C2ST {out['c2st_mean']} over {out['n_compared']} datasets")
    print(f"saved -> {F_OUT}")


if __name__ == "__main__":
    main()
