"""Compare the amortized field posterior against gold-standard NUTS."""

import json
import numpy as np
import torch

import config as C
from sbilgcp.lgcp import LGCPSimulator
from sbilgcp import field as F
from sbilgcp import reference_mcmc as ref

K = 12
F_DATA = C.RESULTS + "/field_test.npz"
F_MODEL = C.RESULTS + "/field_unet.pt"
F_OUT = C.RESULTS + "/field_mcmc.json"


def main():
    d = np.load(F_DATA)
    y = d["y"].astype(np.float32); Z = d["Z"].astype(np.float32)
    sim = LGCPSimulator(G=C.GRID); area = sim.area
    model = F.FieldUNet(rank=16); model.load_state_dict(torch.load(F_MODEL)); model.eval()

    pm_corr, ps_corr, rmse_ref, rmse_npe, cov_agree = [], [], [], [], []
    examples = {}
    for k in range(K):
        r = ref.run_nuts_field(y[k], sim.coords, area, num_warmup=500,
                               num_samples=500, num_chains=2, seed=300 + k)
        if r["rhat"] > 1.1:
            print(f"  [{k}] skip Rhat {r['rhat']:.3f}"); continue
        Zref = r["Z"]                                    # [draws, n]
        with torch.no_grad():
            mu, dd, V = model(torch.as_tensor(y[k:k+1], dtype=torch.float32))
        s = F.sample_field(mu, dd, V, Zref.shape[0], rng=k).numpy()[0]   # [draws,n]
        ztrue = Z[k].reshape(-1)
        # posterior-mean and posterior-sd agreement (per cell), correlation across cells
        pm_corr.append(float(np.corrcoef(Zref.mean(0), s.mean(0))[0, 1]))
        ps_corr.append(float(np.corrcoef(Zref.std(0), s.std(0))[0, 1]))
        rmse_ref.append(float(np.sqrt(((Zref.mean(0) - ztrue) ** 2).mean())))
        rmse_npe.append(float(np.sqrt(((s.mean(0) - ztrue) ** 2).mean())))
        # pointwise 90% coverage agreement
        def cov(samp):
            lo = np.quantile(samp, 0.05, 0); hi = np.quantile(samp, 0.95, 0)
            return float(((ztrue >= lo) & (ztrue <= hi)).mean())
        cov_agree.append((cov(Zref), cov(s)))
        if len(examples) < 2:
            examples[str(k)] = {"ref_mean": Zref.mean(0).tolist(), "npe_mean": s.mean(0).tolist(),
                                "ref_sd": Zref.std(0).tolist(), "npe_sd": s.std(0).tolist()}
        print(f"  [{k}] Rhat {r['rhat']:.3f} pm_corr {pm_corr[-1]:.3f} "
              f"sd_corr {ps_corr[-1]:.3f} cov ref/npe {cov_agree[-1][0]:.2f}/{cov_agree[-1][1]:.2f}",
              flush=True)

    out = {
        "n": len(pm_corr),
        "postmean_corr": float(np.mean(pm_corr)) if pm_corr else None,
        "poststd_corr": float(np.mean(ps_corr)) if ps_corr else None,
        "rmse_ref": float(np.mean(rmse_ref)) if rmse_ref else None,
        "rmse_npe": float(np.mean(rmse_npe)) if rmse_npe else None,
        "cov_ref": float(np.mean([c[0] for c in cov_agree])) if cov_agree else None,
        "cov_npe": float(np.mean([c[1] for c in cov_agree])) if cov_agree else None,
        "examples": examples,
    }
    with open(F_OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"field NPE vs NUTS over {out['n']}: post-mean corr {out['postmean_corr']}, "
          f"cov ref {out['cov_ref']} vs npe {out['cov_npe']}")
    print(f"saved -> {F_OUT}")


if __name__ == "__main__":
    main()
