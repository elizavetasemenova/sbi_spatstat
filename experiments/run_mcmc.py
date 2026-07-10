"""Run gold-standard NUTS on a subset of test datasets (the expensive baseline)."""

import time
import numpy as np

import config as C
from sbilgcp.lgcp import LGCPSimulator
from sbilgcp import reference_mcmc as ref


def main():
    d = np.load(C.F_TEST)
    theta_te = d["theta"].astype(np.float32)
    y_te = d["y"].astype(np.float32)

    sim = LGCPSimulator(G=C.GRID, nu=C.NU_TRAIN)
    idx = np.arange(C.N_MCMC)                      # first N_MCMC test datasets

    all_theta = np.full((C.N_MCMC, C.MCMC_SAMPLES * C.MCMC_CHAINS, 3), np.nan)
    walls = np.zeros(C.N_MCMC)
    rhats = np.zeros((C.N_MCMC, 3))
    print(f"Running NUTS on {C.N_MCMC} datasets "
          f"({C.MCMC_WARMUP} warmup + {C.MCMC_SAMPLES} draws x {C.MCMC_CHAINS} chains)")
    t_start = time.time()
    for k, i in enumerate(idx):
        r = ref.run_nuts(
            y_te[i], sim.coords, sim.area, nu=C.NU_TRAIN,
            num_warmup=C.MCMC_WARMUP, num_samples=C.MCMC_SAMPLES,
            num_chains=C.MCMC_CHAINS, seed=C.SEED_MCMC + i,
        )
        n = r["theta"].shape[0]
        all_theta[k, :n] = r["theta"]
        walls[k] = r["wall"]
        rhats[k] = [r["rhat"]["beta0"], r["rhat"]["sigma"], r["rhat"]["ell"]]
        print(f"  [{k+1}/{C.N_MCMC}] wall {r['wall']:.1f}s  "
              f"maxRhat {rhats[k].max():.3f}  elapsed {(time.time()-t_start)/60:.1f}min",
              flush=True)

    np.savez_compressed(
        C.F_MCMC, theta=all_theta, walls=walls, rhats=rhats,
        idx=idx, true_theta=theta_te[idx],
    )
    print(f"median wall/dataset {np.median(walls):.1f}s  "
          f"max Rhat overall {rhats.max():.3f}")
    print(f"saved -> {C.F_MCMC}")


if __name__ == "__main__":
    main()
