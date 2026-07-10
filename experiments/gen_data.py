"""Generate training, test and misspecification datasets."""

import time
import numpy as np

import config as C
from sbilgcp.lgcp import LGCPSimulator, sample_prior


def main():
    sim = LGCPSimulator(G=C.GRID, nu=C.NU_TRAIN)

    print(f"Simulating {C.N_TRAIN} training LGCPs on {C.GRID}x{C.GRID} grid ...")
    t0 = time.time()
    rng = np.random.default_rng(C.SEED_TRAIN)
    theta_tr, y_tr = sim.simulate_dataset(C.N_TRAIN, rng, batch=1024)
    np.savez_compressed(C.F_TRAIN, theta=theta_tr, y=y_tr.numpy().astype(np.float32))
    print(f"  train done in {time.time()-t0:.1f}s -> {C.F_TRAIN}")

    print(f"Simulating {C.N_TEST} held-out test LGCPs ...")
    rng = np.random.default_rng(C.SEED_TEST)
    theta_te, y_te = sim.simulate_dataset(C.N_TEST, rng, batch=1024)
    np.savez_compressed(C.F_TEST, theta=theta_te, y=y_te.numpy().astype(np.float32))
    print(f"  test done -> {C.F_TEST}")

    print("Simulating misspecification test sets (Matern-3/2, 5/2) ...")
    misspec = {}
    for nu in C.NU_MISSPEC:
        sim_m = LGCPSimulator(G=C.GRID, nu=nu)
        rng = np.random.default_rng(C.SEED_TEST + int(nu * 10))
        th, yy = sim_m.simulate_dataset(C.N_MISSPEC, rng, batch=1024)
        misspec[f"theta_{nu}"] = th
        misspec[f"y_{nu}"] = yy.numpy().astype(np.float32)
    np.savez_compressed(C.F_MISSPEC, **misspec)
    print(f"  misspec done -> {C.F_MISSPEC}")

    # quick summary of point-pattern richness
    tot = y_tr.numpy().reshape(C.N_TRAIN, -1).sum(1)
    print(f"train total-count percentiles [5,50,95]: "
          f"{np.percentile(tot,[5,50,95]).round(0)}")


if __name__ == "__main__":
    main()
