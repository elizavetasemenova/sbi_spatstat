"""Real-data disease mapping: amortized Poisson-offset LGCP on Burkina Faso
malaria parasite-rate survey data, validated against NUTS."""

import json
import time
import csv
import numpy as np
import torch

import config as C
from sbilgcp import lgcp, diseasemap as DM, diagnostics as diag
from sbilgcp.npe import TrainConfig, train_npe
from sbilgcp import reference_mcmc as ref

torch.set_num_threads(4)
G = C.GRID
N_TRAIN = 40000
N_TEST = 2000
DATA = C.ROOT + "/data/BF_malaria_data.csv"
F_RES = C.RESULTS + "/malaria_results.json"
F_ARR = C.RESULTS + "/malaria_arrays.npz"
F_MODEL = C.RESULTS + "/malaria_npe.pt"


def load_grid():
    rows = list(csv.DictReader(open(DATA)))
    lon = np.array([float(r["longitude"]) for r in rows]); lat = np.array([float(r["latitude"]) for r in rows])
    exm = np.array([float(r["examined"]) for r in rows]); pos = np.array([float(r["positives"]) for r in rows])
    ux = (lon - lon.min()) / (lon.max() - lon.min()) * 0.999
    uy = (lat - lat.min()) / (lat.max() - lat.min()) * 0.999
    ix = (ux * G).astype(int); iy = (uy * G).astype(int)
    E = np.zeros((G, G)); Y = np.zeros((G, G))
    for k in range(len(rows)):
        E[ix[k], iy[k]] += exm[k]; Y[ix[k], iy[k]] += pos[k]
    return E.astype(np.float32), Y.astype(np.float32), (lon, lat, pos, exm)


def main():
    E, Y, raw = load_grid()
    sim = DM.DiseaseMapSimulator(offset=E)

    # ---- train amortized disease-mapping NPE (fixed offset) -----------------
    print("simulating + training disease-mapping NPE ...")
    rng = np.random.default_rng(9)
    theta_tr, Ytr = sim.simulate_dataset(N_TRAIN, rng)
    Xtr = sim.two_channel(Ytr)
    model = DM.DMNPE(n_transforms=5, hidden=64)
    rep = train_npe(model, theta_tr, Xtr, TrainConfig(epochs=70, batch_size=256, patience=12,
                                                      log_every=10), transform_fn=DM.to_unconstrained)
    torch.save(model.state_dict(), F_MODEL)
    print(f"  best val {rep.best_val:.3f} ({rep.wall_time/60:.1f} min)")

    # ---- held-out calibration check -----------------------------------------
    rng = np.random.default_rng(10)
    theta_te, Yte = sim.simulate_dataset(N_TEST, rng)
    Xte = sim.two_channel(Yte)
    with torch.no_grad():
        u = model.sample(Xte, n_per=1000)
    post = DM.from_unconstrained(u.reshape(-1, 3)).reshape(N_TEST, 1000, 3).numpy()
    ranks = diag.sbc_ranks(theta_te, post)
    sbc_p = [diag.rank_uniformity_pvalue(ranks[:, p], 1000) for p in range(3)]
    cov90 = diag.credible_coverage(theta_te, post, [0.9])[0]
    print(f"held-out SBC p {np.round(sbc_p,3)} | 90% cov {np.round(cov90,2)}")

    # ---- apply to REAL data (one forward pass) ------------------------------
    Xreal = sim.two_channel(Y)
    t0 = time.time()
    with torch.no_grad():
        u = model.sample(Xreal, n_per=2000)
    theta_npe = DM.from_unconstrained(u.reshape(-1, 3)).numpy()
    t_npe = time.time() - t0

    # ---- gold-standard NUTS (params + risk field) ---------------------------
    t0 = time.time()
    r = ref.run_nuts_diseasemap(Y, E, sim.core.coords, num_warmup=1000, num_samples=1000,
                                num_chains=3, seed=0, return_field=True,
                                beta0_low=DM.BETA0_LOW, beta0_high=DM.BETA0_HIGH)
    t_nuts = time.time() - t0
    theta_nuts = r["theta"]
    risk_map = np.exp(r["Z"]).mean(0)           # posterior mean malaria risk per cell
    risk_sd = np.exp(r["Z"]).std(0)

    c2st = diag.c2st(theta_nuts, theta_npe[:len(theta_nuts)], seed=0)
    res = {
        "n_sites": len(raw[0]), "surveyed_cells": int((E > 0).sum()),
        "total_examined": int(E.sum()), "total_positives": int(Y.sum()),
        "empirical_prev": float(Y.sum() / E.sum()),
        "theta_npe_mean": theta_npe.mean(0).tolist(), "theta_npe_sd": theta_npe.std(0).tolist(),
        "theta_nuts_mean": theta_nuts.mean(0).tolist(), "theta_nuts_sd": theta_nuts.std(0).tolist(),
        "param_c2st": float(c2st), "rhat": r["rhat"],
        "sbc_p": sbc_p, "cov90": cov90.tolist(),
        "t_npe_ms": 1e3 * t_npe, "t_nuts_s": t_nuts, "speedup": t_nuts / t_npe,
        "train_min": rep.wall_time / 60,
    }
    with open(F_RES, "w") as f:
        json.dump(res, f, indent=2)
    np.savez_compressed(F_ARR, E=E, Y=Y, risk_map=risk_map.reshape(G, G),
                        risk_sd=risk_sd.reshape(G, G), theta_npe=theta_npe, theta_nuts=theta_nuts,
                        lon=raw[0], lat=raw[1], pos=raw[2], exm=raw[3])
    print(f"BF malaria: {int(Y.sum())} positives / {int(E.sum())} examined at {len(raw[0])} sites")
    print(f"NPE  theta {np.round(theta_npe.mean(0),3)} sd {np.round(theta_npe.std(0),3)}")
    print(f"NUTS theta {np.round(theta_nuts.mean(0),3)} sd {np.round(theta_nuts.std(0),3)}")
    print(f"param C2ST {c2st:.3f} | NPE {1e3*t_npe:.0f} ms vs NUTS {t_nuts:.0f} s")
    print(f"saved -> {F_RES}")


if __name__ == "__main__":
    main()
