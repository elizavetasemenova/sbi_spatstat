"""Real-data case study: apply the amortized estimators, trained purely on
simulations, to a real crime point pattern, and validate against NUTS.

Data: the ``crimes`` point pattern from the geodanet example bundled with
libpysal (287 recorded crime locations in a road-network neighbourhood).  We do
NOT retrain: the same NPE and field U-Net used throughout are applied directly,
demonstrating that amortized inference transfers from the simulator to real data
in one forward pass.  NUTS on the same grid is the gold-standard reference.
"""

import json
import time
import numpy as np
import torch

import config as C
from sbilgcp import lgcp, field as F, diagnostics as diag
from sbilgcp.npe import NPE, CNNEmbedding
from sbilgcp import reference_mcmc as ref

torch.set_num_threads(4)
CRIMES = "/usr/local/lib/python3.11/dist-packages/libpysal/examples/geodanet/crimes.shp"
F_RES = C.RESULTS + "/real_results.json"
F_ARR = C.RESULTS + "/real_arrays.npz"


def load_grid(G=16):
    import geopandas as gpd, warnings
    warnings.filterwarnings("ignore")
    g = gpd.read_file(CRIMES)
    xy = np.array([(p.x, p.y) for p in g.geometry])
    mn = xy.min(0)
    rng = (xy.max(0) - mn).max()           # uniform scaling preserves geometry
    u = (xy - mn) / rng
    H, _, _ = np.histogram2d(u[:, 0], u[:, 1], bins=G, range=[[0, 1], [0, 1]])
    return H.astype(np.float32), u


def main():
    G = C.GRID
    y_grid, pts = load_grid(G)
    y = torch.as_tensor(y_grid[None], dtype=torch.float32)      # [1,G,G]
    sim = lgcp.LGCPSimulator(G=G)

    # --- amortized parameter posterior (trained model, one forward pass) ------
    npe = NPE(CNNEmbedding(embedding_dim=48), n_transforms=5, hidden=64)
    npe.load_state_dict(torch.load(C.F_NPE)); npe.eval()
    t0 = time.time()
    u = npe.sample(y, n_per=2000)
    theta_npe = lgcp.from_unconstrained(u.reshape(-1, 3)).numpy()
    t_npe_param = time.time() - t0

    # --- amortized intensity surface (trained field U-Net) --------------------
    fm = F.FieldUNet(rank=16); fm.load_state_dict(torch.load(C.RESULTS + "/field_unet.pt")); fm.eval()
    t0 = time.time()
    with torch.no_grad():
        mu, d, V = fm(y)
        Zsamp = F.sample_field(mu, d, V, 2000, rng=0)[0].numpy()   # [S, n]
    t_npe_field = time.time() - t0
    field_mean_npe = mu[0].numpy(); field_sd_npe = Zsamp.std(0)

    # --- gold-standard NUTS on the real grid ---------------------------------
    t0 = time.time()
    r_param = ref.run_nuts(y_grid, sim.coords, sim.area, num_warmup=1000, num_samples=1000, num_chains=3, seed=0)
    r_field = ref.run_nuts_field(y_grid, sim.coords, sim.area, num_warmup=800, num_samples=800, num_chains=2, seed=1)
    t_nuts = time.time() - t0
    theta_nuts = r_param["theta"]
    field_mean_nuts = r_field["Z"].mean(0); field_sd_nuts = r_field["Z"].std(0)

    # --- agreement ------------------------------------------------------------
    c2st = diag.c2st(theta_nuts, theta_npe[:len(theta_nuts)], seed=0)
    field_mean_corr = float(np.corrcoef(field_mean_npe, field_mean_nuts)[0, 1])
    field_sd_corr = float(np.corrcoef(field_sd_npe, field_sd_nuts)[0, 1])

    res = {
        "n_points": int(y_grid.sum()), "grid": G,
        "theta_npe_mean": theta_npe.mean(0).tolist(), "theta_npe_sd": theta_npe.std(0).tolist(),
        "theta_nuts_mean": theta_nuts.mean(0).tolist(), "theta_nuts_sd": theta_nuts.std(0).tolist(),
        "param_c2st": float(c2st),
        "field_mean_corr": field_mean_corr, "field_sd_corr": field_sd_corr,
        "rhat_param": r_param["rhat"], "rhat_field": r_field["rhat"],
        "t_npe_param_ms": 1e3 * t_npe_param, "t_npe_field_ms": 1e3 * t_npe_field,
        "t_nuts_s": t_nuts,
        "speedup": t_nuts / (t_npe_param + t_npe_field),
    }
    with open(F_RES, "w") as f:
        json.dump(res, f, indent=2)
    np.savez_compressed(
        F_ARR, y=y_grid, pts=pts,
        field_mean_npe=field_mean_npe.reshape(G, G), field_sd_npe=field_sd_npe.reshape(G, G),
        field_mean_nuts=field_mean_nuts.reshape(G, G), field_sd_nuts=field_sd_nuts.reshape(G, G),
        theta_npe=theta_npe, theta_nuts=theta_nuts,
    )
    print(f"crimes: {int(y_grid.sum())} points")
    print(f"NPE  theta mean {np.round(theta_npe.mean(0),3)} sd {np.round(theta_npe.std(0),3)}")
    print(f"NUTS theta mean {np.round(theta_nuts.mean(0),3)} sd {np.round(theta_nuts.std(0),3)}")
    print(f"param C2ST {c2st:.3f} | field mean corr {field_mean_corr:.3f} sd corr {field_sd_corr:.3f}")
    print(f"time: NPE {1e3*(t_npe_param+t_npe_field):.0f} ms vs NUTS {t_nuts:.0f} s  (speedup {res['speedup']:.0f}x)")
    print(f"saved -> {F_RES}")


if __name__ == "__main__":
    main()
