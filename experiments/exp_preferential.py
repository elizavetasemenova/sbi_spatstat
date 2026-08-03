"""Methodological extension: amortized inference under preferential sampling.

Trains a design-aware NPE (infers beta0, sigma, ell, gamma from the observed
counts *and* the observation mask) and a design-ignorant baseline (the standard
"ignorable design" analysis, trained on uniform-sampling simulations and applied
to preferentially-sampled data).  Shows that the naive analysis is biased while
the design-aware NPE is unbiased and calibrated, and recovers the preferential
strength gamma.
"""

import json
import time
import numpy as np
import torch

import config as C
from sbilgcp import lgcp, preferential as P, diagnostics as diag
from sbilgcp.npe import TrainConfig, train_npe

torch.set_num_threads(4)

N_TRAIN = 60000
N_TEST = 3000
N_POST = 1000
LEVELS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
SEED = 4321

F_PREF = C.RESULTS + "/pref_results.json"
F_PREF_ARR = C.RESULTS + "/pref_arrays.npz"
F_PREF_PS = C.RESULTS + "/pref_npe_ps.pt"
F_PREF_NAIVE = C.RESULTS + "/pref_npe_naive.pt"
F_PREF_DATA = C.RESULTS + "/pref_test.npz"


def posterior(model, data, pdim, n_per=N_POST, batch=256):
    data = torch.as_tensor(data, dtype=torch.float32)
    tf = P.from_unconstrained if pdim == 4 else lgcp.from_unconstrained
    out = []
    for s in range(0, data.shape[0], batch):
        u = model.sample(data[s:s + batch], n_per=n_per)
        nat = tf(u.reshape(-1, pdim)).reshape(u.shape[0], n_per, pdim)
        out.append(nat.numpy())
    return np.concatenate(out, 0)


def main():
    sim = P.PreferentialSimulator(G=C.GRID)

    # ---- data ---------------------------------------------------------------
    print("simulating preferential training data ...")
    rng = np.random.default_rng(SEED)
    theta_tr, X_tr = sim.simulate_dataset(N_TRAIN, rng, preferential=True)

    print("simulating uniform-design training data (for naive baseline) ...")
    rng = np.random.default_rng(SEED + 1)
    theta_tr_u, X_tr_u = sim.simulate_dataset(N_TRAIN, rng, preferential=False)

    print("simulating preferential test data ...")
    rng = np.random.default_rng(SEED + 2)
    theta_te, X_te = sim.simulate_dataset(N_TEST, rng, preferential=True)
    np.savez_compressed(F_PREF_DATA, theta=theta_te, X=X_te.numpy())

    reports = {}

    # ---- design-aware NPE (4 params) ---------------------------------------
    print("\n=== training design-aware NPE (infers gamma) ===")
    torch.manual_seed(0)
    ps = P.NPEk(P.CNN2ch(48), param_dim=4, n_transforms=5, hidden=64)
    cfg = TrainConfig(epochs=80, batch_size=256, lr=5e-4, patience=15, log_every=5)
    rep = train_npe(ps, theta_tr, X_tr, cfg, transform_fn=P.to_unconstrained)
    torch.save(ps.state_dict(), F_PREF_PS)
    reports["ps"] = {"val": rep.best_val, "wall": rep.wall_time}
    print(f"  best val {rep.best_val:.3f}")

    # ---- design-ignorant NPE (3 params, trained on uniform design) ---------
    print("\n=== training design-ignorant NPE (assumes ignorable sampling) ===")
    torch.manual_seed(0)
    naive = P.NPEk(P.CNN2ch(48), param_dim=3, n_transforms=5, hidden=64)
    rep_n = train_npe(naive, theta_tr_u[:, :3], X_tr_u, cfg, transform_fn=lgcp.to_unconstrained)
    torch.save(naive.state_dict(), F_PREF_NAIVE)
    reports["naive"] = {"val": rep_n.best_val, "wall": rep_n.wall_time}
    print(f"  best val {rep_n.best_val:.3f}")

    # ---- evaluate on preferential test data --------------------------------
    post_ps = posterior(ps, X_te, 4)          # [N,POST,4]
    post_naive = posterior(naive, X_te, 3)    # [N,POST,3]

    results = {"levels": LEVELS}
    # bias of posterior mean for beta0, sigma, ell
    def bias_and_cov(post, true, k):
        mean = post[:, :, :k].mean(1)
        bias = (mean - true[:, :k]).mean(0)
        rmse = np.sqrt(((mean - true[:, :k]) ** 2).mean(0))
        cov = diag.credible_coverage(true[:, :k], post[:, :, :k], LEVELS)
        ranks = diag.sbc_ranks(true[:, :k], post[:, :, :k])
        pvals = [diag.rank_uniformity_pvalue(ranks[:, p], N_POST) for p in range(k)]
        return bias, rmse, cov, pvals, ranks

    b_ps, r_ps, c_ps, p_ps, ranks_ps = bias_and_cov(post_ps, theta_te, 3)
    b_nv, r_nv, c_nv, p_nv, ranks_nv = bias_and_cov(post_naive, theta_te, 3)
    print("\nbias (beta0,sigma,ell):")
    print(f"  design-aware  {np.round(b_ps,3)}  90%cov {np.round(c_ps[LEVELS.index(0.9)],2)}  SBCp {np.round(p_ps,3)}")
    print(f"  design-ignorant {np.round(b_nv,3)}  90%cov {np.round(c_nv[LEVELS.index(0.9)],2)}  SBCp {np.round(p_nv,3)}")

    # gamma recovery (design-aware only)
    gamma_true = theta_te[:, 3]
    gamma_mean = post_ps[:, :, 3].mean(1)
    gamma_r2 = 1 - ((gamma_mean - gamma_true) ** 2).sum() / ((gamma_true - gamma_true.mean()) ** 2).sum()
    ranks_g = diag.sbc_ranks(theta_te[:, 3:4], post_ps[:, :, 3:4])
    gamma_p = diag.rank_uniformity_pvalue(ranks_g[:, 0], N_POST)
    print(f"gamma recovery: R2={gamma_r2:.2f}, SBC p={gamma_p:.3f}")

    # bias stratified by true gamma (5 bins)
    order = np.argsort(gamma_true)
    strat = {"gamma_centers": [], "bias_ps": [], "bias_naive": []}
    for b in np.array_split(order, 5):
        strat["gamma_centers"].append(float(gamma_true[b].mean()))
        strat["bias_ps"].append((post_ps[b][:, :, :3].mean(1) - theta_te[b, :3]).mean(0).tolist())
        strat["bias_naive"].append((post_naive[b][:, :, :3].mean(1) - theta_te[b, :3]).mean(0).tolist())

    results.update({
        "bias_ps": b_ps.tolist(), "bias_naive": b_nv.tolist(),
        "rmse_ps": r_ps.tolist(), "rmse_naive": r_nv.tolist(),
        "cov_ps": c_ps.tolist(), "cov_naive": c_nv.tolist(),
        "sbc_ps": p_ps, "sbc_naive": p_nv,
        "gamma_r2": float(gamma_r2), "gamma_sbc_p": float(gamma_p),
        "strat": strat, "reports": reports,
        "n_test": N_TEST,
    })
    with open(F_PREF, "w") as f:
        json.dump(results, f, indent=2)
    np.savez_compressed(
        F_PREF_ARR, theta_te=theta_te,
        post_mean_ps=post_ps.mean(1), post_mean_naive=post_naive.mean(1),
        post_std_ps=post_ps.std(1), post_std_naive=post_naive.std(1),
        ranks_ps=ranks_ps, ranks_naive=ranks_nv, ranks_gamma=ranks_g,
        gamma_true=gamma_true, gamma_mean=gamma_mean,
    )
    print(f"\nsaved -> {F_PREF}")


if __name__ == "__main__":
    main()
