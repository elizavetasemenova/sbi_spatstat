"""Grid-resolution sweep: train the count-baseline amortizer at several grid
resolutions and record ell R^2 on a common test set. Shows range recovery
degrading as the count grid coarsens (discretization), with grid-free spectral
conditioning as the reference. Same simulator/target/sampler; only the summary."""
import sys, json, time
import numpy as np, torch
sys.path.insert(0, "/home/user/sbi_spatstat")
from spectral import data as D, priors
from spectral.fmpe import FMPE, train_fmpe
from spectral.simulator import SpectralLGCP
from spectral.basis import SpectralBasis
torch.set_num_threads(4)
R = "/home/user/sbi_spatstat/results/"
M, NU, KMAX, FKMAX = 64, 1.5, 6, 8
N_TRAIN, N_TEST, EPOCHS = 80000, 1000, 300
GRIDS = [8, 14]                                # 20 already evaluated separately


def eval_ell(model, te, N_TEST):
    th_mean = np.zeros((N_TEST, 3)); post = np.zeros((N_TEST, 200, 3))
    ph = torch.tensor(te["phi"])
    with torch.no_grad():
        for s in range(0, N_TEST, 200):
            S = model.sample(ph[s:s + 200], n_per=200, steps=80).numpy()
            th = priors.from_unit(S[:, :, :3].reshape(-1, 3)).reshape(-1, 200, 3)
            post[s:s + th.shape[0]] = th; th_mean[s:s + th.shape[0]] = th.mean(1)
    tt = te["theta"]
    r2 = {}
    for p, nm in enumerate(["beta0", "sigma", "ell"]):
        r2[nm] = float(1 - ((th_mean[:, p]-tt[:, p])**2).sum()/((tt[:, p]-tt[:, p].mean())**2).sum())
    return r2


def main():
    sim = SpectralLGCP(M=M, nu=NU); basis = SpectralBasis(KMAX)
    res = {}
    for G in GRIDS:
        t0 = time.time()
        tr = D.generate(N_TRAIN, seed=1, sim=sim, basis=basis, feat_kmax=FKMAX,
                        feature_mode="counts", count_grid=G)
        ps = tr["prior_std"]
        te = D.generate(N_TEST, seed=2, prior_std=ps, sim=sim, basis=basis, feat_kmax=FKMAX,
                        feature_mode="counts", count_grid=G)
        model = FMPE(target_dim=tr["x1"].shape[1], feat_dim=tr["phi"].shape[1], cond_dim=192, hidden=768)
        h = train_fmpe(model, tr["x1"], tr["phi"], epochs=EPOCHS, batch=256, lr=5e-4,
                       patience=30, log_every=50)
        model.eval()
        r2 = eval_ell(model, te, N_TEST)
        res[str(G)] = r2
        print(f"[G={G}] ell R2 {r2['ell']:.3f}  ({time.time()-t0:.0f}s)", flush=True)
        json.dump(res, open(R + "spec_sweep.json", "w"), indent=2)
    print("done", flush=True)


if __name__ == "__main__":
    main()
