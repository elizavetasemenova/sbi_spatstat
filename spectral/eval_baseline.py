"""Lean eval of the saved grid-count baseline: batched, no-grad sampling on a
test subset. Produces the baseline ell R^2 for the range-recovery comparison."""
import sys, json
import numpy as np, torch
from scipy import stats as st
sys.path.insert(0, "/home/user/sbi_spatstat")
from spectral.fmpe import FMPE
from spectral.simulator import SpectralLGCP
from spectral.basis import SpectralBasis
from spectral import data as D, priors
torch.set_num_threads(4)
R = "/home/user/sbi_spatstat/results/"

ckpt = torch.load(R + "spec_baseline.pt", weights_only=False); cfg = ckpt["cfg"]; ps = ckpt["prior_std"]
model = FMPE(target_dim=cfg["target_dim"], feat_dim=cfg["feat_dim"], cond_dim=192, hidden=768)
model.load_state_dict(ckpt["state"]); model.eval()
sim = SpectralLGCP(M=cfg["M"], nu=cfg["nu"]); basis = SpectralBasis(cfg["kmax"])

NT = 1000
te = D.generate(NT, seed=2, prior_std=ps, sim=sim, basis=basis, feat_kmax=cfg["fkmax"],
                feature_mode="counts")
phi = torch.tensor(te["phi"]); theta_true = te["theta"]

th_mean = np.zeros((NT, 3)); post = np.zeros((NT, 200, 3))
with torch.no_grad():
    for s in range(0, NT, 200):
        S = model.sample(phi[s:s + 200], n_per=200, steps=80).numpy()
        th = priors.from_unit(S[:, :, :3].reshape(-1, 3)).reshape(-1, 200, 3)
        post[s:s + th.shape[0]] = th; th_mean[s:s + th.shape[0]] = th.mean(1)

out = {"R2": {}, "cov90": {}, "sbc_p": {}}
for p, nm in enumerate(["beta0", "sigma", "ell"]):
    r2 = 1 - ((th_mean[:, p]-theta_true[:, p])**2).sum()/((theta_true[:, p]-theta_true[:, p].mean())**2).sum()
    lo = np.quantile(post[:, :, p], 0.05, 1); hi = np.quantile(post[:, :, p], 0.95, 1)
    cov = float(((theta_true[:, p] >= lo) & (theta_true[:, p] <= hi)).mean())
    ranks = (post[:, :, p] < theta_true[:, p][:, None]).sum(1)
    cnt, _ = np.histogram(ranks, bins=20, range=(0, 201)); exp = ranks.size/20
    out["R2"][nm] = float(r2); out["cov90"][nm] = cov
    out["sbc_p"][nm] = float(st.chi2.sf(((cnt-exp)**2/exp).sum(), 19))
    print(f"{nm}: R2 {r2:.3f}  90%cov {cov:.2f}", flush=True)
json.dump(out, open(R + "spec_baseline.json", "w"), indent=2)
print("saved", flush=True)
