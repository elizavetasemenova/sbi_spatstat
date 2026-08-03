"""Lean eval + PIT recalibration for the saved v2 model. Batched no-grad
sampling; disjoint calibration (seed 3) and test (seed 2) splits."""
import sys, json
import numpy as np, torch
from scipy import stats as st
sys.path.insert(0, "/home/user/sbi_spatstat")
from spectral import data as D, priors
from spectral.fmpe import FMPE
from spectral.simulator import SpectralLGCP
from spectral.basis import SpectralBasis
torch.set_num_threads(4)
R = "/home/user/sbi_spatstat/results/"
NCAL, NTEST, NPER = 1200, 1200, 200

ck = torch.load(R + "spec_v2.pt", weights_only=False); cfg = ck["cfg"]; ps = ck["prior_std"]
model = FMPE(target_dim=cfg["target_dim"], feat_dim=cfg["feat_dim"], cond_dim=192, hidden=768)
model.load_state_dict(ck["state"]); model.eval()
sim = SpectralLGCP(M=cfg["M"], nu=cfg["nu"]); basis = SpectralBasis(cfg["kmax"])


def bsample(phi_np, n_per=NPER, steps=80, bs=120):
    ph = torch.tensor(phi_np); out = []
    with torch.no_grad():
        for s in range(0, ph.shape[0], bs):
            out.append(model.sample(ph[s:s + bs], n_per=n_per, steps=steps).numpy())
    return np.concatenate(out, 0)


def cov(post, true, lo, hi):
    return float(((true >= np.quantile(post, lo, 1)) & (true <= np.quantile(post, hi, 1))).mean())


def sbc(post, true, nb=20):
    r = (post < true[:, None]).sum(1); n = post.shape[1]
    c, _ = np.histogram(r, bins=nb, range=(0, n + 1)); e = r.size / nb
    return float(st.chi2.sf(((c - e) ** 2 / e).sum(), nb - 1))


cal = D.generate(NCAL, seed=3, prior_std=ps, sim=sim, basis=basis, feat_kmax=cfg["fkmax"], feature_mode=cfg["mode"])
te = D.generate(NTEST, seed=2, prior_std=ps, sim=sim, basis=basis, feat_kmax=cfg["fkmax"], feature_mode=cfg["mode"])
print("sampling...", flush=True)
Sc = bsample(cal["phi"]); Ste = bsample(te["phi"])
thc = priors.from_unit(Sc[:, :, :3].reshape(-1, 3)).reshape(NCAL, NPER, 3)
tht = priors.from_unit(Ste[:, :, :3].reshape(-1, 3)).reshape(NTEST, NPER, 3)

out = {"R2": {}, "cov_pre": {}, "cov_post": {}, "sbc_pre": {}, "sbc_post": {}, "recal_q": {}}
for p, nm in enumerate(["beta0", "sigma", "ell"]):
    tt = te["theta"][:, p]; ct = cal["theta"][:, p]
    r2 = 1 - ((tht[:, :, p].mean(1) - tt) ** 2).sum() / ((tt - tt.mean()) ** 2).sum()
    u_cal = (thc[:, :, p] < ct[:, None]).mean(1)
    q_lo = float(np.quantile(u_cal, 0.05)); q_hi = float(np.quantile(u_cal, 0.95))
    u_te = (tht[:, :, p] < tt[:, None]).mean(1)
    u_re = np.searchsorted(np.sort(u_cal), u_te) / u_cal.size
    c2, _ = np.histogram(u_re, bins=20, range=(0, 1)); e2 = u_re.size / 20
    out["R2"][nm] = float(r2)
    out["cov_pre"][nm] = cov(tht[:, :, p], tt, 0.05, 0.95)
    out["cov_post"][nm] = cov(tht[:, :, p], tt, q_lo, q_hi)
    out["sbc_pre"][nm] = sbc(tht[:, :, p], tt)
    out["sbc_post"][nm] = float(st.chi2.sf(((c2 - e2) ** 2 / e2).sum(), 19))
    out["recal_q"][nm] = [q_lo, q_hi]
    print(f"{nm}: R2 {r2:.3f}  cov90 {out['cov_pre'][nm]:.2f}->{out['cov_post'][nm]:.2f}  "
          f"SBCp {out['sbc_pre'][nm]:.3f}->{out['sbc_post'][nm]:.3f}", flush=True)

cf = Ste[:300, :, 3:].mean(1) * ps[None]
cors = [np.corrcoef((basis.coeffs_to_field(cf[i], cfg["M"]) + tht[i, :, 0].mean()).ravel(),
                    (basis.coeffs_to_field(basis.field_to_coeffs(te["fields"][i] - te["theta"][i, 0]), cfg["M"]) + te["theta"][i, 0]).ravel())[0, 1]
        for i in range(300)]
out["field_corr"] = float(np.mean(cors))
print(f"field corr {out['field_corr']:.3f}", flush=True)
json.dump(out, open(R + "spec_v2_eval.json", "w"), indent=2)
print("saved", flush=True)
