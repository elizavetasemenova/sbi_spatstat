"""Evaluate the saved spectral FMPE model on the saved test set (no retraining)."""
import sys, json
import numpy as np, torch
from scipy import stats as st
sys.path.insert(0, "/home/user/sbi_spatstat")
from spectral.fmpe import FMPE
from spectral.simulator import SpectralLGCP
from spectral.basis import SpectralBasis
from spectral import priors
torch.set_num_threads(4)
R = "/home/user/sbi_spatstat/results/"

ckpt = torch.load(R + "spec_fmpe.pt", weights_only=False); cfg = ckpt["cfg"]; ps = ckpt["prior_std"]
model = FMPE(target_dim=cfg["target_dim"], feat_dim=cfg["feat_dim"], cond_dim=192, hidden=768)
model.load_state_dict(ckpt["state"]); model.eval()
sim = SpectralLGCP(M=cfg["M"], nu=cfg["nu"]); basis = SpectralBasis(cfg["kmax"])

d = np.load(R + "spec_test.npz")
theta_true = d["theta"]; phi = d["phi"]; fields = d["fields"]
N = theta_true.shape[0]; nf = fields.shape[0]

S = model.sample(torch.tensor(phi), n_per=400, steps=80).numpy()   # [N,400,dim]
th_post = priors.from_unit(S[:, :, :3].reshape(-1, 3)).reshape(N, 400, 3)
th_mean = th_post.mean(1)
out = {"R2": {}, "post_sd": {}, "sbc_p": {}, "cov90": {}}
for p, nm in enumerate(["beta0", "sigma", "ell"]):
    r2 = 1 - ((th_mean[:, p]-theta_true[:, p])**2).sum()/((theta_true[:, p]-theta_true[:, p].mean())**2).sum()
    ranks = (th_post[:, :, p] < theta_true[:, p][:, None]).sum(1)
    cnt, _ = np.histogram(ranks, bins=20, range=(0, 401)); exp = ranks.size/20
    pval = float(st.chi2.sf(((cnt-exp)**2/exp).sum(), 19))
    lo = np.quantile(th_post[:, :, p], 0.05, 1); hi = np.quantile(th_post[:, :, p], 0.95, 1)
    cov = float(((theta_true[:, p] >= lo) & (theta_true[:, p] <= hi)).mean())
    out["R2"][nm] = float(r2); out["post_sd"][nm] = float(th_post[:, :, p].std(1).mean())
    out["sbc_p"][nm] = pval; out["cov90"][nm] = cov
    print(f"{nm}: R2 {r2:.3f}  post-sd {out['post_sd'][nm]:.3f}  SBC p {pval:.3f}  90%cov {cov:.2f}", flush=True)

cf = S[:nf, :, 3:].mean(1) * ps[None]
cors = []
for i in range(nf):
    Zr = basis.coeffs_to_field(cf[i], cfg["M"]) + th_mean[i, 0]
    Zt = basis.coeffs_to_field(basis.field_to_coeffs(fields[i]-theta_true[i, 0]), cfg["M"]) + theta_true[i, 0]
    cors.append(np.corrcoef(Zr.ravel(), Zt.ravel())[0, 1])
out["field_corr"] = float(np.mean(cors))
print(f"field posterior-mean vs low-pass corr {out['field_corr']:.3f}", flush=True)
json.dump(out, open(R + "spec_eval.json", "w"), indent=2)
print("saved")
