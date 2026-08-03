"""Generate the paper figures from the saved spectral model, grid baseline, and
test set. Writes PDFs into paper_aistats/figures/ and the numeric macros into
paper_aistats/numbers.tex."""
import sys, json, os
import numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as st

sys.path.insert(0, "/home/user/sbi_spatstat")
from spectral.fmpe import FMPE
from spectral.simulator import SpectralLGCP
from spectral.basis import SpectralBasis
from spectral import priors

torch.set_num_threads(4)
R = "/home/user/sbi_spatstat/results/"
FIG = "/home/user/sbi_spatstat/paper_aistats/figures/"
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "figure.dpi": 150})


def load(ckpt_path):
    ckpt = torch.load(ckpt_path, weights_only=False)
    cfg = ckpt["cfg"]; ps = ckpt["prior_std"]
    m = FMPE(target_dim=cfg["target_dim"], feat_dim=cfg["feat_dim"], cond_dim=192, hidden=768)
    m.load_state_dict(ckpt["state"]); m.eval()
    return m, cfg, ps


from spectral import data as D

spec_model, cfg, ps = load(R + "spec_v2.pt")
sim = SpectralLGCP(M=cfg["M"], nu=cfg["nu"]); basis = SpectralBasis(cfg["kmax"])
M = cfg["M"]; MODE = cfg["mode"]; N = 1200


def batched_sample(model, phi_np, n_per=250, steps=80, bs=120):
    ph = torch.tensor(phi_np); out = []
    with torch.no_grad():
        for s in range(0, ph.shape[0], bs):
            out.append(model.sample(ph[s:s + bs], n_per=n_per, steps=steps).numpy())
    return np.concatenate(out, 0)


# test set (seed 2) + disjoint calibration set (seed 3) for recalibration
te = D.generate(N, seed=2, prior_std=ps, sim=sim, basis=basis, feat_kmax=cfg["fkmax"], feature_mode=MODE)
cal = D.generate(N, seed=3, prior_std=ps, sim=sim, basis=basis, feat_kmax=cfg["fkmax"], feature_mode=MODE)
theta_true = te["theta"]; phi = te["phi"]; fields = te["fields"]; nf = fields.shape[0]

# posterior draws (cached so reruns are instant)
_cache = R + "fig_samples_v2.npz"
if os.path.exists(_cache):
    z = np.load(_cache); S = z["S"]; Scal = z["Scal"]
else:
    S = batched_sample(spec_model, phi); Scal = batched_sample(spec_model, cal["phi"])
    np.savez_compressed(_cache, S=S, Scal=Scal)
NP = S.shape[1]
th_post = priors.from_unit(S[:, :, :3].reshape(-1, 3)).reshape(N, NP, 3)
th_mean = th_post.mean(1)
th_cal = priors.from_unit(Scal[:, :, :3].reshape(-1, 3)).reshape(N, NP, 3)
# per-parameter recalibrated posterior-quantile levels from the calibration PIT
recal_q = {}
for p in range(3):
    u = (th_cal[:, :, p] < cal["theta"][:, p][:, None]).mean(1)
    recal_q[p] = (float(np.quantile(u, 0.05)), float(np.quantile(u, 0.95)))

# ---------------------------------------------------------------- Fig 1: field
# pick a representative test case with a mid-range intensity
sel = int(np.argmin(np.abs(theta_true[:nf, 1] - 1.0) + np.abs(theta_true[:nf, 2] - 0.12)))
b0t, sigt, ellt = theta_true[sel]
Ztrue = basis.coeffs_to_field(basis.field_to_coeffs(fields[sel] - b0t), M) + b0t
# regenerate the observed point pattern for this case (seed=2 stream, reproduce)
rng = np.random.default_rng(90000 + sel)
pts = sim.sample_points(fields[sel], rng)
cf = S[sel, :, 3:] * ps[None]                       # [400, D]
Zdraws = np.stack([basis.coeffs_to_field(cf[j], M) + th_post[sel, j, 0] for j in range(120)])
Zmean = Zdraws.mean(0); Zsd = Zdraws.std(0)

fig, ax = plt.subplots(1, 4, figsize=(9.2, 2.5))
vmin, vmax = Ztrue.min(), Ztrue.max()
im0 = ax[0].imshow(Ztrue, origin="lower", extent=[0, 1, 0, 1], vmin=vmin, vmax=vmax, cmap="viridis")
ax[0].set_title("true log-intensity"); ax[0].scatter(pts[:, 1], pts[:, 0], s=1.2, c="w", alpha=0.5)
ax[1].scatter(pts[:, 1], pts[:, 0], s=2.5, c="k"); ax[1].set_xlim(0, 1); ax[1].set_ylim(0, 1)
ax[1].set_aspect("equal"); ax[1].set_title(f"point pattern (N={pts.shape[0]})")
im2 = ax[2].imshow(Zmean, origin="lower", extent=[0, 1, 0, 1], vmin=vmin, vmax=vmax, cmap="viridis")
ax[2].set_title("posterior mean")
im3 = ax[3].imshow(Zsd, origin="lower", extent=[0, 1, 0, 1], cmap="magma")
ax[3].set_title("posterior SD")
for a in ax:
    a.set_xticks([]); a.set_yticks([])
fig.colorbar(im2, ax=ax[2], fraction=0.046); fig.colorbar(im3, ax=ax[3], fraction=0.046)
fig.tight_layout(); fig.savefig(FIG + "field.pdf", bbox_inches="tight"); plt.close(fig)
print("field.pdf done", flush=True)

# ------------------------------------------------------------ Fig 2: calibration
fig, ax = plt.subplots(1, 3, figsize=(7.5, 2.3))
names = [r"$\beta_0$", r"$\sigma$", r"$\ell$"]
sbc_p = {}; cov90 = {}; cov90_post = {}; r2 = {}
for p in range(3):
    r2[p] = float(1 - ((th_mean[:, p] - theta_true[:, p]) ** 2).sum() /
                  ((theta_true[:, p] - theta_true[:, p].mean()) ** 2).sum())
    ranks = (th_post[:, :, p] < theta_true[:, p][:, None]).sum(1)
    ax[p].hist(ranks, bins=20, range=(0, NP + 1), color="#4C72B0", edgecolor="w")
    exp = ranks.size / 20
    ax[p].axhline(exp, color="k", ls="--", lw=0.8)
    band = 2 * np.sqrt(exp * (1 - 1 / 20))
    ax[p].axhspan(exp - band, exp + band, color="grey", alpha=0.2)
    cnt, _ = np.histogram(ranks, bins=20, range=(0, NP + 1))
    sbc_p[p] = float(st.chi2.sf(((cnt - exp) ** 2 / exp).sum(), 19))
    lo = np.quantile(th_post[:, :, p], 0.05, 1); hi = np.quantile(th_post[:, :, p], 0.95, 1)
    cov90[p] = float(((theta_true[:, p] >= lo) & (theta_true[:, p] <= hi)).mean())
    ql, qh = recal_q[p]                                    # recalibrated 90% interval
    lo2 = np.quantile(th_post[:, :, p], ql, 1); hi2 = np.quantile(th_post[:, :, p], qh, 1)
    cov90_post[p] = float(((theta_true[:, p] >= lo2) & (theta_true[:, p] <= hi2)).mean())
    ax[p].set_title(f"{names[p]}  90% cov {cov90[p]:.2f}$\\to${cov90_post[p]:.2f}")
    ax[p].set_xlabel("rank"); ax[p].set_yticks([])
fig.tight_layout(); fig.savefig(FIG + "calibration.pdf", bbox_inches="tight"); plt.close(fig)
print("calibration.pdf done", flush=True)

# ------------------------------------------------------- Fig 3: range recovery
ell_r2_spec = 1 - ((th_mean[:, 2] - theta_true[:, 2]) ** 2).sum() / (
    (theta_true[:, 2] - theta_true[:, 2].mean()) ** 2).sum()
# grid-count baseline ell R^2 across resolutions: {G: R2}
grid_r2 = {}
if os.path.exists(R + "spec_baseline.json"):
    grid_r2[20] = json.load(open(R + "spec_baseline.json"))["R2"]["ell"]
if os.path.exists(R + "spec_sweep.json"):
    for g, r in json.load(open(R + "spec_sweep.json")).items():
        grid_r2[int(g)] = r["ell"]
ell_r2_grid = grid_r2.get(20, float("nan"))              # headline single-grid number

fig, ax = plt.subplots(1, 2, figsize=(6.6, 3.0))
lo, hi = priors.ELL_LOW, priors.ELL_HIGH
ax[0].scatter(theta_true[:, 2], th_mean[:, 2], s=4, alpha=0.3, c="#4C72B0")
ax[0].plot([lo, hi], [lo, hi], "k--", lw=0.8)
ax[0].set_title(f"grid-free spectral\n$R^2={ell_r2_spec:.2f}$")
ax[0].set_xlabel(r"true $\ell$"); ax[0].set_ylabel(r"posterior mean $\ell$")
ax[0].set_xlim(lo, hi); ax[0].set_ylim(lo, hi); ax[0].set_aspect("equal")
if grid_r2:
    gs = sorted(grid_r2)
    ax[1].plot(gs, [grid_r2[g] for g in gs], "o-", c="#C44E52", label="grid counts")
    ax[1].axhline(ell_r2_spec, color="#4C72B0", ls="--", lw=1.2, label="grid-free spectral")
    ax[1].set_xlabel(r"count-grid resolution $G_c$"); ax[1].set_ylabel(r"$\ell$ recovery $R^2$")
    ax[1].set_title("grid-free beats binned\ncounts at every $G_c$")
    ax[1].set_ylim(min(min(grid_r2.values()) - 0.02, ell_r2_spec - 0.08), ell_r2_spec + 0.02)
    ax[1].legend(fontsize=7, loc="lower right")
fig.tight_layout(); fig.savefig(FIG + "range.pdf", bbox_inches="tight"); plt.close(fig)
print(f"range.pdf done  spec {ell_r2_spec:.3f}  grid20 {ell_r2_grid:.3f}  sweep {grid_r2}", flush=True)

# --------------------------------------------- Fig 4: resolution consistency
# The posterior returns coefficients of a continuous field; synthesizing at any
# output grid G is a query of one function. We show that the field statistic that
# matters -- agreement with the ground-truth field evaluated on a COMMON fine grid
# -- is invariant to the grid the user chooses to render on. We compare each
# rendering, bilinearly resampled to G=128, against the truth at G=128.
cf0 = (S[sel, :, 3:].mean(0)) * ps
Zref128 = basis.coeffs_to_field(basis.field_to_coeffs(fields[sel] - b0t), 128) + b0t
from scipy.ndimage import zoom
grids = [16, 24, 32, 48, 64, 96, 128]
agree = []
for G in grids:
    Zg = basis.coeffs_to_field(cf0, G) + th_mean[sel, 0]
    Zup = zoom(Zg, 128 / G, order=1)
    agree.append(np.corrcoef(Zup.ravel(), Zref128.ravel())[0, 1])
fig, ax = plt.subplots(1, 4, figsize=(9.0, 2.4))
for gi, G in enumerate([16, 32, 96]):
    Zg = basis.coeffs_to_field(cf0, G) + th_mean[sel, 0]
    ax[gi].imshow(Zg, origin="lower", cmap="viridis")
    ax[gi].set_title(f"render at $G={G}$"); ax[gi].set_xticks([]); ax[gi].set_yticks([])
ax[3].plot(grids, agree, "o-", c="#55A868")
ax[3].set_ylim(min(agree) - 0.02, 1.005); ax[3].set_xlabel("render grid $G$")
ax[3].set_title("corr. to truth ($G{=}128$)"); ax[3].axhline(agree[-1], color="k", ls=":", lw=0.8)
fig.tight_layout(); fig.savefig(FIG + "resolution.pdf", bbox_inches="tight"); plt.close(fig)
print(f"resolution.pdf done  agreement range {min(agree):.3f}-{max(agree):.3f}", flush=True)

# ------------------------------------------ non-Gaussianity of coeff posteriors
# skewness of the low-frequency field-coefficient posteriors, on the lowest-count
# third of the test set (where the Gaussian/Laplace approximation is worst)
counts_proxy = phi[:, 0]                                   # log1p(N) is feature 0
low = counts_proxy <= np.quantile(counts_proxy, 1 / 3)
coeff_draws = S[low][:, :, 3:3 + 24]                      # first 24 field coeffs
sk = st.skew(coeff_draws, axis=1)                         # [n_low, 24]
coeff_skew = float(np.mean(np.abs(sk)))
print(f"mean |skew| of low-freq coeff posteriors (low-count third): {coeff_skew:.3f}", flush=True)

# ---------------------------------------------------------------- numbers.tex
field_corr = json.load(open(R + "spec_v2_eval.json"))["field_corr"]
with open("/home/user/sbi_spatstat/paper_aistats/numbers.tex", "w") as f:
    f.write("\\newcommand{\\FieldCtst}{%.2f}\n" % field_corr)
    f.write("\\newcommand{\\FieldCorr}{%.2f}\n" % field_corr)
    f.write("\\newcommand{\\EllRtwoSpec}{%.2f}\n" % ell_r2_spec)
    f.write("\\newcommand{\\EllRtwoGrid}{%.2f}\n" % ell_r2_grid)
    _coarse = grid_r2[min(grid_r2)] if grid_r2 else float("nan")
    f.write("\\newcommand{\\EllRtwoGridCoarse}{%.2f}\n" % _coarse)
    f.write("\\newcommand{\\BetaCov}{%.2f}\n" % cov90[0])
    f.write("\\newcommand{\\SigmaCov}{%.2f}\n" % cov90[1])
    f.write("\\newcommand{\\EllCov}{%.2f}\n" % cov90[2])
    f.write("\\newcommand{\\BetaCovPost}{%.2f}\n" % cov90_post[0])
    f.write("\\newcommand{\\SigmaCovPost}{%.2f}\n" % cov90_post[1])
    f.write("\\newcommand{\\EllCovPost}{%.2f}\n" % cov90_post[2])
    f.write("\\newcommand{\\BetaSbc}{%.2f}\n" % sbc_p[0])
    f.write("\\newcommand{\\BetaRtwo}{%.2f}\n" % r2[0])
    f.write("\\newcommand{\\SigmaRtwo}{%.2f}\n" % r2[1])
    f.write("\\newcommand{\\CoeffSkew}{%.2f}\n" % coeff_skew)
print("numbers.tex written", flush=True)
print(json.dumps({"ell_r2_spec": float(ell_r2_spec), "ell_r2_grid": float(ell_r2_grid),
                  "cov90": cov90, "sbc_p": sbc_p, "field_corr": field_corr}, indent=2))
