"""v2 spectral field posterior: enriched grid-free conditioning (periodogram +
coarse counts + second-order summaries) with post-hoc quantile recalibration.

Trains once, then reports parameter recovery, field recovery, and calibration
BEFORE and AFTER a PIT-based recalibration fitted on a disjoint split. The
recalibration is the principled fix for the sigma/ell interval over-confidence."""
import sys, time, json
import numpy as np, torch
from scipy import stats as st
sys.path.insert(0, "/home/user/sbi_spatstat")
from spectral import data as D, priors
from spectral.fmpe import FMPE, train_fmpe
from spectral.simulator import SpectralLGCP
from spectral.basis import SpectralBasis
torch.set_num_threads(4)
R = "/home/user/sbi_spatstat/results/"
M, NU, KMAX, FKMAX = 64, 1.5, 6, 8
N_TRAIN, N_CAL, N_TEST, EPOCHS = 80000, 1500, 1500, 300
MODE = "spectral2"


def batched_sample(model, phi_np, n_per=300, steps=80, bs=250):
    ph = torch.tensor(phi_np); out = []
    with torch.no_grad():
        for s in range(0, ph.shape[0], bs):
            out.append(model.sample(ph[s:s + bs], n_per=n_per, steps=steps).numpy())
    return np.concatenate(out, 0)


def coverage(post, true, lo=0.05, hi=0.95):
    ql = np.quantile(post, lo, 1); qh = np.quantile(post, hi, 1)
    return float(((true >= ql) & (true <= qh)).mean())


def sbc_p(post, true, nb=20):
    ranks = (post < true[:, None]).sum(1); npd = post.shape[1]
    cnt, _ = np.histogram(ranks, bins=nb, range=(0, npd + 1)); exp = ranks.size / nb
    return float(st.chi2.sf(((cnt - exp) ** 2 / exp).sum(), nb - 1))


def main():
    sim = SpectralLGCP(M=M, nu=NU); basis = SpectralBasis(KMAX)
    t0 = time.time()
    tr = D.generate(N_TRAIN, seed=1, sim=sim, basis=basis, feat_kmax=FKMAX, feature_mode=MODE)
    ps = tr["prior_std"]
    cal = D.generate(N_CAL, seed=3, prior_std=ps, sim=sim, basis=basis, feat_kmax=FKMAX, feature_mode=MODE)
    te = D.generate(N_TEST, seed=2, prior_std=ps, sim=sim, basis=basis, feat_kmax=FKMAX, feature_mode=MODE)
    print(f"data gen {time.time()-t0:.0f}s  target {tr['x1'].shape}  feat {tr['phi'].shape}", flush=True)

    model = FMPE(target_dim=tr["x1"].shape[1], feat_dim=tr["phi"].shape[1], cond_dim=192, hidden=768)
    h = train_fmpe(model, tr["x1"], tr["phi"], epochs=EPOCHS, batch=256, lr=5e-4, patience=30, log_every=25)
    torch.save({"state": model.state_dict(), "prior_std": ps,
                "cfg": dict(M=M, nu=NU, kmax=KMAX, fkmax=FKMAX, mode=MODE,
                            target_dim=tr["x1"].shape[1], feat_dim=tr["phi"].shape[1])},
               R + "spec_v2.pt")
    print(f"train {h['wall']:.0f}s best val {h['best_val']:.4f}", flush=True)
    model.eval()

    # posterior draws on cal + test
    Sc = batched_sample(model, cal["phi"], n_per=400)
    Ste = batched_sample(model, te["phi"], n_per=400)
    thc = priors.from_unit(Sc[:, :, :3].reshape(-1, 3)).reshape(N_CAL, 400, 3)
    tht = priors.from_unit(Ste[:, :, :3].reshape(-1, 3)).reshape(N_TEST, 400, 3)
    out = {"R2": {}, "cov_pre": {}, "cov_post": {}, "sbc_pre": {}, "sbc_post": {},
           "recal_q": {}}
    for p, nm in enumerate(["beta0", "sigma", "ell"]):
        tt = te["theta"][:, p]; ct = cal["theta"][:, p]
        r2 = 1 - ((tht[:, :, p].mean(1) - tt) ** 2).sum() / ((tt - tt.mean()) ** 2).sum()
        # PIT on cal: u_i = P(draw < true); Ghat^{-1}(a) = empirical a-quantile of u
        u_cal = (thc[:, :, p] < ct[:, None]).mean(1)
        q_lo = float(np.quantile(u_cal, 0.05)); q_hi = float(np.quantile(u_cal, 0.95))
        out["R2"][nm] = float(r2)
        out["cov_pre"][nm] = coverage(tht[:, :, p], tt)
        out["cov_post"][nm] = coverage(tht[:, :, p], tt, q_lo, q_hi)
        out["sbc_pre"][nm] = sbc_p(tht[:, :, p], tt)
        # recalibrated PIT on test = Ghat(u_test); should be ~uniform
        u_te = (tht[:, :, p] < tt[:, None]).mean(1)
        u_re = np.searchsorted(np.sort(u_cal), u_te) / u_cal.size
        cnt, _ = np.histogram(u_re, bins=20, range=(0, 1)); exp = u_re.size / 20
        out["sbc_post"][nm] = float(st.chi2.sf(((cnt - exp) ** 2 / exp).sum(), 19))
        out["recal_q"][nm] = [q_lo, q_hi]
        print(f"{nm}: R2 {r2:.3f}  cov90 {out['cov_pre'][nm]:.2f}->{out['cov_post'][nm]:.2f}  "
              f"SBCp {out['sbc_pre'][nm]:.3f}->{out['sbc_post'][nm]:.3f}", flush=True)

    # field recovery
    cf = Ste[:300, :, 3:].mean(1) * ps[None]
    cors = []
    for i in range(300):
        Zr = basis.coeffs_to_field(cf[i], M) + tht[i, :, 0].mean()
        Zt = basis.coeffs_to_field(basis.field_to_coeffs(te["fields"][i] - te["theta"][i, 0]), M) + te["theta"][i, 0]
        cors.append(np.corrcoef(Zr.ravel(), Zt.ravel())[0, 1])
    out["field_corr"] = float(np.mean(cors))
    print(f"field corr {out['field_corr']:.3f}", flush=True)
    # save test set + samples for figures
    np.savez_compressed(R + "spec_v2_test.npz", theta=te["theta"], phi=te["phi"],
                        fields=te["fields"][:300], prior_std=ps)
    json.dump(out, open(R + "spec_v2_eval.json", "w"), indent=2)
    print("saved", flush=True)


if __name__ == "__main__":
    main()
