"""Train the main spectral FMPE posterior at scale and evaluate recovery + SBC."""

import sys, time, json
import numpy as np
import torch

sys.path.insert(0, "/home/user/sbi_spatstat")
from spectral import data as D, priors
from spectral.fmpe import FMPE, train_fmpe
from spectral.simulator import SpectralLGCP
from spectral.basis import SpectralBasis

torch.set_num_threads(4)
R = "/home/user/sbi_spatstat/results/"
M, NU, KMAX, FKMAX = 64, 1.5, 6, 8
N_TRAIN, N_TEST, EPOCHS = 80000, 3000, 300


def main():
    sim = SpectralLGCP(M=M, nu=NU); basis = SpectralBasis(KMAX)
    t0 = time.time()
    tr = D.generate(N_TRAIN, seed=1, sim=sim, basis=basis, feat_kmax=FKMAX)
    ps = tr["prior_std"]
    te = D.generate(N_TEST, seed=2, prior_std=ps, sim=sim, basis=basis, feat_kmax=FKMAX)
    print(f"data gen {time.time()-t0:.0f}s  target {tr['x1'].shape}  feat {tr['phi'].shape}", flush=True)
    np.savez_compressed(R + "spec_test.npz", theta=te["theta"], x1=te["x1"], phi=te["phi"],
                        fields=te["fields"][:600], prior_std=ps)

    model = FMPE(target_dim=tr["x1"].shape[1], feat_dim=tr["phi"].shape[1], cond_dim=192, hidden=768)
    h = train_fmpe(model, tr["x1"], tr["phi"], epochs=EPOCHS, batch=256, lr=5e-4,
                   patience=30, log_every=25)
    torch.save({"state": model.state_dict(), "prior_std": ps,
                "cfg": dict(M=M, nu=NU, kmax=KMAX, fkmax=FKMAX,
                            target_dim=tr["x1"].shape[1], feat_dim=tr["phi"].shape[1])},
               R + "spec_fmpe.pt")
    print(f"train {h['wall']:.0f}s best val {h['best_val']:.4f}", flush=True)

    # ---- evaluate on test -------------------------------------------------
    model.eval()
    S = model.sample(torch.tensor(te["phi"]), n_per=500, steps=100).numpy()   # [N,500,dim]
    th_post = priors.from_unit(S[:, :, :3].reshape(-1, 3)).reshape(N_TEST, 500, 3)
    th_true = te["theta"]; th_mean = th_post.mean(1)
    out = {"R2": {}, "post_sd": {}, "sbc_p": {}}
    from scipy import stats as st
    for p, nm in enumerate(["beta0", "sigma", "ell"]):
        r2 = 1 - ((th_mean[:, p] - th_true[:, p]) ** 2).sum() / ((th_true[:, p] - th_true[:, p].mean()) ** 2).sum()
        ranks = (th_post[:, :, p] < th_true[:, p][:, None]).sum(1)
        cnt, _ = np.histogram(ranks, bins=20, range=(0, 501)); exp = ranks.size / 20
        chi2 = ((cnt - exp) ** 2 / exp).sum(); pval = st.chi2.sf(chi2, 19)
        out["R2"][nm] = float(r2); out["post_sd"][nm] = float(th_post[:, :, p].std(1).mean())
        out["sbc_p"][nm] = float(pval)
        print(f"{nm}: R2 {r2:.3f}  post-sd {out['post_sd'][nm]:.3f}  SBC p {pval:.3f}", flush=True)

    # field recovery vs true low-pass
    cf = S[:, :, 3:].mean(1) * ps[None]
    cors = []
    for i in range(300):
        Zr = basis.coeffs_to_field(cf[i], M) + th_mean[i, 0]
        Zt = basis.coeffs_to_field(basis.field_to_coeffs(te["fields"][i] - th_true[i, 0]), M) + th_true[i, 0]
        cors.append(np.corrcoef(Zr.ravel(), Zt.ravel())[0, 1])
    out["field_corr"] = float(np.mean(cors))
    print(f"field posterior-mean vs low-pass corr {out['field_corr']:.3f}", flush=True)
    json.dump(out, open(R + "spec_eval.json", "w"), indent=2)
    print("saved", flush=True)


if __name__ == "__main__":
    main()
