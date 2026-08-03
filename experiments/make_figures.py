"""Generate all publication figures and LaTeX tables from saved results."""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import config as C
from sbilgcp import lgcp, diagnostics as diag

# ---- validated, colorblind-safe palette (from dataviz reference) -----------
BLUE = "#2a78d6"; AQUA = "#1baf7a"; YELLOW = "#eda100"
VIOLET = "#4a3aa7"; RED = "#e34948"; ORANGE = "#eb6834"; GREY = "#7a7a76"
INK = "#0b0b0b"; INK2 = "#52514e"

plt.rcParams.update({
    "figure.dpi": 140, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
    "axes.edgecolor": "#666", "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": "#e6e6e2", "grid.linewidth": 0.7,
    "axes.axisbelow": True, "legend.frameon": False, "legend.fontsize": 9,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.labelcolor": INK, "font.family": "serif",
})
PNAMES = lgcp.PARAM_NAMES
FIG = C.FIGDIR


def savefig(fig, name):
    p = os.path.join(FIG, name)
    fig.savefig(p); plt.close(fig)
    print("wrote", p)


# ---------------------------------------------------------------------------
def fig_data_examples():
    from sbilgcp.lgcp import LGCPSimulator
    sim = LGCPSimulator(G=C.GRID, nu=0.5)
    rng = np.random.default_rng(3)
    # (sigma, ell) settings, fixed beta0
    settings = [(5.5, 0.4, 0.10), (5.5, 0.4, 0.35),
                (5.5, 1.6, 0.10), (5.5, 1.6, 0.35)]
    theta = np.array(settings)
    y, Z = sim.simulate(theta, rng, return_field=True)
    y = y.numpy(); Z = Z.numpy()
    fig, axes = plt.subplots(2, 4, figsize=(9.2, 4.7))
    for i, (b0, sg, el) in enumerate(settings):
        imf = axes[0, i].imshow(Z[i], cmap="viridis", origin="lower")
        axes[0, i].set_title(fr"$\sigma={sg},\ \ell={el}$", fontsize=9)
        axes[0, i].set_xticks([]); axes[0, i].set_yticks([])
        plt.colorbar(imf, ax=axes[0, i], fraction=0.046, pad=0.02)
        imc = axes[1, i].imshow(y[i], cmap="magma", origin="lower")
        axes[1, i].set_xticks([]); axes[1, i].set_yticks([])
        plt.colorbar(imc, ax=axes[1, i], fraction=0.046, pad=0.02)
    axes[0, 0].set_ylabel("log-intensity $Z$", fontsize=9)
    axes[1, 0].set_ylabel("counts $y$", fontsize=9)
    fig.suptitle("LGCP realisations: aggregation strength ($\\sigma$) and range ($\\ell$)",
                 fontsize=10)
    savefig(fig, "fig_data_examples.pdf")


# ---------------------------------------------------------------------------
def fig_sbc(arrays):
    npe = arrays["npe_ranks"]; reg = arrays["reg_ranks"]        # [M,3]
    M = npe.shape[0]; L = C.N_POSTERIOR
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.1))
    grid, lo, hi = diag.sbc_ecdf_bands(200, L, M, alpha=0.05)
    for p in range(3):
        ax = axes[p]
        ax.fill_between(grid, lo, hi, color="#d9d9d6", alpha=0.9, lw=0,
                        label="95% band" if p == 0 else None)
        ecdf = np.arange(1, M + 1) / M
        ax.plot(np.sort(reg[:, p] / L), ecdf, color=ORANGE, lw=1.8,
                label="regressor" if p == 0 else None)
        ax.plot(np.sort(npe[:, p] / L), ecdf, color=BLUE, lw=1.8,
                label="NPE (CNN+MAF)" if p == 0 else None)
        ax.plot([0, 1], [0, 1], color=GREY, lw=1.0, ls="--")
        ax.set_title(PNAMES[p]); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("normalised rank")
        if p == 0:
            ax.set_ylabel("ECDF"); ax.legend(loc="upper left", fontsize=8)
    fig.suptitle("Simulation-based calibration: NPE ranks are uniform, "
                 "the regressor's are not", fontsize=10)
    savefig(fig, "fig_sbc.pdf")


def fig_adaptivity(cal):
    """Does the reported uncertainty predict the actual error? (sharpness).

    NPE posterior width varies across datasets and tracks the realised error;
    the regressor's homoscedastic width does not---most starkly for beta0, whose
    predictive width is essentially constant. This is the concrete failure that
    SBC flags and that marginal coverage cannot see.
    """
    import torch
    from sbilgcp.npe import NPE, CNNEmbedding, CNNRegressor

    d = np.load(C.F_TEST)
    theta = d["theta"].astype(np.float32); y = d["y"].astype(np.float32)
    npe = NPE(CNNEmbedding(embedding_dim=48), n_transforms=5, hidden=64)
    npe.load_state_dict(torch.load(C.F_NPE)); npe.eval()
    reg = CNNRegressor(); reg.load_state_dict(torch.load(C.F_REG)); reg.eval()

    yt = torch.as_tensor(y, dtype=torch.float32)
    with torch.no_grad():
        u = npe.sample(yt, n_per=C.N_POSTERIOR)
        npe_s = lgcp.from_unconstrained(u.reshape(-1, 3)).reshape(
            len(y), C.N_POSTERIOR, 3).numpy()
        dtr = np.load(C.F_TRAIN)
        ytr = torch.as_tensor(dtr["y"][:5000], dtype=torch.float32)
        thtr_u = lgcp.to_unconstrained(torch.as_tensor(dtr["theta"][:5000], dtype=torch.float32))
        resid = (reg(ytr) - thtr_u).std(0).numpy()
        pred_u = reg(yt).numpy()
    rng = np.random.default_rng(0)
    draws = pred_u[:, None, :] + resid[None, None, :] * rng.standard_normal(
        (len(y), C.N_POSTERIOR, 3))
    reg_s = lgcp.from_unconstrained(torch.as_tensor(draws.reshape(-1, 3),
             dtype=torch.float32)).reshape(len(y), C.N_POSTERIOR, 3).numpy()

    sd_npe = npe_s.std(1); err_npe = np.abs(npe_s.mean(1) - theta)
    sd_reg = reg_s.std(1); err_reg = np.abs(reg_s.mean(1) - theta)

    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.2))
    corr_npe, corr_reg = [], []
    for p in range(3):
        ax = axes[p]
        hi = np.quantile(np.concatenate([sd_npe[:, p], sd_reg[:, p]]), 0.99)
        bins = np.linspace(0, hi, 40)
        ax.hist(sd_reg[:, p], bins=bins, density=True, color=ORANGE, alpha=0.75,
                label="regressor" if p == 0 else None)
        ax.hist(sd_npe[:, p], bins=bins, density=True, histtype="step",
                color=BLUE, lw=1.8, label="NPE" if p == 0 else None)
        cn = np.corrcoef(sd_npe[:, p], err_npe[:, p])[0, 1]
        cr = np.corrcoef(sd_reg[:, p], err_reg[:, p])[0, 1]
        corr_npe.append(cn); corr_reg.append(cr)
        ax.set_title(f"{PNAMES[p]}   corr(SD, error): NPE {cn:.2f}, reg {cr:.2f}",
                     fontsize=8.5)
        ax.set_xlabel("reported posterior SD"); ax.set_yticks([])
        if p == 0:
            ax.set_ylabel("density across datasets")
            ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("Adaptive uncertainty: NPE interval widths vary with the data and "
                 "track the error; the regressor's are near-fixed", fontsize=9.5)
    savefig(fig, "fig_adaptivity.pdf")

    cv_reg_b0 = float(sd_reg[:, 0].std() / sd_reg[:, 0].mean())
    cv_npe_b0 = float(sd_npe[:, 0].std() / sd_npe[:, 0].mean())
    with open(C.ROOT + "/paper/numbers.tex", "a") as f:
        f.write(f"\\newcommand{{\\CorrNpeB}}{{{corr_npe[0]:.2f}}}\n")
        f.write(f"\\newcommand{{\\CorrNpeS}}{{{corr_npe[1]:.2f}}}\n")
        f.write(f"\\newcommand{{\\CorrNpeL}}{{{corr_npe[2]:.2f}}}\n")
        f.write(f"\\newcommand{{\\CorrRegB}}{{{corr_reg[0]:.2f}}}\n")
        f.write(f"\\newcommand{{\\CvRegB}}{{{cv_reg_b0:.2f}}}\n")
        f.write(f"\\newcommand{{\\CvNpeB}}{{{cv_npe_b0:.2f}}}\n")
    print(f"adaptivity: corr NPE {np.round(corr_npe,2)} reg {np.round(corr_reg,2)}  "
          f"cv(beta0) NPE {cv_npe_b0:.2f} reg {cv_reg_b0:.2f}")


# ---------------------------------------------------------------------------
def fig_coverage(cal):
    levels = np.array(cal["levels"])
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.1))
    methods = [("npe_cnn", "NPE (CNN+MAF)", BLUE, "o", "-"),
               ("npe_handcrafted", "NPE (summaries)", AQUA, "s", "--"),
               ("regressor", "Regressor", ORANGE, "^", ":")]
    for p in range(3):
        ax = axes[p]
        ax.plot([0, 1], [0, 1], color=GREY, lw=1.0, ls="--")
        for key, lab, col, mk, ls in methods:
            cov = np.array(cal[key]["coverage"])[:, p]
            ax.plot(levels, cov, color=col, marker=mk, ms=4, lw=1.6, ls=ls,
                    label=lab if p == 0 else None)
        ax.set_title(PNAMES[p]); ax.set_xlim(0.45, 1.0); ax.set_ylim(0.35, 1.02)
        ax.set_xlabel("nominal level")
        if p == 0:
            ax.set_ylabel("empirical coverage"); ax.legend(loc="upper left")
    fig.suptitle("Credible-interval calibration", fontsize=10)
    savefig(fig, "fig_coverage.pdf")


# ---------------------------------------------------------------------------
def fig_recovery(arrays):
    theta = arrays["theta_te"]; mean = arrays["npe_post_mean"]; std = arrays["npe_post_std"]
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.2))
    n_show = 400
    idx = np.random.default_rng(0).choice(theta.shape[0], n_show, replace=False)
    for p in range(3):
        ax = axes[p]
        lo, hi = theta[:, p].min(), theta[:, p].max()
        ax.errorbar(theta[idx, p], mean[idx, p],
                    yerr=1.645 * std[idx, p], fmt="o", ms=2.5, lw=0.5,
                    color=BLUE, ecolor="#b7d3f6", alpha=0.7)
        ax.plot([lo, hi], [lo, hi], color=GREY, lw=1.2, ls="--")
        r2 = 1 - ((mean[:, p] - theta[:, p]) ** 2).sum() / (
            (theta[:, p] - theta[:, p].mean()) ** 2).sum()
        ax.set_title(f"{PNAMES[p]}   $R^2$={r2:.2f}")
        ax.set_xlabel("true");
        if p == 0: ax.set_ylabel("posterior mean (90% CI)")
    fig.suptitle("Posterior recovery on held-out simulations (NPE)", fontsize=10)
    savefig(fig, "fig_recovery.pdf")


# ---------------------------------------------------------------------------
def fig_npe_vs_mcmc(cmp):
    mean_ref = np.array(cmp["mean_ref"]); mean_npe = np.array(cmp["mean_npe"])
    std_ref = np.array(cmp["std_ref"]); std_npe = np.array(cmp["std_npe"])
    ex = cmp["examples"]; key = list(ex.keys())[0]
    ref = np.array(ex[key]["ref"]); npe = np.array(ex[key]["npe"]); true = ex[key]["true"]

    fig = plt.figure(figsize=(9.2, 6.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.05, 1])

    # top row: overlaid marginal posteriors for one example dataset
    for p in range(3):
        ax = fig.add_subplot(gs[0, p])
        ax.hist(ref[:, p], bins=30, density=True, color="#c9c9c4", alpha=0.9,
                label="NUTS" if p == 0 else None)
        ax.hist(npe[:, p], bins=30, density=True, histtype="step", color=BLUE,
                lw=1.8, label="NPE" if p == 0 else None)
        ax.axvline(true[p], color=RED, lw=1.3, ls="--",
                   label="truth" if p == 0 else None)
        ax.set_title(PNAMES[p]); ax.set_yticks([])
        if p == 0:
            ax.set_ylabel("density"); ax.legend(loc="upper right", fontsize=8)
    # bottom: agreement scatters
    labels = ["posterior mean", "posterior sd"]
    for j, (ref_v, npe_v, lab) in enumerate(
        [(mean_ref, mean_npe, "mean"), (std_ref, std_npe, "sd")]
    ):
        ax = fig.add_subplot(gs[1, j])
        cols = [BLUE, AQUA, ORANGE]
        for p in range(3):
            ax.scatter(ref_v[:, p], npe_v[:, p], s=16, color=cols[p],
                       alpha=0.8, label=PNAMES[p], edgecolor="white", lw=0.3)
        lo = min(ref_v.min(), npe_v.min()); hi = max(ref_v.max(), npe_v.max())
        ax.plot([lo, hi], [lo, hi], color=GREY, ls="--", lw=1.1)
        ax.set_xlabel(f"NUTS {lab}"); ax.set_ylabel(f"NPE {lab}")
        if j == 0: ax.legend(fontsize=8, loc="upper left")
    # C2ST histogram
    ax = fig.add_subplot(gs[1, 2])
    c2 = np.array(cmp["c2st_all"])
    ax.hist(c2, bins=np.linspace(0.4, 0.8, 13), color=VIOLET, alpha=0.85)
    ax.axvline(0.5, color=GREY, ls="--", lw=1.2)
    ax.axvline(c2.mean(), color=RED, lw=1.4, label=f"mean {c2.mean():.2f}")
    ax.set_xlabel("C2ST accuracy (NPE vs NUTS)"); ax.set_ylabel("count")
    ax.legend(fontsize=8)
    fig.suptitle("NPE reproduces the gold-standard NUTS posterior", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    savefig(fig, "fig_npe_vs_mcmc.pdf")


# ---------------------------------------------------------------------------
def fig_amortization(cmp, cal):
    npe_ms = cmp["npe_amortized_per_dataset_ms"]
    mcmc_s = cmp["mcmc_wall_median"]
    n = np.arange(0, 2001)
    npe_total = np.full_like(n, 0.0, dtype=float)
    # NPE: one-time training cost + amortized inference
    train_s = json.load(open(C.F_REPORTS))["npe_cnn"]["wall"]
    npe_total = train_s + n * (npe_ms / 1e3)
    mcmc_total = n * mcmc_s
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    ax = axes[0]
    ax.plot(n, mcmc_total / 60, color=ORANGE, lw=2, label="NUTS (per-dataset)")
    ax.plot(n, npe_total / 60, color=BLUE, lw=2, label="NPE (train once + amortize)")
    # crossover
    cross = train_s / (mcmc_s - npe_ms / 1e3)
    ax.axvline(cross, color=GREY, ls="--", lw=1.1)
    ax.text(cross + 40, ax.get_ylim()[1] * 0.5, f"break-even\n≈{cross:.0f} datasets",
            fontsize=8, color=INK2)
    ax.set_xlabel("number of datasets analysed"); ax.set_ylabel("wall-clock (min)")
    ax.set_title("Cumulative inference cost"); ax.legend(loc="upper left")

    ax = axes[1]
    labels = ["NUTS\n(1 dataset)", "NPE\n(1 dataset)"]
    vals = [mcmc_s * 1e3, npe_ms]
    bars = ax.bar(labels, vals, color=[ORANGE, BLUE], width=0.6)
    ax.set_yscale("log"); ax.set_ylabel("time per dataset (ms, log)")
    speed = cmp.get("speedup", mcmc_s / (npe_ms / 1e3))
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.15, f"{v:.0f} ms" if v < 1000
                else f"{v/1000:.1f} s", ha="center", fontsize=9)
    ax.set_title(f"Per-dataset speed-up ≈ {speed:,.0f}×")
    fig.suptitle("Amortization: SBI pays a one-time training cost, then near-free inference",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    savefig(fig, "fig_amortization.pdf")


# ---------------------------------------------------------------------------
def fig_misspec(cal, arrays):
    levels = np.array(cal["levels"])
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.1))
    specs = [("well-specified\n(Matern-1/2)", cal["npe_cnn"], BLUE, "-", "o"),
             ("Matern-3/2", cal["misspec"]["1.5"], YELLOW, "--", "s"),
             ("Matern-5/2", cal["misspec"]["2.5"], RED, ":", "^")]
    for p in range(3):
        ax = axes[p]
        ax.plot([0, 1], [0, 1], color=GREY, lw=1.0, ls="--")
        for lab, res, col, ls, mk in specs:
            cov = np.array(res["coverage"])[:, p]
            ax.plot(levels, cov, color=col, ls=ls, marker=mk, ms=4, lw=1.6,
                    label=lab if p == 0 else None)
        ax.set_title(PNAMES[p]); ax.set_xlim(0.45, 1.0); ax.set_ylim(0.2, 1.02)
        ax.set_xlabel("nominal level")
        if p == 0:
            ax.set_ylabel("empirical coverage"); ax.legend(loc="lower right", fontsize=8)
    fig.suptitle("Robustness to kernel misspecification", fontsize=10)
    savefig(fig, "fig_misspec.pdf")


# ---------------------------------------------------------------------------
def fig_training(reports):
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for key, lab, col in [("npe_cnn", "NPE (CNN+MAF)", BLUE),
                          ("npe_handcrafted", "NPE (summaries)", AQUA)]:
        ax.plot(reports[key]["val_curve"], color=col, lw=1.6, label=lab)
    ax.set_xlabel("epoch"); ax.set_ylabel("validation NLL")
    ax.set_title("NPE training curves"); ax.legend()
    savefig(fig, "fig_training.pdf")


# ---------------------------------------------------------------------------
def write_tables(cal, cmp, reports):
    pm = PNAMES
    def row(name, res):
        rmse = res["rmse"]
        cov90 = np.array(res["coverage"])[cal["levels"].index(0.9)]
        pvals = res["sbc_pvalues"]
        sbc = f"{min(pvals):.3f}"
        cells = [f"{rmse[p]:.3f}" for p in range(3)]
        cells += [f"{cov90[p]:.2f}" for p in range(3)]
        return f"{name} & " + " & ".join(cells) + f" & {sbc}" + r" \\"

    with open(os.path.join(FIG, "..", "table_methods.tex"), "w") as f:
        f.write("% auto-generated\n")
        f.write(r"\begin{tabular}{l ccc ccc c}" + "\n\\toprule\n")
        f.write(r" & \multicolumn{3}{c}{RMSE $\downarrow$} & "
                r"\multicolumn{3}{c}{90\% coverage} & SBC \\" + "\n")
        f.write(r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}" + "\n")
        f.write(r"Method & $\beta_0$ & $\sigma$ & $\ell$ & "
                r"$\beta_0$ & $\sigma$ & $\ell$ & $\min p$ \\" + "\n\\midrule\n")
        f.write(row("NPE (CNN+MAF)", cal["npe_cnn"]) + "\n")
        f.write(row("NPE (hand-crafted)", cal["npe_handcrafted"]) + "\n")
        f.write(row("CNN regressor", cal["regressor"]) + "\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    # MCMC comparison table
    with open(os.path.join(FIG, "..", "table_mcmc.tex"), "w") as f:
        f.write("% auto-generated\n")
        f.write(r"\begin{tabular}{lcccc}" + "\n\\toprule\n")
        f.write(r"Parameter & C2ST & 1-Wass. (/prior sd) & mean corr & sd ratio \\"
                + "\n\\midrule\n")
        wn = cmp["wasserstein_norm_mean"]; mc = cmp["post_mean_corr"]
        sr = cmp["post_std_ratio_mean"]
        for p in range(3):
            f.write(f"{pm[p]} & -- & {wn[p]:.3f} & {mc[p]:.3f} & {sr[p]:.2f} \\\\\n")
        f.write(r"\midrule" + "\n")
        f.write(f"overall (3-D) & {cmp['c2st_mean']:.3f} & "
                f"{np.mean(wn):.3f} & -- & -- \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print("wrote LaTeX tables")


def main():
    cal = json.load(open(C.F_CALIB))
    reports = json.load(open(C.F_REPORTS))
    arrays = dict(np.load(C.RESULTS + "/eval_arrays.npz"))
    fig_data_examples()
    fig_sbc(arrays)
    fig_coverage(cal)
    fig_adaptivity(cal)
    fig_recovery(arrays)
    fig_misspec(cal, arrays)
    fig_training(reports)
    if os.path.exists(C.F_MCMC_CMP):
        cmp = json.load(open(C.F_MCMC_CMP))
        fig_npe_vs_mcmc(cmp)
        fig_amortization(cmp, cal)
        write_tables(cal, cmp, reports)
    else:
        print("MCMC comparison not found -- skipping those figures/tables")


if __name__ == "__main__":
    main()
