"""Figures, table and macros for the preferential-sampling section."""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C

BLUE = "#2a78d6"; ORANGE = "#eb6834"; GREY = "#7a7a76"; VIOLET = "#4a3aa7"; RED = "#e34948"
plt.rcParams.update({
    "figure.dpi": 140, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
    "axes.edgecolor": "#666", "axes.linewidth": 0.8, "axes.grid": True,
    "grid.color": "#e6e6e2", "grid.linewidth": 0.7, "axes.axisbelow": True,
    "legend.frameon": False, "legend.fontsize": 9, "font.family": "serif",
})
FIG = C.FIGDIR
PN = [r"$\beta_0$", r"$\sigma$", r"$\ell$"]
LEVELS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]


def save(fig, name):
    p = os.path.join(FIG, name); fig.savefig(p); plt.close(fig); print("wrote", p)


def fig_bias(res):
    strat = res["strat"]
    gc = strat["gamma_centers"]
    bp = np.array(strat["bias_ps"]); bn = np.array(strat["bias_naive"])
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.1))
    for p in range(3):
        ax = axes[p]
        ax.axhline(0, color=GREY, lw=1.0, ls="--")
        ax.plot(gc, bn[:, p], color=ORANGE, marker="^", ms=5, lw=1.6,
                label="design-ignorant" if p == 0 else None)
        ax.plot(gc, bp[:, p], color=BLUE, marker="o", ms=5, lw=1.6,
                label="design-aware (ours)" if p == 0 else None)
        ax.set_title(PN[p]); ax.set_xlabel(r"preferentiality $\gamma$ (true)")
        if p == 0:
            ax.set_ylabel("bias of posterior mean"); ax.legend(loc="upper left", fontsize=8)
    fig.suptitle("Ignoring an informative design biases inference; the design-aware "
                 "NPE removes it", fontsize=9.5)
    save(fig, "fig_pref_bias.pdf")


def fig_gamma(res, arr, mc):
    gt = arr["gamma_true"]; gm = arr["gamma_mean"]
    # posterior std for gamma from arrays? we saved post_std_ps [N,4]
    gsd = arr["post_std_ps"][:, 3]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3))
    ax = axes[0]
    idx = np.random.default_rng(0).choice(len(gt), min(400, len(gt)), replace=False)
    ax.errorbar(gt[idx], gm[idx], yerr=1.645 * gsd[idx], fmt="o", ms=2.5, lw=0.4,
                color=VIOLET, ecolor="#cfc8ee", alpha=0.7)
    ax.plot([0, 3], [0, 3], color=GREY, ls="--", lw=1.1)
    ax.set_xlabel(r"true $\gamma$"); ax.set_ylabel(r"posterior mean $\gamma$ (90% CI)")
    ax.set_title(fr"Preferentiality is identifiable ($R^2$={res['gamma_r2']:.2f})", fontsize=9)

    ax = axes[1]
    if mc is not None and mc.get("examples"):
        key = list(mc["examples"].keys())[0]
        ex = mc["examples"][key]
        ref = np.array(ex["ref"]); npe = np.array(ex["npe"]); true = ex["true"]
        ax.hist(ref[:, 3], bins=25, density=True, color="#c9c9c4", alpha=0.9, label="NUTS")
        ax.hist(npe[:, 3], bins=25, density=True, histtype="step", color=BLUE, lw=1.8, label="NPE")
        ax.axvline(true[3], color=RED, lw=1.3, ls="--", label="truth")
        ax.set_xlabel(r"$\gamma$"); ax.set_yticks([])
        ax.set_title(f"vs gold-standard NUTS (C2ST {mc['c2st_mean']:.2f})", fontsize=9)
        ax.legend(fontsize=8)
    else:
        ax.hist(gm - gt, bins=30, color=VIOLET, alpha=0.85)
        ax.axvline(0, color=GREY, ls="--")
        ax.set_xlabel(r"$\gamma$ posterior-mean error"); ax.set_title("error distribution", fontsize=9)
    fig.suptitle("Design-aware NPE recovers the preferentiality parameter", fontsize=9.5)
    save(fig, "fig_pref_gamma.pdf")


def write_table(res):
    cov_ps = np.array(res["cov_ps"])[LEVELS.index(0.9)]
    cov_nv = np.array(res["cov_naive"])[LEVELS.index(0.9)]
    bp = res["bias_ps"]; bn = res["bias_naive"]
    pp = res["sbc_ps"]; pn = res["sbc_naive"]
    with open(os.path.join(FIG, "..", "table_pref.tex"), "w") as f:
        f.write("% auto-generated\n\\begin{tabular}{l ccc ccc c}\n\\toprule\n")
        f.write(r" & \multicolumn{3}{c}{bias} & \multicolumn{3}{c}{90\% coverage} & SBC \\" + "\n")
        f.write(r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}" + "\n")
        f.write(r"Analysis & $\beta_0$ & $\sigma$ & $\ell$ & $\beta_0$ & $\sigma$ & $\ell$ & $\min p$ \\" + "\n\\midrule\n")
        f.write(f"Design-ignorant & {bn[0]:+.2f} & {bn[1]:+.2f} & {bn[2]:+.2f} & "
                f"{cov_nv[0]:.2f} & {cov_nv[1]:.2f} & {cov_nv[2]:.2f} & {min(pn):.3f} \\\\\n")
        f.write(f"Design-aware (ours) & {bp[0]:+.2f} & {bp[1]:+.2f} & {bp[2]:+.2f} & "
                f"{cov_ps[0]:.2f} & {cov_ps[1]:.2f} & {cov_ps[2]:.2f} & {min(pp):.3f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print("wrote table_pref.tex")


def write_macros(res, mc):
    bn = res["bias_naive"]; bp = res["bias_ps"]
    cov_nv = np.array(res["cov_naive"])[LEVELS.index(0.9)]
    cov_ps = np.array(res["cov_ps"])[LEVELS.index(0.9)]
    with open(C.ROOT + "/paper/numbers.tex", "a") as f:
        f.write(f"\\newcommand{{\\PrefBiasNaiveBeta}}{{{bn[0]:+.2f}}}\n")
        f.write(f"\\newcommand{{\\PrefBiasNaiveSig}}{{{bn[1]:+.2f}}}\n")
        f.write(f"\\newcommand{{\\PrefBiasPsBeta}}{{{bp[0]:+.2f}}}\n")
        f.write(f"\\newcommand{{\\PrefBiasPsSig}}{{{bp[1]:+.2f}}}\n")
        f.write(f"\\newcommand{{\\PrefCovNaiveMin}}{{{cov_nv.min():.2f}}}\n")
        f.write(f"\\newcommand{{\\PrefCovPsMin}}{{{cov_ps.min():.2f}}}\n")
        f.write(f"\\newcommand{{\\PrefSbcNaive}}{{{min(res['sbc_naive']):.3f}}}\n")
        f.write(f"\\newcommand{{\\PrefSbcPs}}{{{min(res['sbc_ps']):.2f}}}\n")
        f.write(f"\\newcommand{{\\GammaRtwo}}{{{res['gamma_r2']:.2f}}}\n")
        if mc is not None and mc.get("c2st_mean") is not None:
            f.write(f"\\newcommand{{\\PrefCtst}}{{{mc['c2st_mean']:.3f}}}\n")
            f.write(f"\\newcommand{{\\PrefNCompared}}{{{mc['n_compared']}}}\n")
    print("wrote preferential macros")


def main():
    res = json.load(open(C.RESULTS + "/pref_results.json"))
    arr = dict(np.load(C.RESULTS + "/pref_arrays.npz"))
    mc = None
    if os.path.exists(C.RESULTS + "/pref_mcmc.json"):
        mc = json.load(open(C.RESULTS + "/pref_mcmc.json"))
    fig_bias(res)
    fig_gamma(res, arr, mc)
    write_table(res)
    write_macros(res, mc)


if __name__ == "__main__":
    main()
