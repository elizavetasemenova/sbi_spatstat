"""Figures and macros for the change-of-support / aggregation section."""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C

BLUE = "#2a78d6"; ORANGE = "#eb6834"; GREY = "#7a7a76"; AQUA = "#1baf7a"; RED = "#e34948"; VIOLET = "#4a3aa7"
plt.rcParams.update({
    "figure.dpi": 140, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
    "axes.edgecolor": "#666", "axes.linewidth": 0.8, "legend.frameon": False,
    "legend.fontsize": 9, "font.family": "serif",
})
FIG = C.FIGDIR
PN = [r"$\beta_0$", r"$\sigma$", r"$\ell$"]
LEVELS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]


def save(fig, name):
    p = os.path.join(FIG, name); fig.savefig(p); plt.close(fig); print("wrote", p)


def fig_illustration(arr):
    """Show one fine pattern and its coarse aggregation."""
    from sbilgcp.lgcp import LGCPSimulator, sample_prior
    from sbilgcp.aggregation import aggregate
    import torch
    rng = np.random.default_rng(2); sim = LGCPSimulator(G=C.GRID)
    th = sample_prior(2, rng); y = sim.simulate(th, rng)
    r = aggregate(y, 4)
    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.7))
    im0 = axes[0].imshow(y[0], cmap="magma", origin="lower")
    axes[0].set_title("fine counts (16$\\times$16, latent)", fontsize=9)
    im1 = axes[1].imshow(r[0], cmap="magma", origin="lower")
    axes[1].set_title("observed region totals (4$\\times$4)", fontsize=9)
    for ax, im in [(axes[0], im0), (axes[1], im1)]:
        ax.set_xticks([]); ax.set_yticks([]); plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle("Change of support: only coarse totals are observed", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, "fig_agg_illustration.pdf")


def fig_results(res, arr, mc):
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.1))
    lv = np.array(LEVELS)
    # (a) coverage (calibrated despite aggregation)
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color=GREY, ls="--", lw=1.0)
    cols = [BLUE, AQUA, ORANGE]
    for p in range(3):
        ax.plot(lv, np.array(res["coverage"])[:, p], color=cols[p], marker="o", ms=3.5, lw=1.5, label=PN[p])
    ax.set_xlim(0.45, 1); ax.set_ylim(0.4, 1.0); ax.set_xlabel("nominal level")
    ax.set_ylabel("empirical coverage"); ax.set_title("Calibrated from coarse data", fontsize=9)
    ax.legend(fontsize=8, loc="upper left")
    # (b) information loss: posterior width ratio agg/full
    ax = axes[1]
    wr = res["width_ratio"]
    bars = ax.bar(PN, wr, color=[BLUE, AQUA, ORANGE], width=0.6)
    ax.axhline(1.0, color=GREY, ls="--", lw=1.0)
    for b, v in zip(bars, wr):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.1f}$\\times$", ha="center", fontsize=9)
    ax.set_ylabel("posterior width vs fully observed"); ax.set_title("Cost of aggregation", fontsize=9)
    ax.set_ylim(0, max(wr) * 1.25)
    # (c) recovery R2 agg vs full
    ax = axes[2]
    x = np.arange(3); w = 0.36
    ax.bar(x - w/2, res["r2_full"], w, color="#b7d3f6", label="fully observed")
    ax.bar(x + w/2, res["r2"], w, color=BLUE, label="aggregated")
    ax.set_xticks(x); ax.set_xticklabels(PN); ax.set_ylabel("$R^2$"); ax.set_ylim(0, 1)
    ax.set_title("Recovery", fontsize=9); ax.legend(fontsize=8)
    fig.suptitle("Amortized inference from aggregated counts: calibrated, with honest information loss",
                 fontsize=9.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, "fig_agg_results.pdf")


def write_macros(res, mc):
    i90 = LEVELS.index(0.9)
    cov90 = np.array(res["coverage"])[i90]
    with open(C.ROOT + "/paper/numbers.tex", "a") as f:
        def c(n, v): f.write(f"\\newcommand{{\\{n}}}{{{v}}}\n")
        c("AggCovB", f"{cov90[0]:.2f}"); c("AggCovS", f"{cov90[1]:.2f}"); c("AggCovL", f"{cov90[2]:.2f}")
        c("AggSbcMin", f"{min(res['sbc_pvalues']):.2f}")
        c("AggRtwoB", f"{res['r2'][0]:.2f}"); c("AggRtwoBfull", f"{res['r2_full'][0]:.2f}")
        c("AggRtwoLfull", f"{res['r2_full'][2]:.2f}"); c("AggRtwoL", f"{res['r2'][2]:.2f}")
        c("AggWidthB", f"{res['width_ratio'][0]:.1f}")
        c("AggWidthL", f"{res['width_ratio'][2]:.1f}")
        if mc is not None and mc.get("c2st_mean") is not None:
            c("AggCtst", f"{mc['c2st_mean']:.2f}")
            c("AggN", f"{mc['n_compared']}")
            c("AggMcmcSec", f"{mc['mcmc_wall_median']:.0f}")
    print("wrote aggregation macros")


def main():
    res = json.load(open(C.RESULTS + "/agg_results.json"))
    arr = dict(np.load(C.RESULTS + "/agg_arrays.npz"))
    mc = None
    if os.path.exists(C.RESULTS + "/agg_mcmc.json"):
        mc = json.load(open(C.RESULTS + "/agg_mcmc.json"))
    fig_illustration(arr)
    fig_results(res, arr, mc)
    write_macros(res, mc)


if __name__ == "__main__":
    main()
