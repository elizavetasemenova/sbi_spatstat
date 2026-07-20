"""Figure and macros for the real-data (crimes) case study."""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C

BLUE = "#2a78d6"; GREY = "#7a7a76"; RED = "#e34948"
plt.rcParams.update({
    "figure.dpi": 140, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
    "axes.edgecolor": "#666", "axes.linewidth": 0.8, "legend.frameon": False,
    "legend.fontsize": 8, "font.family": "serif",
})
FIG = C.FIGDIR
PN = [r"$\beta_0$", r"$\sigma$", r"$\ell$"]


def save(fig, name):
    p = os.path.join(FIG, name); fig.savefig(p); plt.close(fig); print("wrote", p)


def main():
    res = json.load(open(C.RESULTS + "/real_results.json"))
    arr = dict(np.load(C.RESULTS + "/real_arrays.npz"))
    y = arr["y"]; pts = arr["pts"]
    mmap = arr["field_mean_npe"]; smap = arr["field_sd_npe"]
    th_npe = arr["theta_npe"]; th_nuts = arr["theta_nuts"]
    G = y.shape[0]

    fig = plt.figure(figsize=(9.2, 5.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1])

    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(pts[:, 0] * G, pts[:, 1] * G, s=7, color=RED, alpha=0.6, edgecolor="none")
    ax.set_xlim(0, G); ax.set_ylim(0, G); ax.set_aspect("equal")
    ax.set_title(f"crime locations (n={res['n_points']})", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(mmap.T, origin="lower", cmap="viridis")
    ax.set_title("amortized posterior mean\nlog-intensity", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([]); plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    ax = fig.add_subplot(gs[0, 2])
    im = ax.imshow(smap.T, origin="lower", cmap="cividis")
    ax.set_title("posterior SD (uncertainty)", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([]); plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    for p in range(3):
        ax = fig.add_subplot(gs[1, p])
        lo = min(th_npe[:, p].min(), th_nuts[:, p].min())
        hi = max(th_npe[:, p].max(), th_nuts[:, p].max())
        bins = np.linspace(lo, hi, 30)
        ax.hist(th_nuts[:, p], bins=bins, density=True, color="#c9c9c4", alpha=0.9,
                label="NUTS" if p == 0 else None)
        ax.hist(th_npe[:, p], bins=bins, density=True, histtype="step", color=BLUE, lw=1.8,
                label="NPE" if p == 0 else None)
        ax.set_title(PN[p], fontsize=10); ax.set_yticks([])
        if p == 0:
            ax.set_ylabel("density"); ax.legend(fontsize=8, loc="upper right")
    fig.suptitle(f"Real crime pattern: amortized inference (trained on simulations) matches NUTS "
                 f"in {res['t_npe_param_ms']+res['t_npe_field_ms']:.0f} ms vs {res['t_nuts_s']:.0f} s",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save(fig, "fig_realdata.pdf")

    with open(C.ROOT + "/paper/numbers.tex", "a") as f:
        def c(n, v): f.write(f"\\newcommand{{\\{n}}}{{{v}}}\n")
        c("RealN", res["n_points"])
        c("RealCtst", f"{res['param_c2st']:.2f}")
        c("RealFieldCorr", f"{res['field_mean_corr']:.3f}")
        c("RealFieldSdCorr", f"{res['field_sd_corr']:.2f}")
        c("RealNpeMs", f"{res['t_npe_param_ms']+res['t_npe_field_ms']:.0f}")
        c("RealNutsSec", f"{res['t_nuts_s']:.0f}")
        c("RealSpeedup", f"{res['speedup']:,.0f}")
        c("RealBetaN", f"{res['theta_npe_mean'][0]:.2f}"); c("RealBetaM", f"{res['theta_nuts_mean'][0]:.2f}")
        c("RealSigN", f"{res['theta_npe_mean'][1]:.2f}"); c("RealSigM", f"{res['theta_nuts_mean'][1]:.2f}")
        c("RealEllN", f"{res['theta_npe_mean'][2]:.2f}"); c("RealEllM", f"{res['theta_nuts_mean'][2]:.2f}")
    print("wrote real-data macros")


if __name__ == "__main__":
    main()
