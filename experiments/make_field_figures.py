"""Figures and macros for the intensity-surface reconstruction section."""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C

BLUE = "#2a78d6"; ORANGE = "#eb6834"; GREY = "#7a7a76"; AQUA = "#1baf7a"; RED = "#e34948"
plt.rcParams.update({
    "figure.dpi": 140, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
    "axes.edgecolor": "#666", "axes.linewidth": 0.8, "legend.frameon": False,
    "legend.fontsize": 9, "font.family": "serif",
})
FIG = C.FIGDIR
G = C.GRID
LEVELS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]


def save(fig, name):
    p = os.path.join(FIG, name); fig.savefig(p); plt.close(fig); print("wrote", p)


def fig_maps(arr):
    y = arr["y"]; Zt = arr["Ztrue"]; pm = arr["pmean"].reshape(-1, G, G); ps = arr["pstd"].reshape(-1, G, G)
    ndat = 3
    fig, axes = plt.subplots(ndat, 4, figsize=(8.6, 2.15 * ndat))
    col = ["observed counts", "true log-intensity", "posterior mean", "posterior SD"]
    for i in range(ndat):
        vmin = min(Zt[i].min(), pm[i].min()); vmax = max(Zt[i].max(), pm[i].max())
        im0 = axes[i, 0].imshow(y[i], cmap="magma", origin="lower")
        im1 = axes[i, 1].imshow(Zt[i], cmap="viridis", origin="lower", vmin=vmin, vmax=vmax)
        im2 = axes[i, 2].imshow(pm[i], cmap="viridis", origin="lower", vmin=vmin, vmax=vmax)
        im3 = axes[i, 3].imshow(ps[i], cmap="cividis", origin="lower")
        for j, im in enumerate([im0, im1, im2, im3]):
            axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
            plt.colorbar(im, ax=axes[i, j], fraction=0.046, pad=0.02)
            if i == 0:
                axes[i, j].set_title(col[j], fontsize=9)
    fig.suptitle("Amortized reconstruction of the log-intensity surface (one forward pass)",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save(fig, "fig_field_maps.pdf")


def fig_calib(res, arr):
    lv = np.array(LEVELS)
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.1))
    # (a) pointwise coverage
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color=GREY, ls="--", lw=1.0)
    ax.plot(lv, res["pointwise_cov"], color=BLUE, marker="o", ms=4, lw=1.6)
    ax.set_xlim(0.45, 1); ax.set_ylim(0.45, 1.0)
    ax.set_xlabel("nominal level"); ax.set_ylabel("empirical coverage")
    ax.set_title("Pointwise (per-cell)", fontsize=9)
    # (b) abundance functional coverage: full vs diagonal
    ax = axes[1]
    ax.plot([0, 1], [0, 1], color=GREY, ls="--", lw=1.0)
    ax.plot(lv, res["abundance_cov_diag"], color=ORANGE, marker="^", ms=4, lw=1.6,
            label="diagonal only")
    ax.plot(lv, res["abundance_cov_full"], color=BLUE, marker="o", ms=4, lw=1.6,
            label="low-rank (ours)")
    ax.set_xlim(0.45, 1); ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("nominal level"); ax.set_title("Total abundance", fontsize=9)
    ax.legend(loc="lower right", fontsize=8)
    # (c) pixel z-score histogram vs N(0,1)
    ax = axes[2]
    z = arr["zpix"]
    ax.hist(z, bins=40, density=True, color="#c9c9c4", alpha=0.9)
    xs = np.linspace(-4, 4, 200)
    ax.plot(xs, np.exp(-xs ** 2 / 2) / np.sqrt(2 * np.pi), color=RED, lw=1.5)
    ax.set_xlim(-4, 4); ax.set_yticks([])
    ax.set_xlabel("pixel z-score"); ax.set_title(f"Calibration (sd {z.std():.2f})", fontsize=9)
    fig.suptitle("The field posterior is calibrated pointwise and for aggregate functionals",
                 fontsize=9.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, "fig_field_calib.pdf")


def fig_mcmc(mc):
    ex = list(mc["examples"].values())[0]
    rm = np.array(ex["ref_mean"]).reshape(G, G); nm = np.array(ex["npe_mean"]).reshape(G, G)
    rs = np.array(ex["ref_sd"]).reshape(G, G); ns = np.array(ex["npe_sd"]).reshape(G, G)
    fig, axes = plt.subplots(1, 4, figsize=(9.0, 2.5))
    vmin = min(rm.min(), nm.min()); vmax = max(rm.max(), nm.max())
    smax = max(rs.max(), ns.max())
    for ax, arr2, ttl, cm, vlim in [
        (axes[0], rm, "NUTS mean", "viridis", (vmin, vmax)),
        (axes[1], nm, "NPE mean", "viridis", (vmin, vmax)),
        (axes[2], rs, "NUTS SD", "cividis", (0, smax)),
        (axes[3], ns, "NPE SD", "cividis", (0, smax))]:
        im = ax.imshow(arr2, cmap=cm, origin="lower", vmin=vlim[0], vmax=vlim[1])
        ax.set_title(ttl, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(f"Field posterior vs gold-standard NUTS "
                 f"(mean corr {mc['postmean_corr']:.3f}, SD corr {mc['poststd_corr']:.2f})",
                 fontsize=9.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, "fig_field_mcmc.pdf")


def write_macros(res, mc):
    i90 = LEVELS.index(0.9)
    with open(C.ROOT + "/paper/numbers.tex", "a") as f:
        def c(n, v): f.write(f"\\newcommand{{\\{n}}}{{{v}}}\n")
        c("FieldRMSE", f"{res['rmse']:.2f}")
        c("FieldCorr", f"{res['field_corr']:.3f}")
        c("FieldPointCov", f"{res['pointwise_cov'][i90]:.2f}")
        c("FieldAbFull", f"{res['abundance_cov_full'][i90]:.2f}")
        c("FieldAbDiag", f"{res['abundance_cov_diag'][i90]:.2f}")
        c("FieldRegFull", f"{res['region_cov90_full']:.2f}")
        c("FieldRegDiag", f"{res['region_cov90_diag']:.2f}")
        c("FieldZstd", f"{res['zpix_std']:.2f}")
        c("FieldTrainMin", f"{res['train_min']:.1f}")
        if mc is not None and mc.get("postmean_corr") is not None:
            c("FieldMcmcMeanCorr", f"{mc['postmean_corr']:.3f}")
            c("FieldMcmcSdCorr", f"{mc['poststd_corr']:.2f}")
            c("FieldMcmcCovRef", f"{mc['cov_ref']:.2f}")
            c("FieldMcmcCovNpe", f"{mc['cov_npe']:.2f}")
            c("FieldMcmcN", f"{mc['n']}")
    print("wrote field macros")


def main():
    res = json.load(open(C.RESULTS + "/field_results.json"))
    arr = dict(np.load(C.RESULTS + "/field_arrays.npz"))
    mc = None
    if os.path.exists(C.RESULTS + "/field_mcmc.json"):
        mc = json.load(open(C.RESULTS + "/field_mcmc.json"))
    fig_maps(arr)
    fig_calib(res, arr)
    if mc is not None and mc.get("examples"):
        fig_mcmc(mc)
    write_macros(res, mc)


if __name__ == "__main__":
    main()
