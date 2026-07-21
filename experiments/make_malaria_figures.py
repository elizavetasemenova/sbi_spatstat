"""Figure and macros for the malaria disease-mapping case study."""

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
    res = json.load(open(C.RESULTS + "/malaria_results.json"))
    arr = dict(np.load(C.RESULTS + "/malaria_arrays.npz"))
    E = arr["E"]; risk = arr["risk_map"]; risk_sd = arr["risk_sd"]
    th_npe = arr["theta_npe"]; th_nuts = arr["theta_nuts"]
    lon = arr["lon"]; lat = arr["lat"]; pos = arr["pos"]; exm = arr["exm"]
    prev = pos / exm
    G = E.shape[0]

    fig = plt.figure(figsize=(9.2, 5.7))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1])

    # observed prevalence at survey sites
    ax = fig.add_subplot(gs[0, 0])
    sc = ax.scatter(lon, lat, c=prev, s=18, cmap="magma", vmin=0, vmax=1, edgecolor="k", lw=0.2)
    ax.set_title("observed prevalence\n(survey clusters)", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("auto")
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)

    # posterior mean risk map (masked to interior)
    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(risk.T, origin="lower", cmap="magma", vmin=0, vmax=min(1.0, risk.max()))
    ax.set_title("posterior mean malaria risk\n$\\exp(Z)$", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([]); plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    ax = fig.add_subplot(gs[0, 2])
    im = ax.imshow(risk_sd.T, origin="lower", cmap="cividis")
    ax.set_title("risk posterior SD\n(uncertainty)", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([]); plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    for p in range(3):
        ax = fig.add_subplot(gs[1, p])
        lo = min(th_npe[:, p].min(), th_nuts[:, p].min()); hi = max(th_npe[:, p].max(), th_nuts[:, p].max())
        bins = np.linspace(lo, hi, 30)
        ax.hist(th_nuts[:, p], bins=bins, density=True, color="#c9c9c4", alpha=0.9,
                label="NUTS" if p == 0 else None)
        ax.hist(th_npe[:, p], bins=bins, density=True, histtype="step", color=BLUE, lw=1.8,
                label="NPE" if p == 0 else None)
        ax.set_title(PN[p], fontsize=10); ax.set_yticks([])
        if p == 0:
            ax.set_ylabel("density"); ax.legend(fontsize=8, loc="upper left")
    fig.suptitle(f"Malaria risk mapping (Burkina Faso, {res['n_sites']} clusters): "
                 f"amortized inference matches NUTS in {res['t_npe_ms']:.0f} ms vs {res['t_nuts_s']:.0f} s",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save(fig, "fig_malaria.pdf")

    with open(C.ROOT + "/paper/numbers.tex", "a") as f:
        def c(n, v): f.write(f"\\newcommand{{\\{n}}}{{{v}}}\n")
        c("MalPos", f"{res['total_positives']:,}"); c("MalExam", f"{res['total_examined']:,}")
        c("MalSites", res["n_sites"]); c("MalCells", res["surveyed_cells"])
        c("MalPrev", f"{100*res['empirical_prev']:.0f}")
        c("MalCtst", f"{res['param_c2st']:.2f}")
        c("MalSbc", f"{min(res['sbc_p']):.2f}")
        c("MalNpeMs", f"{res['t_npe_ms']:.0f}"); c("MalNutsSec", f"{res['t_nuts_s']:.0f}")
        c("MalSpeedup", f"{res['speedup']:,.0f}")
        c("MalBetaN", f"{res['theta_npe_mean'][0]:.2f}"); c("MalBetaM", f"{res['theta_nuts_mean'][0]:.2f}")
        c("MalSigN", f"{res['theta_npe_mean'][1]:.2f}"); c("MalSigM", f"{res['theta_nuts_mean'][1]:.2f}")
        c("MalEllN", f"{res['theta_npe_mean'][2]:.2f}"); c("MalEllM", f"{res['theta_nuts_mean'][2]:.2f}")
    print("wrote malaria macros")


if __name__ == "__main__":
    main()
