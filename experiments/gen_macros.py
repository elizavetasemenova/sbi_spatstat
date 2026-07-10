"""Emit paper/numbers.tex with LaTeX macros for every inline figure in the text."""

import json
import numpy as np
import config as C


def main():
    cal = json.load(open(C.F_CALIB))
    reports = json.load(open(C.F_REPORTS))
    lines = ["% auto-generated numeric macros -- do not edit by hand"]

    def cmd(name, val):
        lines.append(f"\\newcommand{{\\{name}}}{{{val}}}")

    # dataset / model sizes
    cmd("Grid", C.GRID)
    cmd("NCells", C.GRID * C.GRID)
    cmd("NTrain", f"{C.N_TRAIN:,}")
    cmd("NTest", f"{C.N_TEST:,}")
    cmd("NPost", f"{C.N_POSTERIOR:,}")
    cmd("NpeParams", f"{reports['npe_cnn']['n_params']:,}")
    cmd("TrainMin", f"{reports['npe_cnn']['wall']/60:.1f}")
    cmd("MCMCchains", C.MCMC_CHAINS)
    cmd("MCMCwarm", C.MCMC_WARMUP)
    cmd("MCMCsamp", C.MCMC_SAMPLES)

    # inference timing
    ms = 1e3 * cal["npe_infer_time_per_dataset"]
    cmd("NpeMs", f"{ms:.2f}")

    # calibration: SBC p-values and 90% coverage for NPE-CNN
    lv = cal["levels"].index(0.9)
    covn = np.array(cal["npe_cnn"]["coverage"])[lv]
    covr = np.array(cal["regressor"]["coverage"])[lv]
    cmd("CovNpeB", f"{covn[0]:.2f}"); cmd("CovNpeS", f"{covn[1]:.2f}"); cmd("CovNpeL", f"{covn[2]:.2f}")
    cmd("CovRegB", f"{covr[0]:.2f}"); cmd("CovRegS", f"{covr[1]:.2f}"); cmd("CovRegL", f"{covr[2]:.2f}")
    pmin = min(cal["npe_cnn"]["sbc_pvalues"])
    cmd("SbcPmin", f"{pmin:.2f}")

    # recovery R2 for NPE
    r2 = cal["npe_cnn"]["r2"]
    cmd("RtwoB", f"{r2[0]:.2f}"); cmd("RtwoS", f"{r2[1]:.2f}"); cmd("RtwoL", f"{r2[2]:.2f}")

    # MCMC comparison
    try:
        cmp = json.load(open(C.F_MCMC_CMP))
        cmd("CtstMean", f"{cmp['c2st_mean']:.3f}")
        cmd("NCompared", cmp["n_compared"])
        cmd("McmcSec", f"{cmp['mcmc_wall_median']:.0f}")
        speed = cmp.get("speedup", cmp["mcmc_wall_median"] / (ms / 1e3))
        cmd("Speedup", f"{speed:,.0f}")
        cross = reports['npe_cnn']['wall'] / (cmp['mcmc_wall_median'] - ms / 1e3)
        cmd("BreakEven", f"{cross:.0f}")
        wn = cmp["wasserstein_norm_mean"]
        cmd("WassMean", f"{np.mean(wn):.3f}")
        mc = cmp["post_mean_corr"]
        cmd("MeanCorrMin", f"{min(mc):.2f}")
    except FileNotFoundError:
        pass

    # misspecification 90% coverage (worst param) for nu=2.5
    m25 = np.array(cal["misspec"]["2.5"]["coverage"])[lv]
    cmd("CovMisB", f"{m25[0]:.2f}"); cmd("CovMisS", f"{m25[1]:.2f}"); cmd("CovMisL", f"{m25[2]:.2f}")

    with open(C.ROOT + "/paper/numbers.tex", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote paper/numbers.tex with", len(lines) - 1, "macros")


if __name__ == "__main__":
    main()
