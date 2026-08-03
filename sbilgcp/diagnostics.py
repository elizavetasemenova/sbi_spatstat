"""Calibration and posterior-comparison diagnostics for SBI."""

from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold


# ---------------------------------------------------------------------------
# Simulation-based calibration (SBC).
# ---------------------------------------------------------------------------

def sbc_ranks(true_theta: np.ndarray, posterior_samples: np.ndarray) -> np.ndarray:
    """Rank of each true parameter among its posterior draws.

    ``true_theta``:        ``[M, P]``
    ``posterior_samples``: ``[M, L, P]``
    Returns integer ranks ``[M, P]`` in ``{0, ..., L}``.
    """
    return (posterior_samples < true_theta[:, None, :]).sum(axis=1)


def sbc_ecdf_bands(n_ranks, L, n_sims, alpha=0.05):
    """Simultaneous confidence band for the rank ECDF under uniformity.

    Returns ``(grid, lower, upper)`` on the normalised-rank axis ``[0, 1]``.
    Uses the pointwise Binomial band (a standard, slightly conservative choice).
    """
    grid = np.linspace(0, 1, n_ranks)
    lo = stats.binom.ppf(alpha / 2, n_sims, grid) / n_sims
    hi = stats.binom.ppf(1 - alpha / 2, n_sims, grid) / n_sims
    return grid, lo, hi


def rank_uniformity_pvalue(ranks_1d, L):
    """Chi-squared goodness-of-fit p-value for uniform ranks in ``{0..L}``."""
    n_bins = min(20, L + 1)
    counts, _ = np.histogram(ranks_1d, bins=n_bins, range=(0, L + 1))
    expected = np.full(n_bins, ranks_1d.size / n_bins)
    chi2 = ((counts - expected) ** 2 / expected).sum()
    dof = n_bins - 1
    return float(stats.chi2.sf(chi2, dof))


# ---------------------------------------------------------------------------
# Coverage of central credible intervals.
# ---------------------------------------------------------------------------

def credible_coverage(true_theta, posterior_samples, levels):
    """Empirical coverage of central credible intervals at nominal ``levels``.

    Returns array ``[len(levels), P]`` of empirical coverage.
    """
    M, L, P = posterior_samples.shape
    cov = np.zeros((len(levels), P))
    for j, lv in enumerate(levels):
        lo = np.quantile(posterior_samples, (1 - lv) / 2, axis=1)
        hi = np.quantile(posterior_samples, 1 - (1 - lv) / 2, axis=1)
        cov[j] = ((true_theta >= lo) & (true_theta <= hi)).mean(0)
    return cov


# ---------------------------------------------------------------------------
# Point-estimate quality.
# ---------------------------------------------------------------------------

def recovery_metrics(true_theta, posterior_samples):
    """RMSE, MAE, R^2 and mean posterior z-score per parameter."""
    post_mean = posterior_samples.mean(1)
    post_std = posterior_samples.std(1)
    err = post_mean - true_theta
    rmse = np.sqrt((err ** 2).mean(0))
    mae = np.abs(err).mean(0)
    ss_res = (err ** 2).sum(0)
    ss_tot = ((true_theta - true_theta.mean(0)) ** 2).sum(0)
    r2 = 1 - ss_res / ss_tot
    z = err / (post_std + 1e-8)
    return {
        "rmse": rmse, "mae": mae, "r2": r2,
        "z_mean": z.mean(0), "z_std": z.std(0),
    }


# ---------------------------------------------------------------------------
# Posterior-vs-posterior comparison (SBI against MCMC reference).
# ---------------------------------------------------------------------------

def c2st(x, y, n_folds=5, seed=0):
    """Classifier two-sample test accuracy between samples ``x`` and ``y``.

    Returns cross-validated accuracy; 0.5 means the two sample sets are
    indistinguishable (posteriors agree), 1.0 means perfectly separable.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    n = min(len(x), len(y))
    rng = np.random.default_rng(seed)
    x = x[rng.choice(len(x), n, replace=False)]
    y = y[rng.choice(len(y), n, replace=False)]
    data = np.concatenate([x, y], 0)
    labels = np.concatenate([np.zeros(n), np.ones(n)])
    mu, sd = data.mean(0), data.std(0) + 1e-8
    data = (data - mu) / sd
    accs = []
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for tr, te in skf.split(data, labels):
        clf = MLPClassifier(hidden_layer_sizes=(32, 32), max_iter=300, random_state=seed)
        clf.fit(data[tr], labels[tr])
        accs.append(clf.score(data[te], labels[te]))
    return float(np.mean(accs))


def wasserstein1_marginal(x, y):
    """1-Wasserstein distance per marginal dimension."""
    return np.array([stats.wasserstein_distance(x[:, d], y[:, d]) for d in range(x.shape[1])])


def posterior_mean_std_agreement(theta_ref, theta_npe):
    """Compare posterior mean/std between two sample sets for one dataset."""
    return {
        "mean_ref": theta_ref.mean(0), "mean_npe": theta_npe.mean(0),
        "std_ref": theta_ref.std(0), "std_npe": theta_npe.std(0),
    }
