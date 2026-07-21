"""Prior over LGCP hyperparameters (beta0, sigma, ell) and unconstraining maps."""

from __future__ import annotations
import numpy as np

BETA0_MEAN, BETA0_SD = 5.0, 0.6
SIGMA_LOW, SIGMA_HIGH = 0.3, 1.8
ELL_LOW, ELL_HIGH = 0.03, 0.30
PARAM_NAMES = (r"$\beta_0$", r"$\sigma$", r"$\ell$")


def sample_prior(n, rng):
    beta0 = rng.normal(BETA0_MEAN, BETA0_SD, n)
    sigma = rng.uniform(SIGMA_LOW, SIGMA_HIGH, n)
    ell = rng.uniform(ELL_LOW, ELL_HIGH, n)
    return np.stack([beta0, sigma, ell], 1).astype(np.float64)


def to_unit(theta):
    """Map theta to approx-standard-normal coordinates for the flow target."""
    b = (theta[:, 0] - BETA0_MEAN) / BETA0_SD
    s = (theta[:, 1] - SIGMA_LOW) / (SIGMA_HIGH - SIGMA_LOW)          # in (0,1)
    e = (theta[:, 2] - ELL_LOW) / (ELL_HIGH - ELL_LOW)
    # probit of the uniforms so the prior is ~N(0,1)
    from scipy.stats import norm
    return np.stack([b, norm.ppf(np.clip(s, 1e-4, 1 - 1e-4)),
                     norm.ppf(np.clip(e, 1e-4, 1 - 1e-4))], 1).astype(np.float32)


def from_unit(u):
    from scipy.stats import norm
    b = u[:, 0] * BETA0_SD + BETA0_MEAN
    s = SIGMA_LOW + (SIGMA_HIGH - SIGMA_LOW) * norm.cdf(u[:, 1])
    e = ELL_LOW + (ELL_HIGH - ELL_LOW) * norm.cdf(u[:, 2])
    return np.stack([b, s, e], 1)
