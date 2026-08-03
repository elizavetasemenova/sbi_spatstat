"""Grid-discretized Log-Gaussian Cox process: prior, simulator, transforms.

Model
-----
On the unit square :math:`D = [0, 1]^2` discretised into a ``G x G`` regular
grid of cells with centroids :math:`s_i` and equal area :math:`a = 1/G^2`, the
log-intensity surface is

    Z(s) = beta0 + f(s),      f ~ GP(0, k_theta),

with a Matern covariance ``k(s, s') = sigma^2 * M_nu(||s - s'|| / ell)``.  The
default kernel is the exponential (Matern-1/2) kernel used throughout the
experiments; ``nu = 1.5`` and ``2.5`` are available for misspecification
studies.  Cell counts are conditionally Poisson,

    y_i | Z ~ Poisson(a * exp(Z_i)).

The unknown parameters are ``theta = (beta0, sigma, ell)``:

* ``beta0``  -- baseline log-intensity (overall abundance),
* ``sigma``  -- marginal standard deviation of the log-intensity field
  (degree of spatial aggregation / over-dispersion),
* ``ell``    -- correlation length (spatial scale of aggregation).

All three are scientifically interpretable quantities in ecology and spatial
epidemiology, which is exactly the setting where fast amortized inference is
valuable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Prior specification.  Parameters live in a bounded natural space; we map them
# to an unconstrained space for the normalising flow so that posterior samples
# always respect the prior support.
# ---------------------------------------------------------------------------

#: (low, high) support of the (approximately) marginal priors.
BETA0_MEAN, BETA0_SD = 5.0, 0.7          # beta0 ~ Normal(5.0, 0.7^2)
SIGMA_LOW, SIGMA_HIGH = 0.20, 1.80       # sigma ~ Uniform
ELL_LOW, ELL_HIGH = 0.05, 0.50           # ell   ~ Uniform

PARAM_NAMES = (r"$\beta_0$", r"$\sigma$", r"$\ell$")
PARAM_KEYS = ("beta0", "sigma", "ell")
N_PARAMS = 3


def sample_prior(n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw ``n`` parameter vectors ``(beta0, sigma, ell)`` from the prior."""
    beta0 = rng.normal(BETA0_MEAN, BETA0_SD, size=n)
    sigma = rng.uniform(SIGMA_LOW, SIGMA_HIGH, size=n)
    ell = rng.uniform(ELL_LOW, ELL_HIGH, size=n)
    return np.stack([beta0, sigma, ell], axis=1).astype(np.float64)


def prior_log_prob(theta: np.ndarray) -> np.ndarray:
    """Log density of the prior at ``theta`` (shape ``[n, 3]``)."""
    beta0, sigma, ell = theta[:, 0], theta[:, 1], theta[:, 2]
    lp = -0.5 * ((beta0 - BETA0_MEAN) / BETA0_SD) ** 2 - math.log(
        BETA0_SD * math.sqrt(2 * math.pi)
    )
    in_sigma = (sigma >= SIGMA_LOW) & (sigma <= SIGMA_HIGH)
    in_ell = (ell >= ELL_LOW) & (ell <= ELL_HIGH)
    lp = lp - math.log(SIGMA_HIGH - SIGMA_LOW) - math.log(ELL_HIGH - ELL_LOW)
    lp = np.where(in_sigma & in_ell, lp, -np.inf)
    return lp


# ---------------------------------------------------------------------------
# Transform between natural parameters and the unconstrained space used by the
# flow.  beta0 is standardised; sigma and ell are mapped through a logit of
# their (rescaled) uniform support and then standardised, so that the flow
# operates on roughly zero-mean, unit-scale coordinates on all of R^3.
# ---------------------------------------------------------------------------

def _logit(u: torch.Tensor) -> torch.Tensor:
    u = u.clamp(1e-6, 1 - 1e-6)
    return torch.log(u) - torch.log1p(-u)


def _sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)


# Empirical mean/std of the logit-transformed uniforms (Uniform(0,1) -> logit),
# used only to standardise; computed once for reproducibility.
_LOGIT_MEAN = 0.0
_LOGIT_STD = 1.8137993642  # std of logit(U), U~Uniform(0,1)


def to_unconstrained(theta: torch.Tensor) -> torch.Tensor:
    """Map natural ``theta`` (``[n, 3]``) to standardised unconstrained space."""
    beta0, sigma, ell = theta[:, 0], theta[:, 1], theta[:, 2]
    u_beta = (beta0 - BETA0_MEAN) / BETA0_SD
    u_sigma = (_logit((sigma - SIGMA_LOW) / (SIGMA_HIGH - SIGMA_LOW)) - _LOGIT_MEAN) / _LOGIT_STD
    u_ell = (_logit((ell - ELL_LOW) / (ELL_HIGH - ELL_LOW)) - _LOGIT_MEAN) / _LOGIT_STD
    return torch.stack([u_beta, u_sigma, u_ell], dim=1)


def from_unconstrained(u: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`to_unconstrained`."""
    u_beta, u_sigma, u_ell = u[:, 0], u[:, 1], u[:, 2]
    beta0 = u_beta * BETA0_SD + BETA0_MEAN
    sigma = SIGMA_LOW + (SIGMA_HIGH - SIGMA_LOW) * _sigmoid(u_sigma * _LOGIT_STD + _LOGIT_MEAN)
    ell = ELL_LOW + (ELL_HIGH - ELL_LOW) * _sigmoid(u_ell * _LOGIT_STD + _LOGIT_MEAN)
    return torch.stack([beta0, sigma, ell], dim=1)


# ---------------------------------------------------------------------------
# Geometry and kernels.
# ---------------------------------------------------------------------------

def grid_coords(G: int) -> np.ndarray:
    """Return ``[G*G, 2]`` cell-centroid coordinates on the unit square."""
    c = (np.arange(G) + 0.5) / G
    xx, yy = np.meshgrid(c, c, indexing="ij")
    return np.stack([xx.ravel(), yy.ravel()], axis=1)


def _pairwise_dist(coords: np.ndarray) -> np.ndarray:
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((diff ** 2).sum(-1))


def matern_correlation(dist: torch.Tensor, ell: torch.Tensor, nu: float) -> torch.Tensor:
    """Matern correlation matrices for a batch of length-scales.

    ``dist`` is ``[m, m]`` (shared grid), ``ell`` is ``[B]``; returns ``[B, m, m]``.
    Supports ``nu in {0.5, 1.5, 2.5}`` in closed form.
    """
    d = dist[None] / ell[:, None, None]
    if nu == 0.5:
        return torch.exp(-d)
    if nu == 1.5:
        s = math.sqrt(3.0) * d
        return (1.0 + s) * torch.exp(-s)
    if nu == 2.5:
        s = math.sqrt(5.0) * d
        return (1.0 + s + s ** 2 / 3.0) * torch.exp(-s)
    raise ValueError(f"unsupported nu={nu}")


@dataclass
class LGCPSimulator:
    """Batched LGCP simulator on a fixed ``G x G`` grid.

    Parameters
    ----------
    G:      grid resolution (number of cells per side).
    nu:     Matern smoothness of the generative kernel.
    jitter: diagonal added before the Cholesky factorisation.
    """

    G: int = 16
    nu: float = 0.5
    jitter: float = 1e-5

    def __post_init__(self) -> None:
        self.coords = grid_coords(self.G)
        self.n_cells = self.G * self.G
        self.area = 1.0 / self.n_cells
        self._dist = torch.tensor(_pairwise_dist(self.coords), dtype=torch.float64)

    def simulate(
        self,
        theta: np.ndarray,
        rng: np.random.Generator,
        return_field: bool = False,
    ):
        """Simulate count grids for parameter matrix ``theta`` (``[B, 3]``).

        Returns counts ``y`` of shape ``[B, G, G]`` (float32).  If
        ``return_field`` also returns the latent log-intensity ``Z``.
        """
        theta_t = torch.as_tensor(theta, dtype=torch.float64)
        B = theta_t.shape[0]
        beta0 = theta_t[:, 0]
        sigma = theta_t[:, 1]
        ell = theta_t[:, 2]

        corr = matern_correlation(self._dist, ell, self.nu)          # [B, m, m]
        cov = (sigma[:, None, None] ** 2) * corr
        cov = cov + self.jitter * torch.eye(self.n_cells, dtype=torch.float64)[None]
        L = torch.linalg.cholesky(cov)                                # [B, m, m]

        eps = torch.as_tensor(
            rng.standard_normal((B, self.n_cells)), dtype=torch.float64
        )
        f = torch.einsum("bij,bj->bi", L, eps)                        # [B, m]
        Z = beta0[:, None] + f                                        # [B, m]
        lam = self.area * torch.exp(Z)
        lam = lam.clamp(max=1e6)
        counts = torch.as_tensor(
            rng.poisson(lam.numpy()), dtype=torch.float32
        )
        y = counts.reshape(B, self.G, self.G)
        if return_field:
            return y, Z.reshape(B, self.G, self.G).to(torch.float32)
        return y

    def simulate_dataset(
        self, n: int, rng: np.random.Generator, batch: int = 2048
    ):
        """Draw ``theta`` from the prior and simulate ``n`` datasets."""
        theta = sample_prior(n, rng)
        ys = []
        for start in range(0, n, batch):
            th = theta[start : start + batch]
            ys.append(self.simulate(th, rng))
        y = torch.cat(ys, dim=0)
        return theta, y


# ---------------------------------------------------------------------------
# Hand-crafted spatial summary statistics (for the ablation baseline).
# ---------------------------------------------------------------------------

def handcrafted_summaries(y: torch.Tensor, n_rings: int = 6) -> torch.Tensor:
    """Interpretable spatial summaries of count grids ``y`` (``[B, G, G]``).

    Returns a feature matrix combining global count statistics with the
    empirical spatial autocorrelation profile (a binned pair-correlation of the
    count field), which is what a spatial statistician would compute by hand.
    """
    B, G, _ = y.shape
    flat = y.reshape(B, -1)
    total = flat.sum(1)
    logN = torch.log1p(total)
    mean = flat.mean(1)
    var = flat.var(1)
    # index of dispersion (var/mean) -- clustering signal
    disp = var / (mean + 1e-6)
    frac_zero = (flat == 0).float().mean(1)
    qmax = flat.max(1).values

    # Radially binned spatial autocorrelation of the (log) count field via FFT.
    logy = torch.log1p(y)
    logy = logy - logy.mean(dim=(1, 2), keepdim=True)
    F = torch.fft.rfft2(logy)
    power = (F.real ** 2 + F.imag ** 2)
    acf = torch.fft.irfft2(power, s=(G, G))                # unnormalised autocov
    acf = acf / (acf[:, :1, :1] + 1e-8)                    # normalise by lag-0

    # distance of each lag on the torus
    idx = torch.arange(G)
    dd = torch.minimum(idx, G - idx).float()
    lag = torch.sqrt(dd[:, None] ** 2 + dd[None, :] ** 2)  # [G, G]
    maxlag = lag.max()
    edges = torch.linspace(0.5, float(maxlag), n_rings + 1)
    rings = []
    for r in range(n_rings):
        m = (lag >= edges[r]) & (lag < edges[r + 1])
        if m.any():
            rings.append(acf[:, m].mean(1))
        else:
            rings.append(torch.zeros(B))
    ring_feats = torch.stack(rings, dim=1)

    feats = torch.cat(
        [
            logN[:, None], mean[:, None], var[:, None], disp[:, None],
            frac_zero[:, None], torch.log1p(qmax)[:, None], ring_feats,
        ],
        dim=1,
    )
    return feats
