"""Amortized disease mapping: a Poisson-offset LGCP for prevalence-survey data.

Prevalence surveys report, at each location, the number ``examined`` and the
number ``positive``.  The standard spatial model is a log-Gaussian Cox / Poisson
model with a known offset,

    Y_i ~ Poisson( E_i * exp(Z_i) ),   Z_i = beta0 + f_i,   f ~ GP(0, k),

where ``E_i`` is the number examined in cell ``i`` (the offset), ``exp(Z_i)`` is
the malaria risk (relative rate), and ``beta0`` is the log baseline risk.  We
amortize inference for a *fixed* survey design: the offset grid ``E`` is held at
the observed values for every simulation, so after training, inference on the
real positive-count grid is a single forward pass.  Cells with ``E_i = 0`` are
unsurveyed and carry no likelihood; the field there is inferred from neighbours.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from . import lgcp
from .flows import MAF

# log baseline-risk prior spans very-low to high transmission
BETA0_LOW, BETA0_HIGH = -3.5, -0.1
PARAM_NAMES = (r"$\beta_0$", r"$\sigma$", r"$\ell$")


def sample_prior(n, rng):
    beta0 = rng.uniform(BETA0_LOW, BETA0_HIGH, size=n)
    sigma = rng.uniform(lgcp.SIGMA_LOW, lgcp.SIGMA_HIGH, size=n)
    ell = rng.uniform(lgcp.ELL_LOW, lgcp.ELL_HIGH, size=n)
    return np.stack([beta0, sigma, ell], axis=1)


def to_unconstrained(theta):
    b = (theta[:, 0] - BETA0_LOW) / (BETA0_HIGH - BETA0_LOW)
    ub = (lgcp._logit(b) - lgcp._LOGIT_MEAN) / lgcp._LOGIT_STD
    us = (lgcp._logit((theta[:, 1] - lgcp.SIGMA_LOW) / (lgcp.SIGMA_HIGH - lgcp.SIGMA_LOW)) - lgcp._LOGIT_MEAN) / lgcp._LOGIT_STD
    ue = (lgcp._logit((theta[:, 2] - lgcp.ELL_LOW) / (lgcp.ELL_HIGH - lgcp.ELL_LOW)) - lgcp._LOGIT_MEAN) / lgcp._LOGIT_STD
    return torch.stack([ub, us, ue], dim=1)


def from_unconstrained(u):
    b = BETA0_LOW + (BETA0_HIGH - BETA0_LOW) * torch.sigmoid(u[:, 0] * lgcp._LOGIT_STD + lgcp._LOGIT_MEAN)
    s = lgcp.SIGMA_LOW + (lgcp.SIGMA_HIGH - lgcp.SIGMA_LOW) * torch.sigmoid(u[:, 1] * lgcp._LOGIT_STD + lgcp._LOGIT_MEAN)
    e = lgcp.ELL_LOW + (lgcp.ELL_HIGH - lgcp.ELL_LOW) * torch.sigmoid(u[:, 2] * lgcp._LOGIT_STD + lgcp._LOGIT_MEAN)
    return torch.stack([b, s, e], dim=1)


@dataclass
class DiseaseMapSimulator:
    """Poisson-offset LGCP on a fixed grid with a fixed offset field ``E``."""

    offset: np.ndarray            # [G, G] examined counts (fixed design)
    nu: float = 0.5
    jitter: float = 1e-5

    def __post_init__(self):
        self.G = self.offset.shape[0]
        self.core = lgcp.LGCPSimulator(G=self.G, nu=self.nu, jitter=self.jitter)
        self.E = torch.as_tensor(self.offset.reshape(-1), dtype=torch.float64)

    def simulate(self, theta, rng):
        theta_t = torch.as_tensor(theta, dtype=torch.float64)
        B = theta_t.shape[0]
        beta0, sigma, ell = theta_t[:, 0], theta_t[:, 1], theta_t[:, 2]
        corr = lgcp.matern_correlation(self.core._dist, ell, self.nu)
        cov = (sigma[:, None, None] ** 2) * corr + self.jitter * torch.eye(self.G ** 2, dtype=torch.float64)[None]
        L = torch.linalg.cholesky(cov)
        eps = torch.as_tensor(rng.standard_normal((B, self.G ** 2)), dtype=torch.float64)
        f = torch.einsum("bij,bj->bi", L, eps)
        Z = beta0[:, None] + f
        rate = (self.E[None] * torch.exp(Z)).clamp(max=1e6)
        Y = torch.as_tensor(rng.poisson(rate.numpy()), dtype=torch.float32)
        return Y.reshape(B, self.G, self.G)

    def simulate_dataset(self, n, rng, batch=1024):
        theta = sample_prior(n, rng)
        out = []
        for s in range(0, n, batch):
            out.append(self.simulate(theta[s:s + batch], rng))
        return theta, torch.cat(out, 0)

    def two_channel(self, Y):
        """Stack observed positives and the (fixed) offset as a 2-channel image."""
        Yt = torch.as_tensor(Y, dtype=torch.float32)
        if Yt.dim() == 2:
            Yt = Yt[None]
        E = torch.as_tensor(self.offset, dtype=torch.float32)[None].expand(Yt.shape[0], -1, -1)
        return torch.stack([torch.log1p(Yt), torch.log1p(E)], dim=1)


class DMEmbedding(nn.Module):
    """2-channel CNN embedding of (positives, offset)."""

    def __init__(self, embedding_dim=48):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 24, 3, padding=1), nn.ReLU(),
            nn.Conv2d(24, 32, 3, padding=1), nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(32, 48, 3, padding=1), nn.ReLU(),
            nn.Conv2d(48, 48, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(nn.Linear(48 + 2, 96), nn.ReLU(), nn.Linear(96, embedding_dim))
        self.embedding_dim = embedding_dim

    def forward(self, x):
        h = self.conv(x).flatten(1)
        Y = torch.expm1(x[:, 0])
        g = torch.stack([torch.log1p(Y.sum((1, 2))), torch.log1p(Y.var((1, 2)))], dim=1)
        return self.head(torch.cat([h, g], dim=1))


class DMNPE(nn.Module):
    def __init__(self, n_transforms=5, hidden=64, seed=0):
        super().__init__()
        self.embedding = DMEmbedding(48)
        self.flow = MAF(3, 48, n_transforms, hidden, seed=seed)

    def log_prob(self, theta_u, x):
        return self.flow.log_prob(theta_u, self.embedding(x))

    @torch.no_grad()
    def sample(self, x, n_per=1000):
        return self.flow.sample(self.embedding(x), n_per=n_per)
