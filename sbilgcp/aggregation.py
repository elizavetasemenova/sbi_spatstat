"""LGCP inference from spatially aggregated counts (change of support).

In most disease-mapping and socio-economic applications the point pattern is
never observed directly: counts are reported as totals over coarse
administrative regions.  Recovering fine-scale parameters (and the fine
intensity surface) from such aggregates is the *change-of-support* /
spatial-misalignment problem.  The aggregated likelihood involves sums of
correlated log-Gaussian Poisson rates within each region, which is awkward for
analytic/likelihood-based tools.  For amortized SBI it is, once again, just a
change to the simulator: aggregate the simulated fine counts to the observation
support and learn the posterior from the coarse totals.

Model
-----
Fine grid ``G x G`` as usual: ``Z_i = beta0 + f_i``, ``y_i ~ Poisson(a e^{Z_i})``.
The domain is partitioned into ``C x C`` coarse regions (each ``G/C`` cells wide);
the *observed* data are the region totals ``r_b = sum_{i in b} y_i``.  We infer
``theta = (beta0, sigma, ell)`` from ``r`` alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from . import lgcp
from .flows import MAF


def aggregate(y: torch.Tensor, C: int) -> torch.Tensor:
    """Sum a fine count grid ``[B,G,G]`` into ``C x C`` region totals ``[B,C,C]``."""
    B, G, _ = y.shape
    b = G // C
    return y.reshape(B, C, b, C, b).sum(dim=(2, 4))


@dataclass
class AggregationSimulator:
    G: int = 16
    C: int = 4                      # coarse regions per side
    nu: float = 0.5

    def __post_init__(self):
        self.core = lgcp.LGCPSimulator(G=self.G, nu=self.nu)
        self.area = self.core.area

    def simulate(self, theta, rng):
        y = self.core.simulate(theta[:, :3], rng)           # fine counts [B,G,G]
        return aggregate(y, self.C).to(torch.float32)        # [B,C,C]

    def simulate_dataset(self, n, rng, batch=1024):
        theta = lgcp.sample_prior(n, rng)
        out = []
        for s in range(0, n, batch):
            out.append(self.simulate(theta[s:s + batch], rng))
        return theta, torch.cat(out, 0)


class CoarseEmbedding(nn.Module):
    """Small CNN embedding of the coarse region-total grid (no pooling: C is small)."""

    def __init__(self, C=4, embedding_dim=48):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 48, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(nn.Linear(48 + 2, 96), nn.ReLU(), nn.Linear(96, embedding_dim))
        self.embedding_dim = embedding_dim

    def forward(self, r):
        x = torch.log1p(r).unsqueeze(1)
        h = self.conv(x).flatten(1)
        flat = r.reshape(r.shape[0], -1)
        g = torch.stack([torch.log1p(flat.sum(1)), torch.log1p(flat.var(1))], dim=1)
        return self.head(torch.cat([h, g], dim=1))


class AggNPE(nn.Module):
    def __init__(self, C=4, n_transforms=5, hidden=64, seed=0):
        super().__init__()
        self.embedding = CoarseEmbedding(C=C, embedding_dim=48)
        self.flow = MAF(lgcp.N_PARAMS, 48, n_transforms, hidden, seed=seed)

    def log_prob(self, theta_u, x):
        return self.flow.log_prob(theta_u, self.embedding(x))

    @torch.no_grad()
    def sample(self, x, n_per=1000):
        return self.flow.sample(self.embedding(x), n_per=n_per)
