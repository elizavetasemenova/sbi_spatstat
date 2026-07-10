"""LGCP under a *preferential* (informative) observation process.

Standard LGCP inference assumes the set of observed locations is chosen
independently of the latent field ("ignorable" / non-preferential design).  In
many real studies it is not: monitors are placed where the phenomenon is
strong, sightings are reported where animals are abundant, samples are taken
where a biofilm looks dense.  Ignoring this **preferential sampling** biases the
inferred intensity and variance (Diggle, Menezes & Su, 2010).

We extend the grid LGCP with an explicit, tunable preferential design and use it
to make a *methodological* point: because amortized SBI is likelihood-free, the
informative design can simply be added to the simulator, and a design-aware
neural posterior estimator recovers unbiased, calibrated parameters *and* the
strength of the preferentiality -- something a design-ignorant analysis cannot.

Model
-----
    Z_i     = beta0 + f_i,           f ~ N(0, K(sigma, ell))   (exponential kernel)
    pi_i    = sigmoid(gamma * f_i)                             (selection prob.)
    r_i     ~ Bernoulli(pi_i)                                  (observed?)
    y_i     ~ Poisson(area * exp(Z_i))  observed only where r_i = 1

At ``gamma = 0`` the design is uniform (each cell observed w.p. 1/2,
independent of the field); ``gamma > 0`` preferentially observes
high-intensity cells.  The unknowns are ``theta = (beta0, sigma, ell, gamma)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from . import lgcp
from .flows import MAF

GAMMA_LOW, GAMMA_HIGH = 0.0, 3.0
PARAM_NAMES = (r"$\beta_0$", r"$\sigma$", r"$\ell$", r"$\gamma$")
N_PARAMS = 4


# --- prior / transforms (reuse the 3-parameter core, append gamma) ----------

def sample_prior(n, rng):
    base = lgcp.sample_prior(n, rng)                      # (beta0, sigma, ell)
    gamma = rng.uniform(GAMMA_LOW, GAMMA_HIGH, size=n)
    return np.concatenate([base, gamma[:, None]], axis=1)


def to_unconstrained(theta):
    base = lgcp.to_unconstrained(theta[:, :3])
    g = theta[:, 3]
    u_g = (lgcp._logit((g - GAMMA_LOW) / (GAMMA_HIGH - GAMMA_LOW)) - lgcp._LOGIT_MEAN) / lgcp._LOGIT_STD
    return torch.cat([base, u_g[:, None]], dim=1)


def from_unconstrained(u):
    base = lgcp.from_unconstrained(u[:, :3])
    g = GAMMA_LOW + (GAMMA_HIGH - GAMMA_LOW) * torch.sigmoid(u[:, 3] * lgcp._LOGIT_STD + lgcp._LOGIT_MEAN)
    return torch.cat([base, g[:, None]], dim=1)


# --- simulator --------------------------------------------------------------

@dataclass
class PreferentialSimulator:
    G: int = 16
    nu: float = 0.5
    jitter: float = 1e-5

    def __post_init__(self):
        self.core = lgcp.LGCPSimulator(G=self.G, nu=self.nu, jitter=self.jitter)
        self.area = self.core.area

    def simulate(self, theta, rng):
        """Return 2-channel observation tensors ``[B, 2, G, G]``.

        Channel 0 = log1p(observed counts) (0 where unobserved);
        channel 1 = observation mask r.
        """
        theta = np.asarray(theta, dtype=np.float64)
        y, Z = self.core.simulate(theta[:, :3], rng, return_field=True)  # [B,G,G]
        y = y.numpy(); Z = Z.numpy()
        B = theta.shape[0]
        beta0 = theta[:, 0][:, None, None]
        gamma = theta[:, 3][:, None, None]
        f = Z - beta0                                       # zero-mean field
        pi = 1.0 / (1.0 + np.exp(-gamma * f))
        r = (rng.random((B, self.G, self.G)) < pi).astype(np.float32)
        obs = np.log1p(y) * r
        data = np.stack([obs, r], axis=1).astype(np.float32)   # [B,2,G,G]
        return torch.as_tensor(data)

    def simulate_uniform(self, theta3, rng, frac=0.5):
        """Non-preferential (ignorable) design: each cell observed w.p. ``frac``.

        Used to train the design-ignorant baseline.  Returns the same 2-channel
        representation so the two estimators are strictly comparable.
        """
        y = self.core.simulate(theta3[:, :3], rng)         # [B,G,G]
        y = y.numpy()
        B = theta3.shape[0]
        r = (rng.random((B, self.G, self.G)) < frac).astype(np.float32)
        obs = np.log1p(y) * r
        data = np.stack([obs, r], axis=1).astype(np.float32)
        return torch.as_tensor(data)

    def simulate_dataset(self, n, rng, batch=1024, preferential=True):
        theta = sample_prior(n, rng)
        out = []
        for s in range(0, n, batch):
            th = theta[s:s + batch]
            if preferential:
                out.append(self.simulate(th, rng))
            else:
                out.append(self.simulate_uniform(th, rng))
        return theta, torch.cat(out, 0)


# --- design-aware embedding (2-channel CNN) ---------------------------------

class CNN2ch(nn.Module):
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
        # explicit globals: log #observed, mean observed log-count, obs fraction
        self.head = nn.Sequential(nn.Linear(48 + 3, 96), nn.ReLU(), nn.Linear(96, embedding_dim))
        self.embedding_dim = embedding_dim

    def forward(self, data):
        h = self.conv(data).flatten(1)
        obs, r = data[:, 0], data[:, 1]
        nobs = r.sum((1, 2))
        g = torch.stack([
            torch.log1p(nobs),
            obs.sum((1, 2)) / (nobs + 1e-6),
            r.mean((1, 2)),
        ], dim=1)
        return self.head(torch.cat([h, g], dim=1))


class NPEk(nn.Module):
    """NPE with a k-dimensional parameter flow and a given embedding."""

    def __init__(self, embedding, param_dim, n_transforms=5, hidden=64, seed=0):
        super().__init__()
        self.embedding = embedding
        self.flow = MAF(param_dim, embedding.embedding_dim, n_transforms, hidden, seed=seed)

    def log_prob(self, theta_u, x):
        return self.flow.log_prob(theta_u, self.embedding(x))

    @torch.no_grad()
    def sample(self, x, n_per=1000):
        return self.flow.sample(self.embedding(x), n_per=n_per)
