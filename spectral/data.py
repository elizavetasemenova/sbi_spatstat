"""Generate training data for the spectral FMPE posterior."""

from __future__ import annotations
import numpy as np

from .simulator import SpectralLGCP
from .basis import SpectralBasis
from . import priors


def generate(n, M=64, nu=1.5, kmax=6, feat_kmax=8, seed=0, prior_std=None, sim=None, basis=None):
    """Return theta [n,3], target x1 [n, 3+D], features PHI [n,F], and the
    per-coefficient prior std used for whitening."""
    sim = sim or SpectralLGCP(M=M, nu=nu)
    basis = basis or SpectralBasis(kmax)
    rng = np.random.default_rng(seed)
    theta = priors.sample_prior(n, rng)
    if prior_std is None:
        prior_std = basis.prior_std(sim, n=1500, seed=seed + 777)
    D = basis.dim
    coeffs = np.zeros((n, D), np.float32)
    feats = []
    fields = np.zeros((n, M, M), np.float32)
    for i in range(n):
        b0, sig, ell = theta[i]
        Z = sim.sample_field(b0, sig, ell, rng)
        fields[i] = Z
        coeffs[i] = basis.field_to_coeffs(Z - b0)          # field part only
        pts = sim.sample_points(Z, rng)
        feats.append(sim.point_features(pts, feat_kmax))
    PHI = np.stack(feats).astype(np.float32)
    theta_u = priors.to_unit(theta)
    x1 = np.concatenate([theta_u, coeffs / prior_std[None]], axis=1).astype(np.float32)
    return dict(theta=theta, x1=x1, phi=PHI, prior_std=prior_std, fields=fields,
                sim=sim, basis=basis)
