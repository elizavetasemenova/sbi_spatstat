"""Generate training data for the spectral FMPE posterior."""

from __future__ import annotations
import numpy as np

from .simulator import SpectralLGCP
from .basis import SpectralBasis
from . import priors


def _coarse_counts(pts, Gc):
    """log1p coarse count grid (Gc x Gc), flattened -- anchors level/dispersion."""
    if pts.shape[0] == 0:
        return np.zeros(Gc * Gc, np.float32)
    ix = np.clip((pts[:, 0] * Gc).astype(int), 0, Gc - 1)
    iy = np.clip((pts[:, 1] * Gc).astype(int), 0, Gc - 1)
    H = np.zeros((Gc, Gc), np.float32)
    np.add.at(H, (ix, iy), 1.0)
    return np.log1p(H).ravel()


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
    Gc = 10                                                # coarse count summary
    for i in range(n):
        b0, sig, ell = theta[i]
        Z = sim.sample_field(b0, sig, ell, rng)
        fields[i] = Z
        coeffs[i] = basis.field_to_coeffs(Z - b0)          # field part only
        pts = sim.sample_points(Z, rng)
        spec = sim.point_features(pts, feat_kmax)          # grid-free spectral features
        cc = _coarse_counts(pts, Gc)                       # anchors level / variance
        feats.append(np.concatenate([spec, cc]))
    PHI = np.stack(feats).astype(np.float32)
    theta_u = priors.to_unit(theta)
    x1 = np.concatenate([theta_u, coeffs / prior_std[None]], axis=1).astype(np.float32)
    return dict(theta=theta, x1=x1, phi=PHI, prior_std=prior_std, fields=fields,
                sim=sim, basis=basis)
