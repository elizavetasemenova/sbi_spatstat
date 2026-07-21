"""Spectral (FFT-based) Log-Gaussian Cox process on the torus [0,1]^2.

The latent field is a stationary Gaussian process represented in the Fourier
basis, which diagonalises any stationary kernel on the torus:

    f(s) = sum_k c_k exp(i 2 pi k . s),   c_k ~ CN(0, S_theta(k)),

with S_theta the Matern spectral density.  This gives (a) exact, O(M^2 log M)
simulation via the FFT, (b) a *prior-aligned* coordinate system in which the
field posterior is a distribution over the ordered coefficients c_k, and (c) a
natural truncation (drop high frequencies) whose error is the tail spectral mass.

Observations are a point pattern: N ~ Poisson(int lambda), lambda = exp(beta0+f),
with locations drawn with density proportional to lambda.  The pattern is encoded
*grid-free* by its low-frequency Fourier features (the point-process periodogram),
    Phi_k(X) = sum_j exp(-i 2 pi k . s_j),
which are exactly the statistics that inform the low-frequency field coefficients.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def matern_spectral_density(kx, ky, sigma, ell, nu, d=2):
    """Matern spectral density S(omega) on R^d evaluated at angular freq 2*pi*k.

    Returns an array matching the broadcast of kx, ky (integer wavenumbers).
    Normalised afterwards so that the field has marginal variance sigma^2.
    """
    omega2 = (2 * math.pi * kx) ** 2 + (2 * math.pi * ky) ** 2
    a = 2 * nu / (ell ** 2)
    # unnormalised Matern SDF shape; global constant folded into normalisation
    S = (a + omega2) ** (-(nu + d / 2.0))
    return S


@dataclass
class SpectralLGCP:
    M: int = 64            # field grid resolution (torus)
    nu: float = 0.5        # Matern smoothness of the generative kernel

    def __post_init__(self):
        M = self.M
        idx = np.arange(M)
        # torus (wrapped) lag in units of grid spacing, along each axis
        self._dx = np.minimum(idx, M - idx)
        c = (np.arange(M) + 0.5) / M
        self.gx, self.gy = np.meshgrid(c, c, indexing="ij")   # cell centres

    def _torus_cov(self, sigma, ell):
        """Covariance grid c(lag) = sigma^2 exp(-r/ell) on the torus (Matern-1/2)."""
        dxx, dyy = np.meshgrid(self._dx, self._dx, indexing="ij")
        r = np.sqrt(dxx ** 2 + dyy ** 2) / self.M
        if self.nu == 0.5:
            return (sigma ** 2) * np.exp(-r / ell)
        if self.nu == 1.5:
            s = math.sqrt(3) * r / ell
            return (sigma ** 2) * (1 + s) * np.exp(-s)
        if self.nu == 2.5:
            s = math.sqrt(5) * r / ell
            return (sigma ** 2) * (1 + s + s ** 2 / 3.0) * np.exp(-s)
        raise ValueError(self.nu)

    def spectrum(self, sigma, ell):
        """Circulant eigenvalues = discrete spectral density of the torus GP."""
        cov = self._torus_cov(sigma, ell)
        Lam = np.fft.fft2(cov).real
        return np.clip(Lam, 0.0, None)                    # tiny negatives -> 0

    # --- field simulation (exact torus GP via circulant embedding) ---------
    def sample_field(self, beta0, sigma, ell, rng):
        """Return a real log-intensity field Z on the M x M torus grid with
        covariance sigma^2 exp(-r/ell) (torus)."""
        Lam = self.spectrum(sigma, ell)
        xi = rng.standard_normal((self.M, self.M)) + 1j * rng.standard_normal((self.M, self.M))
        f = np.fft.ifft2(np.sqrt(Lam) * xi).real * self.M   # exact stationary GP, var sigma^2
        return beta0 + f

    def sample_points(self, Z, rng, area=1.0):
        """Draw an inhomogeneous Poisson point pattern with intensity exp(Z)."""
        lam = np.exp(Z)
        cell = area / (self.M ** 2)
        mean_counts = lam * cell
        counts = rng.poisson(mean_counts)
        N = int(counts.sum())
        if N == 0:
            return np.zeros((0, 2))
        idx = np.repeat(np.arange(self.M * self.M), counts.ravel())
        ci, cj = np.divmod(idx, self.M)
        jitter = rng.random((N, 2))
        xs = (ci + jitter[:, 0]) / self.M
        ys = (cj + jitter[:, 1]) / self.M
        return np.stack([xs, ys], axis=1)

    # --- grid-free spectral features of a point pattern --------------------
    def point_features(self, pts, kmax):
        """Low-frequency Fourier features Phi_k(X) for |k_x|,|k_y| <= kmax.

        Returns a real vector [logN, Re Phi (normalised), Im Phi ...].
        """
        N = pts.shape[0]
        ks = np.arange(-kmax, kmax + 1)
        KX, KY = np.meshgrid(ks, ks, indexing="ij")
        KX, KY = KX.ravel(), KY.ravel()
        if N == 0:
            phi = np.zeros(KX.size, dtype=complex)
        else:
            ang = -2 * math.pi * (np.outer(pts[:, 0], KX) + np.outer(pts[:, 1], KY))
            phi = np.exp(1j * ang).sum(0) / math.sqrt(max(N, 1))
        feat = np.concatenate([[math.log1p(N)], phi.real, phi.imag]).astype(np.float32)
        return feat

    def field_coefficients(self, Z, beta0, kmax):
        """Low-frequency Fourier coefficients of the field f = Z - beta0.

        Returns stacked [Re c_k, Im c_k] for |k_x|,|k_y| <= kmax (the target the
        posterior must recover).  Uses the FFT of the field on the M-grid.
        """
        f = Z - beta0
        C = np.fft.fft2(f) / (self.M ** 2)
        ks = np.arange(-kmax, kmax + 1)
        coeffs = []
        for kx in ks:
            for ky in ks:
                coeffs.append(C[kx % self.M, ky % self.M])
        coeffs = np.array(coeffs)
        return np.concatenate([coeffs.real, coeffs.imag]).astype(np.float32)

    def reconstruct(self, coeff_vec, beta0, kmax, out_grid=None):
        """Reconstruct the field Z on an (optionally different) grid from the
        low-frequency coefficients -- resolution-free evaluation."""
        n = (2 * kmax + 1) ** 2
        re, im = coeff_vec[:n], coeff_vec[n:]
        C = re + 1j * im
        G = out_grid if out_grid is not None else self.M
        c = (np.arange(G) + 0.5) / G
        gx, gy = np.meshgrid(c, c, indexing="ij")
        ks = np.arange(-kmax, kmax + 1)
        Z = np.full((G, G), float(beta0))
        idx = 0
        for kx in ks:
            for ky in ks:
                phase = np.exp(2j * math.pi * (kx * gx + ky * gy))
                Z += (C[idx] * phase).real
                idx += 1
        return Z
