"""Independent real spectral coefficients with resolution-free reconstruction.

We represent the low-frequency part of a real field on the torus by the
independent entries of its real FFT (``rfft2``) up to wavenumber ``kmax``.  The
coefficients are *normalised* (divided by the analysis grid size) so they are the
continuum Fourier coefficients and can be synthesised at any target resolution,
which is what makes the field estimator resolution-free.
"""

from __future__ import annotations

import numpy as np


def coeff_index_set(kmax):
    """Independent (kx, ky) rfft entries for |kx|,|ky| <= kmax on the torus.

    rfft2 stores ky in 0..M/2; kx in FFT order.  For a real field the only
    redundancy inside the ky>=0 half-plane is the ky==0 line, where
    C[-kx,0] = conj(C[kx,0]).  We keep kx in 0..kmax for ky==0 and both signs of
    kx (as kx and M-kx, resolved at build time) for ky=1..kmax.
    """
    idx = []
    # ky = 0 line: kx = 0..kmax (kx<0 is redundant)
    for kx in range(0, kmax + 1):
        idx.append((kx, 0))
    # ky = 1..kmax: kx = -kmax..kmax
    for ky in range(1, kmax + 1):
        for kx in range(-kmax, kmax + 1):
            idx.append((kx, ky))
    return idx


class SpectralBasis:
    def __init__(self, kmax):
        self.kmax = kmax
        self.index = coeff_index_set(kmax)
        # real DOF: each mode contributes (Re, Im) except purely-real ones
        self._real_only = {(0, 0)}   # DC is real; (others generally complex)

    @property
    def dim(self):
        # Re and Im for every mode, minus the imaginary DOF of DC (which is 0)
        return 2 * len(self.index) - 1

    def field_to_coeffs(self, f):
        """Real field [M,M] -> real coefficient vector (normalised)."""
        M = f.shape[0]
        R = np.fft.rfft2(f) / (M * M)          # continuum Fourier coefficients
        re, im = [], []
        for (kx, ky) in self.index:
            c = R[kx % M, ky]
            re.append(c.real)
            if (kx, ky) not in self._real_only:
                im.append(c.imag)
        return np.concatenate([np.array(re), np.array(im)]).astype(np.float32)

    def coeffs_to_field(self, vec, G):
        """Real coefficient vector -> field on a G x G grid (any G >= 2*kmax+1)."""
        nre = len(self.index)
        re = vec[:nre]
        im_vals = vec[nre:]
        R = np.zeros((G, G // 2 + 1), dtype=complex)
        it = iter(im_vals)
        for j, (kx, ky) in enumerate(self.index):
            val = re[j]
            if (kx, ky) not in self._real_only:
                val = val + 1j * next(it)
            R[kx % G, ky] = val * (G * G)      # de-normalise for this resolution
        return np.fft.irfft2(R, s=(G, G))

    def prior_std(self, sim, n=2000, seed=0):
        """Per-coefficient marginal prior std over the theta prior (for whitening)."""
        from .priors import sample_prior
        rng = np.random.default_rng(seed)
        theta = sample_prior(n, rng)
        C = np.zeros((n, self.dim), dtype=np.float64)
        for i in range(n):
            f = sim.sample_field(0.0, theta[i, 1], theta[i, 2], rng)
            C[i] = self.field_to_coeffs(f)
        return C.std(0).astype(np.float32) + 1e-8
