"""Amortized posterior over the whole log-intensity surface of an LGCP.

Parameter inference tells us how aggregated the pattern is; practitioners
usually also want the *map*: the latent log-intensity field
``Z(s) = beta0 + f(s)`` (equivalently the intensity ``Lambda = exp(Z)``) at every
location, with uncertainty.  This is a spatial-prediction task.

We amortize it with a U-Net that, in a single forward pass, outputs a structured
Gaussian posterior over the field,

    Z | y  ~  N( mu(y),  diag(d(y)^2) + V(y) V(y)^T ),

with a low-rank factor ``V`` (rank ``k``) that captures the *spatial correlation*
of the posterior.  The low-rank term is what makes aggregate functionals (total
abundance, regional counts) calibrated: a purely per-pixel (diagonal) posterior
is calibrated pointwise but badly underestimates the uncertainty of sums, whose
errors are correlated across cells.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field as _field

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# U-Net (small; for GxG grids with G a multiple of 4).
# ---------------------------------------------------------------------------

def _block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU(),
        nn.Conv2d(cout, cout, 3, padding=1), nn.ReLU(),
    )


class FieldUNet(nn.Module):
    """Outputs per-cell posterior mean, log-diagonal-std and ``k`` low-rank
    factor maps for the log-intensity field."""

    def __init__(self, rank=16, base=32):
        super().__init__()
        self.rank = rank
        self.enc1 = _block(1, base)
        self.enc2 = _block(base, base * 2)
        self.pool = nn.AvgPool2d(2)
        self.bott = _block(base * 2, base * 4)
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = _block(base * 4 + base * 2, base * 2)
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = _block(base * 2 + base, base)
        self.head_mu = nn.Conv2d(base, 1, 1)
        self.head_logd = nn.Conv2d(base, 1, 1)
        self.head_V = nn.Conv2d(base, rank, 1)

    def forward(self, y):
        # fixed affine normalisation -- preserves the absolute intensity level,
        # which the field posterior must recover (per-image centring would erase it)
        x = (torch.log1p(y).unsqueeze(1) - 2.0) / 2.0         # [B,1,G,G]
        s1 = self.enc1(x)                                     # G
        s2 = self.enc2(self.pool(s1))                         # G/2
        b = self.bott(self.pool(s2))                          # G/4
        d2 = self.dec2(torch.cat([self.up2(b), s2], 1))       # G/2
        d1 = self.dec1(torch.cat([self.up1(d2), s1], 1))      # G
        B = y.shape[0]; n = y.shape[1] * y.shape[2]
        # data-anchored mean: start from the pointwise estimate log((y+0.5)/area),
        # area = 1/n, and let the U-Net predict a (smoothing) correction.  This
        # gives the field its correct absolute scale without the net having to
        # learn the large -log(area) offset from scratch.
        anchor = torch.log(y + 0.5) + math.log(n)             # [B,G,G]
        mu = (anchor + self.head_mu(d1).squeeze(1)).reshape(B, n)
        logd = self.head_logd(d1).clamp(-6, 3)
        V = self.head_V(d1)
        d = torch.exp(logd).reshape(B, n)
        V = V.reshape(B, self.rank, n).transpose(1, 2)        # [B, n, k]
        return mu, d, V


# ---------------------------------------------------------------------------
# Structured low-rank + diagonal Gaussian: NLL, sampling, functional stats.
# All operations use the Woodbury identity / matrix-determinant lemma so the
# n x n covariance is never formed.
# ---------------------------------------------------------------------------

def _capacitance(d, V):
    """Return (M, Dinv_V) where M = I_k + V^T D^{-1} V  (D = diag(d^2))."""
    Dinv = 1.0 / (d ** 2)                                     # [B,n]
    DinvV = Dinv.unsqueeze(-1) * V                            # [B,n,k]
    k = V.shape[-1]
    M = torch.eye(k, device=d.device).unsqueeze(0) + V.transpose(1, 2) @ DinvV
    return M, DinvV, Dinv


def lowrank_gaussian_nll(z, mu, d, V):
    """Negative log density of ``z`` under N(mu, diag(d^2)+V V^T). ``z``:[B,n]."""
    B, n = z.shape
    M, DinvV, Dinv = _capacitance(d, V)
    r = z - mu
    Dinv_r = Dinv * r                                         # [B,n]
    # quadratic form via Woodbury:  r^T Sig^-1 r = r^T Dinv r - (V^T Dinv r)^T M^-1 (V^T Dinv r)
    Vt_Dinv_r = (DinvV * r.unsqueeze(-1)).sum(1)             # [B,k]
    L = torch.linalg.cholesky(M)
    sol = torch.cholesky_solve(Vt_Dinv_r.unsqueeze(-1), L).squeeze(-1)   # M^-1 (...)
    quad = (r * Dinv_r).sum(1) - (Vt_Dinv_r * sol).sum(1)
    # log|Sig| = sum log d^2 + log|M|
    logdet = torch.log(d ** 2).sum(1) + 2.0 * torch.log(torch.diagonal(L, dim1=1, dim2=2)).sum(1)
    return 0.5 * (quad + logdet + n * math.log(2 * math.pi))


@torch.no_grad()
def sample_field(mu, d, V, n_samp, rng=None):
    """Draw ``n_samp`` joint field samples: Z = mu + d*eps_n + V eps_k. -> [B,S,n]."""
    B, n = mu.shape
    k = V.shape[-1]
    g = torch.Generator(device=mu.device)
    if rng is not None:
        g.manual_seed(rng)
    en = torch.randn(B, n_samp, n, generator=g, device=mu.device)
    ek = torch.randn(B, n_samp, k, generator=g, device=mu.device)
    return mu[:, None, :] + d[:, None, :] * en + torch.einsum("bnk,bsk->bsn", V, ek)


# ---------------------------------------------------------------------------
# Training.
# ---------------------------------------------------------------------------

@dataclass
class FieldTrainReport:
    train: list = _field(default_factory=list)
    val: list = _field(default_factory=list)
    best_val: float = float("inf")
    wall: float = 0.0


def train_field(model, y, Z, epochs=60, batch=128, lr=5e-4, val_frac=0.1,
                patience=12, seed=0, device="cpu", log_every=5):
    """Train the field U-Net by structured-Gaussian maximum likelihood."""
    torch.manual_seed(seed)
    model.to(device)
    y = torch.as_tensor(y, dtype=torch.float32).to(device)
    Z = torch.as_tensor(Z, dtype=torch.float32).reshape(y.shape[0], -1).to(device)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(y.shape[0]); nv = int(val_frac * len(idx))
    va, tr = torch.as_tensor(idx[:nv]), torch.as_tensor(idx[nv:])
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=4)
    rep = FieldTrainReport(); best = None; bad = 0; t0 = time.time()
    for ep in range(epochs):
        model.train(); perm = tr[torch.randperm(len(tr))]; tot = 0.0
        for s in range(0, len(perm), batch):
            b = perm[s:s + batch]
            opt.zero_grad()
            mu, d, V = model(y[b])
            loss = lowrank_gaussian_nll(Z[b], mu, d, V).mean() / Z.shape[1]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); tot += loss.item() * len(b)
        model.eval()
        with torch.no_grad():
            vtot = 0.0
            for s in range(0, len(va), 256):
                b = va[s:s + 256]
                mu, d, V = model(y[b])
                vtot += (lowrank_gaussian_nll(Z[b], mu, d, V).sum() / Z.shape[1]).item()
            vl = vtot / len(va)
        sched.step(vl); rep.train.append(tot / len(perm)); rep.val.append(vl)
        if vl < rep.best_val - 1e-4:
            rep.best_val = vl; best = {k: v.detach().clone() for k, v in model.state_dict().items()}; bad = 0
        else:
            bad += 1
        if log_every and ep % log_every == 0:
            print(f"  epoch {ep:3d}  train {tot/len(perm):8.4f}  val {vl:8.4f}", flush=True)
        if bad >= patience:
            print(f"  early stop at epoch {ep}"); break
    if best is not None:
        model.load_state_dict(best)
    rep.wall = time.time() - t0
    return rep
