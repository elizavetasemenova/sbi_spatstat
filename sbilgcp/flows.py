"""A compact conditional Masked Autoregressive Flow (MAF).

The flow models the density of the (standardised, unconstrained) parameter
vector conditioned on a context embedding of the observed count grid.  It is
implemented from scratch so the paper's method has no hidden dependencies.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_degrees(in_features: int, hidden: int, n_layers: int, seed: int):
    """Degrees for the input, hidden and output units of a MADE block."""
    rng = np.random.default_rng(seed)
    degrees = [np.arange(1, in_features + 1)]
    for _ in range(n_layers):
        # hidden degrees in [min_prev .. in_features-1]
        min_prev = degrees[-1].min()
        degrees.append(rng.integers(min_prev, in_features, size=hidden))
    return degrees


class MaskedLinear(nn.Linear):
    def __init__(self, in_f, out_f, mask):
        super().__init__(in_f, out_f)
        self.register_buffer("mask", torch.as_tensor(mask, dtype=torch.float32))

    def forward(self, x):
        return F.linear(x, self.weight * self.mask, self.bias)


class ConditionalMADE(nn.Module):
    """Masked autoregressive density estimator with an external context.

    Outputs per-dimension shift ``mu`` and log-scale ``alpha`` such that, in the
    "data -> noise" direction, ``u_i = (x_i - mu_i) * exp(-alpha_i)``.
    """

    def __init__(self, dim, context_dim, hidden=64, n_layers=2, seed=0):
        super().__init__()
        self.dim = dim
        degrees = _make_degrees(dim, hidden, n_layers, seed)

        # input -> hidden (masked) plus context -> hidden (dense)
        self.blocks = nn.ModuleList()
        self.ctx = nn.ModuleList()
        sizes = [dim] + [hidden] * n_layers
        for l in range(n_layers):
            m = (degrees[l + 1][:, None] >= degrees[l][None, :]).astype(np.float32)
            self.blocks.append(MaskedLinear(sizes[l], sizes[l + 1], m))
            self.ctx.append(nn.Linear(context_dim, sizes[l + 1]))

        # hidden -> output (strict inequality), separate heads for mu and alpha
        out_mask = (degrees[0][:, None] > degrees[-1][None, :]).astype(np.float32)
        self.mu_out = MaskedLinear(hidden, dim, out_mask)
        self.alpha_out = MaskedLinear(hidden, dim, out_mask)

    def forward(self, x, context):
        h = x
        for lin, cl in zip(self.blocks, self.ctx):
            h = torch.relu(lin(h) + cl(context))
        mu = self.mu_out(h)
        alpha = torch.tanh(self.alpha_out(h))  # bounded log-scale for stability
        return mu, alpha


class MAF(nn.Module):
    """Stack of conditional MADE transforms with alternating input order."""

    def __init__(self, dim, context_dim, n_transforms=5, hidden=64, n_layers=2, seed=0):
        super().__init__()
        self.dim = dim
        self.transforms = nn.ModuleList(
            [
                ConditionalMADE(dim, context_dim, hidden, n_layers, seed + i)
                for i in range(n_transforms)
            ]
        )
        # alternate variable ordering between transforms
        perms = []
        base = np.arange(dim)
        for i in range(n_transforms):
            perms.append(base[::-1].copy() if i % 2 else base.copy())
        self.register_buffer("perms", torch.as_tensor(np.stack(perms)))

    def log_prob(self, x, context):
        """Log density of ``x`` (``[B, dim]``) given ``context``."""
        log_det = torch.zeros(x.shape[0], device=x.device)
        u = x
        for t, perm in zip(self.transforms, self.perms):
            u = u[:, perm]
            mu, alpha = t(u, context)
            u = (u - mu) * torch.exp(-alpha)
            log_det = log_det - alpha.sum(1)
        # base density N(0, I)
        base = -0.5 * (u ** 2).sum(1) - 0.5 * self.dim * np.log(2 * np.pi)
        return base + log_det

    @torch.no_grad()
    def sample(self, context, n_per=1):
        """Draw samples given a (possibly batched) context.

        ``context`` is ``[C, context_dim]``; returns ``[C, n_per, dim]``.
        """
        C = context.shape[0]
        ctx = context.repeat_interleave(n_per, dim=0)
        u = torch.randn(C * n_per, self.dim, device=context.device)
        # invert transforms in reverse order
        for t, perm in zip(reversed(list(self.transforms)), reversed(list(self.perms))):
            inv_perm = torch.argsort(perm)
            x = torch.zeros_like(u)
            # sequential autoregressive inversion
            for i in range(self.dim):
                mu, alpha = t(x, ctx)
                x[:, i] = u[:, i] * torch.exp(alpha[:, i]) + mu[:, i]
            u = x[:, inv_perm]
        return u.reshape(C, n_per, self.dim)
