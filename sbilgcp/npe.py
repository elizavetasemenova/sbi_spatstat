"""Amortized Neural Posterior Estimation for the LGCP.

Combines a summary/embedding network for the count grid with the conditional
MAF in :mod:`sbilgcp.flows`.  Also provides the two baselines used in the
ablation: a hand-crafted-summary NPE and a point-estimate CNN regressor.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from .flows import MAF
from . import lgcp


def preprocess_counts(y: torch.Tensor) -> torch.Tensor:
    """log1p transform + per-image standardisation, returned as ``[B,1,G,G]``."""
    x = torch.log1p(y)
    return x.unsqueeze(1)


class CNNEmbedding(nn.Module):
    """Convolutional embedding of a count grid into a context vector."""

    def __init__(self, embedding_dim=48):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.AvgPool2d(2),                                 # G -> G/2
            nn.Conv2d(32, 48, 3, padding=1), nn.ReLU(),
            nn.Conv2d(48, 48, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        # append two global count features (log total, log variance) explicitly
        self.head = nn.Sequential(
            nn.Linear(48 + 2, 96), nn.ReLU(),
            nn.Linear(96, embedding_dim),
        )
        self.embedding_dim = embedding_dim

    def forward(self, y):
        x = preprocess_counts(y)
        h = self.conv(x).flatten(1)
        flat = y.reshape(y.shape[0], -1)
        g = torch.stack([torch.log1p(flat.sum(1)), torch.log1p(flat.var(1))], dim=1)
        return self.head(torch.cat([h, g], dim=1))


class MLPEmbedding(nn.Module):
    """Encoder for hand-crafted summary vectors."""

    def __init__(self, in_dim, embedding_dim=48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 96), nn.ReLU(),
            nn.Linear(96, 96), nn.ReLU(),
            nn.Linear(96, embedding_dim),
        )
        self.embedding_dim = embedding_dim
        self.register_buffer("mu", torch.zeros(in_dim))
        self.register_buffer("sd", torch.ones(in_dim))

    def set_normalizer(self, feats):
        self.mu = feats.mean(0)
        self.sd = feats.std(0) + 1e-6

    def forward(self, feats):
        return self.net((feats - self.mu) / self.sd)


class NPE(nn.Module):
    """Embedding network + conditional MAF over unconstrained parameters."""

    def __init__(self, embedding: nn.Module, n_transforms=5, hidden=64, seed=0):
        super().__init__()
        self.embedding = embedding
        self.flow = MAF(
            dim=lgcp.N_PARAMS,
            context_dim=embedding.embedding_dim,
            n_transforms=n_transforms,
            hidden=hidden,
            seed=seed,
        )

    def log_prob(self, theta_u, x):
        return self.flow.log_prob(theta_u, self.embedding(x))

    @torch.no_grad()
    def sample(self, x, n_per=1000):
        return self.flow.sample(self.embedding(x), n_per=n_per)


@dataclass
class TrainConfig:
    epochs: int = 60
    batch_size: int = 256
    lr: float = 5e-4
    weight_decay: float = 1e-5
    val_frac: float = 0.1
    patience: int = 12
    seed: int = 0
    log_every: int = 5


@dataclass
class TrainReport:
    train_loss: list = field(default_factory=list)
    val_loss: list = field(default_factory=list)
    best_val: float = float("inf")
    wall_time: float = 0.0
    best_epoch: int = 0


def _split(n, val_frac, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = int(n * val_frac)
    return idx[n_val:], idx[:n_val]


def train_npe(
    model: NPE,
    theta: np.ndarray,
    x,                       # tensor of grids [N,G,G] or feats [N,D]
    cfg: TrainConfig,
    device="cpu",
    transform_fn=None,
) -> TrainReport:
    """Maximum-likelihood training of an NPE model.

    ``transform_fn`` maps natural parameters to the unconstrained space the flow
    models; defaults to the 3-parameter LGCP transform.
    """
    if transform_fn is None:
        transform_fn = lgcp.to_unconstrained
    torch.manual_seed(cfg.seed)
    model.to(device)
    theta_u = transform_fn(torch.as_tensor(theta, dtype=torch.float32)).to(device)
    x = torch.as_tensor(x, dtype=torch.float32).to(device)

    tr, va = _split(theta.shape[0], cfg.val_frac, cfg.seed)
    tr_t, va_t = torch.as_tensor(tr), torch.as_tensor(va)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)

    rep = TrainReport()
    best_state = None
    bad = 0
    t0 = time.time()
    for epoch in range(cfg.epochs):
        model.train()
        perm = tr_t[torch.randperm(len(tr_t))]
        tot = 0.0
        for s in range(0, len(perm), cfg.batch_size):
            b = perm[s : s + cfg.batch_size]
            opt.zero_grad()
            loss = -model.log_prob(theta_u[b], x[b]).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item() * len(b)
        train_loss = tot / len(perm)

        model.eval()
        with torch.no_grad():
            vtot = 0.0
            for s in range(0, len(va_t), 1024):
                b = va_t[s : s + 1024]
                vtot += -model.log_prob(theta_u[b], x[b]).sum().item()
            val_loss = vtot / len(va_t)
        sched.step(val_loss)
        rep.train_loss.append(train_loss)
        rep.val_loss.append(val_loss)

        if val_loss < rep.best_val - 1e-4:
            rep.best_val = val_loss
            rep.best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if cfg.log_every and epoch % cfg.log_every == 0:
            print(f"  epoch {epoch:3d}  train {train_loss:8.4f}  val {val_loss:8.4f}")
        if bad >= cfg.patience:
            print(f"  early stop at epoch {epoch}")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    rep.wall_time = time.time() - t0
    return rep


# ---------------------------------------------------------------------------
# Point-estimate regressor baseline (notebook-08 style, "SBI as regression").
# ---------------------------------------------------------------------------

class CNNRegressor(nn.Module):
    """CNN mapping a count grid to a point estimate of theta (unconstrained)."""

    def __init__(self):
        super().__init__()
        self.embed = CNNEmbedding(embedding_dim=48)
        self.head = nn.Sequential(nn.Linear(48, 64), nn.ReLU(), nn.Linear(64, lgcp.N_PARAMS))

    def forward(self, y):
        return self.head(self.embed(y))


def train_regressor(model, theta, y, cfg: TrainConfig, device="cpu") -> TrainReport:
    torch.manual_seed(cfg.seed)
    model.to(device)
    theta_u = lgcp.to_unconstrained(torch.as_tensor(theta, dtype=torch.float32)).to(device)
    y = torch.as_tensor(y, dtype=torch.float32).to(device)
    tr, va = _split(theta.shape[0], cfg.val_frac, cfg.seed)
    tr_t, va_t = torch.as_tensor(tr), torch.as_tensor(va)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()
    rep = TrainReport()
    best_state = None
    bad = 0
    t0 = time.time()
    for epoch in range(cfg.epochs):
        model.train()
        perm = tr_t[torch.randperm(len(tr_t))]
        tot = 0.0
        for s in range(0, len(perm), cfg.batch_size):
            b = perm[s : s + cfg.batch_size]
            opt.zero_grad()
            loss = loss_fn(model(y[b]), theta_u[b])
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        model.eval()
        with torch.no_grad():
            vl = loss_fn(model(y[va_t]), theta_u[va_t]).item()
        rep.train_loss.append(tot / len(perm))
        rep.val_loss.append(vl)
        if vl < rep.best_val - 1e-5:
            rep.best_val = vl
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= cfg.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    rep.wall_time = time.time() - t0
    return rep
