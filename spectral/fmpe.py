"""Conditional flow-matching posterior estimator (FMPE) for the LGCP.

Target x = [theta_unit (3), whitened low-frequency field coefficients].  The base
measure is N(0, I), which in whitened coordinates is (approximately) the GP prior,
so the flow transports the prior to the posterior (a prior-aligned reference
measure).  We train a velocity field with the conditional flow-matching /
rectified-flow objective and sample the posterior by integrating the ODE.
"""

from __future__ import annotations

import time
import numpy as np
import torch
import torch.nn as nn


def timestep_embedding(t, dim=32):
    half = dim // 2
    freqs = torch.exp(-np.log(10000) * torch.arange(half, device=t.device) / half)
    a = t[:, None] * freqs[None]
    return torch.cat([torch.sin(a), torch.cos(a)], dim=1)


class FeatureEncoder(nn.Module):
    def __init__(self, in_dim, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU(),
            nn.Linear(256, out_dim),
        )
        self.register_buffer("mu", torch.zeros(in_dim))
        self.register_buffer("sd", torch.ones(in_dim))
        self.out_dim = out_dim

    def set_norm(self, feats):
        self.mu = feats.mean(0); self.sd = feats.std(0) + 1e-6

    def forward(self, phi):
        return self.net((phi - self.mu) / self.sd)


class VelocityNet(nn.Module):
    def __init__(self, dim, cond_dim, hidden=512, t_dim=32):
        super().__init__()
        self.t_dim = t_dim
        self.net = nn.Sequential(
            nn.Linear(dim + cond_dim + t_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x, t, cond):
        te = timestep_embedding(t, self.t_dim)
        return self.net(torch.cat([x, cond, te], dim=1))


class FMPE(nn.Module):
    def __init__(self, target_dim, feat_dim, cond_dim=128, hidden=512):
        super().__init__()
        self.encoder = FeatureEncoder(feat_dim, cond_dim)
        self.vnet = VelocityNet(target_dim, cond_dim, hidden)
        self.target_dim = target_dim

    def loss(self, x1, phi):
        B = x1.shape[0]
        x0 = torch.randn_like(x1)
        t = torch.rand(B, device=x1.device)
        xt = (1 - t)[:, None] * x0 + t[:, None] * x1
        v = self.vnet(xt, t, self.encoder(phi))
        return ((v - (x1 - x0)) ** 2).mean()

    @torch.no_grad()
    def sample(self, phi, n_per=1000, steps=100):
        """Return posterior samples [C, n_per, target_dim] by integrating the ODE."""
        C = phi.shape[0]
        cond = self.encoder(phi).repeat_interleave(n_per, 0)
        x = torch.randn(C * n_per, self.target_dim, device=phi.device)
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((x.shape[0],), i * dt, device=x.device)
            k1 = self.vnet(x, t, cond)
            k2 = self.vnet(x + dt * k1, torch.clamp(t + dt, max=1.0), cond)  # Heun
            x = x + 0.5 * dt * (k1 + k2)
        return x.reshape(C, n_per, self.target_dim)


def train_fmpe(model, X1, PHI, epochs=60, batch=256, lr=5e-4, val_frac=0.1,
               patience=12, seed=0, device="cpu", log_every=5):
    torch.manual_seed(seed)
    model.to(device)
    X1 = torch.as_tensor(X1, dtype=torch.float32).to(device)
    PHI = torch.as_tensor(PHI, dtype=torch.float32).to(device)
    model.encoder.set_norm(PHI)
    n = X1.shape[0]
    rng = np.random.default_rng(seed); idx = rng.permutation(n); nv = int(val_frac * n)
    va, tr = torch.as_tensor(idx[:nv]), torch.as_tensor(idx[nv:])
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=4)
    best = None; bad = 0; t0 = time.time(); hist = {"train": [], "val": []}
    for ep in range(epochs):
        model.train(); perm = tr[torch.randperm(len(tr))]; tot = 0.0
        for s in range(0, len(perm), batch):
            b = perm[s:s + batch]
            opt.zero_grad(); l = model.loss(X1[b], PHI[b]); l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            tot += l.item() * len(b)
        model.eval()
        with torch.no_grad():
            vl = np.mean([model.loss(X1[va[s:s+512]], PHI[va[s:s+512]]).item()
                          for s in range(0, len(va), 512)])
        sched.step(vl); hist["train"].append(tot/len(perm)); hist["val"].append(float(vl))
        if vl < (best if best is not None else 1e9) - 1e-4:
            best = vl; best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}; bad = 0
        else:
            bad += 1
        if log_every and ep % log_every == 0:
            print(f"  epoch {ep:3d}  train {tot/len(perm):.4f}  val {vl:.4f}", flush=True)
        if bad >= patience:
            print(f"  early stop {ep}"); break
    model.load_state_dict(best_state)
    hist["wall"] = time.time() - t0; hist["best_val"] = best
    return hist
