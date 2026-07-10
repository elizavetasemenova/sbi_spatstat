"""Gold-standard NUTS posterior for the LGCP via NumPyro.

We use a non-centred parameterisation of the latent Gaussian field on the same
grid used by the simulator, and place the *same* priors on (beta0, sigma, ell)
as the SBI method, so the two posteriors are directly comparable.  The Cholesky
factor of the Matern covariance is rebuilt at every leapfrog step because it
depends on the sampled length-scale -- this is precisely the per-dataset cost
that amortized SBI avoids.
"""

from __future__ import annotations

import time

import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

from . import lgcp


def _dist_matrix(coords):
    diff = coords[:, None, :] - coords[None, :, :]
    return jnp.sqrt(jnp.sum(diff ** 2, axis=-1) + 1e-12)


def _matern_corr(dist_m, ell, nu):
    d = dist_m / ell
    if nu == 0.5:
        return jnp.exp(-d)
    if nu == 1.5:
        s = jnp.sqrt(3.0) * d
        return (1.0 + s) * jnp.exp(-s)
    if nu == 2.5:
        s = jnp.sqrt(5.0) * d
        return (1.0 + s + s ** 2 / 3.0) * jnp.exp(-s)
    raise ValueError(nu)


def make_model(coords, area, nu=0.5, jitter=1e-5):
    dist_m = _dist_matrix(jnp.asarray(coords))
    n = coords.shape[0]
    eye = jnp.eye(n)

    def model(y):
        beta0 = numpyro.sample("beta0", dist.Normal(lgcp.BETA0_MEAN, lgcp.BETA0_SD))
        sigma = numpyro.sample("sigma", dist.Uniform(lgcp.SIGMA_LOW, lgcp.SIGMA_HIGH))
        ell = numpyro.sample("ell", dist.Uniform(lgcp.ELL_LOW, lgcp.ELL_HIGH))
        corr = _matern_corr(dist_m, ell, nu)
        cov = sigma ** 2 * corr + jitter * eye
        L = jnp.linalg.cholesky(cov)
        eta = numpyro.sample("eta", dist.Normal(0.0, 1.0).expand([n]))
        f = L @ eta
        log_rate = beta0 + f + jnp.log(area)
        numpyro.sample("y", dist.Poisson(jnp.exp(log_rate)), obs=y)

    return model


def run_nuts(
    y_grid,
    coords,
    area,
    nu=0.5,
    num_warmup=500,
    num_samples=500,
    num_chains=2,
    seed=0,
    progress=False,
):
    """Run NUTS for one count grid.  Returns dict of param samples + timing."""
    y = jnp.asarray(np.asarray(y_grid).reshape(-1))
    model = make_model(coords, area, nu=nu)
    kernel = NUTS(model, target_accept_prob=0.9, max_tree_depth=8)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method="sequential",
        progress_bar=progress,
    )
    t0 = time.time()
    mcmc.run(jax.random.PRNGKey(seed), y=y)
    wall = time.time() - t0
    samples = mcmc.get_samples()
    out = {k: np.asarray(samples[k]) for k in ("beta0", "sigma", "ell")}
    theta = np.stack([out["beta0"], out["sigma"], out["ell"]], axis=1)
    # split-Rhat for the three parameters of interest
    grouped = mcmc.get_samples(group_by_chain=True)
    rhat = {k: float(_split_rhat(np.asarray(grouped[k]))) for k in ("beta0", "sigma", "ell")}
    return {"theta": theta, "wall": wall, "rhat": rhat}


def _split_rhat(x):
    """Split-Rhat for array [chains, draws]."""
    m, n = x.shape
    if n < 4:
        return np.nan
    half = n // 2
    s = np.concatenate([x[:, :half], x[:, half:2 * half]], axis=0)
    m2, n2 = s.shape
    chain_means = s.mean(1)
    chain_vars = s.var(1, ddof=1)
    W = chain_vars.mean()
    B = n2 * chain_means.var(ddof=1)
    var_hat = (n2 - 1) / n2 * W + B / n2
    return np.sqrt(var_hat / (W + 1e-12))
