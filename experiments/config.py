"""Shared configuration and paths for the LGCP-SBI experiments."""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
FIGDIR = os.path.join(ROOT, "paper", "figures")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIGDIR, exist_ok=True)

# Simulator / grid
GRID = 16
NU_TRAIN = 0.5           # exponential (Matern-1/2) generative kernel

# Data sizes
N_TRAIN = 60000
N_TEST = 3000            # held-out prior-predictive test set (SBC / coverage)
N_POSTERIOR = 1000       # posterior draws per test dataset from NPE

# MCMC benchmark
N_MCMC = 30              # number of test datasets given the gold-standard treatment
MCMC_WARMUP = 600
MCMC_SAMPLES = 500
MCMC_CHAINS = 2

# Misspecification study
NU_MISSPEC = (1.5, 2.5)
N_MISSPEC = 2000

# Reproducibility
SEED_TRAIN = 12345
SEED_TEST = 999
SEED_MCMC = 2024

# Files
F_TRAIN = os.path.join(RESULTS, "train.npz")
F_TEST = os.path.join(RESULTS, "test.npz")
F_MISSPEC = os.path.join(RESULTS, "misspec.npz")
F_NPE = os.path.join(RESULTS, "npe_cnn.pt")
F_NPE_HC = os.path.join(RESULTS, "npe_handcrafted.pt")
F_REG = os.path.join(RESULTS, "regressor.pt")
F_REPORTS = os.path.join(RESULTS, "train_reports.json")
F_CALIB = os.path.join(RESULTS, "calibration.json")
F_MCMC = os.path.join(RESULTS, "mcmc_samples.npz")
F_MCMC_CMP = os.path.join(RESULTS, "mcmc_comparison.json")
