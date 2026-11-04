"""
Central configuration for M5 forecasting pipeline: horizon, paths, folds, random seed.
"""
HORIZON_DAYS = 28
N_FOLDS = 3
RANDOM_SEED = 42

# Primary evaluation metric parameters
QUANTILES = [0.1, 0.5, 0.9]
