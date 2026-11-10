from pathlib import Path

HORIZON = 28
SEED = 42

# cutoff = last training day; the test window is cutoff+1 ... cutoff+HORIZON
CUTOFFS = {"TUNE": 1829, "F1": 1857, "F2": 1885, "F3": 1913}
DEV_FOLDS = ["F1", "F2"]  # Model selection happens across these folds
HOLDOUT = "F3"  # Reported once at the end, never used for decision making

TRAIN_START_D = 1000  # Memory efficiency: train on d >= 1000 (~2.6 years)
LAGS = (28, 35, 42, 364)
ROLL_WINDOWS = (7, 28, 56)

# Quantiles for prediction intervals (P10, P50, P90)
QUANTILES = [0.1, 0.5, 0.9]

ROOT = Path(__file__).resolve().parents[1]
EXPORTS = ROOT / "data" / "exports"
