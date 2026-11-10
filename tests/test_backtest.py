"""Unit tests for rolling-origin backtesting folds and splitting logic."""

import pandas as pd
import pytest

from src.backtest import split_fold
from src.config import CUTOFFS, HORIZON


def test_split_fold_boundaries():
    df = pd.DataFrame({"d_num": list(range(1, 2000)), "sales": [1] * 1999})
    cutoff = CUTOFFS["F1"]  # 1857
    train, test = split_fold(df, cutoff=cutoff, train_start=1000, horizon=HORIZON)

    # Train boundary checks
    assert train["d_num"].min() == 1000
    assert train["d_num"].max() == cutoff

    # Test boundary checks
    assert test["d_num"].min() == cutoff + 1
    assert test["d_num"].max() == cutoff + HORIZON
    assert len(test) == HORIZON

    # Ensure zero overlap between train and test windows (leakage-free)
    assert len(set(train["d_num"]).intersection(set(test["d_num"]))) == 0
