import numpy as np
import pandas as pd
import pytest

from src.baselines import OUT_COLS, fit_baselines
from src.config import CUTOFFS, HORIZON


def test_fit_baselines_single_series():
    """
    Test fit_baselines on a single synthetic series without Spark.
    Verifies:
      - 3 models x 4 folds x 28 days = 336 rows
      - Output schema matches OUT_COLS
      - Forecast horizon d_num is exactly cutoff + 1 ... cutoff + 28
      - Predictions and lower intervals are non-negative
    """
    # Create 800 days of synthetic sales for one series (>= 730 days of history needed)
    d_nums = np.arange(1200, 2000)
    sales = np.random.poisson(lam=2.0, size=len(d_nums))

    pdf = pd.DataFrame(
        {
            "id": "FOODS_1_001_CA_1",
            "d_num": d_nums,
            "sales": sales,
        }
    )

    result = fit_baselines(pdf, cutoffs=CUTOFFS, horizon=HORIZON)

    # 1. Output columns match expected schema
    assert list(result.columns) == OUT_COLS

    # 2. Number of rows: 3 models ('SeasonalNaive', 'AutoETS', 'Croston') x 4 folds x 28 days
    expected_models = {"SeasonalNaive", "AutoETS", "Croston"}
    assert set(result["model"].unique()) == expected_models
    assert set(result["fold"].unique()) == set(CUTOFFS.keys())
    assert len(result) == len(expected_models) * len(CUTOFFS) * HORIZON

    # 3. For each fold, d_num must run from cutoff + 1 to cutoff + 28
    for fold, cutoff in CUTOFFS.items():
        fold_df = result[result["fold"] == fold]
        expected_d_nums = set(range(cutoff + 1, cutoff + HORIZON + 1))
        for m in expected_models:
            model_fold_df = fold_df[fold_df["model"] == m]
            assert set(model_fold_df["d_num"].tolist()) == expected_d_nums

    # 4. Non-negative predictions (clipped)
    assert (result["yhat"] >= 0).all()
    # lo80 must be >= 0 (except Croston which produces NaN)
    ets_lo80 = result[result["model"].isin(["SeasonalNaive", "AutoETS"])]["lo80"]
    assert (ets_lo80 >= 0).all()


def test_fit_baselines_series_too_short():
    """If a series has fewer than 28 days of history before cutoff, it should skip."""
    pdf = pd.DataFrame(
        {
            "id": "NEW_ITEM_CA_1",
            "d_num": [1820, 1821, 1822],
            "sales": [1, 2, 0],
        }
    )
    result = fit_baselines(pdf, cutoffs={"TUNE": 1829}, horizon=28)
    assert len(result) == 0
    assert list(result.columns) == OUT_COLS
