import numpy as np
import pandas as pd

from src.metrics import coverage, eligible_ids, pinball, wape, wrmsse


def test_perfect_forecast_wrmsse():
    """A perfect forecast has WRMSSE = 0."""
    stats = pd.DataFrame({
        "id": ["item_1", "item_2"],
        "scale": [1.0, 2.0],
        "weight": [0.4, 0.6],
        "n_hist": [100, 100],
    })
    pred = pd.DataFrame({
        "id": ["item_1", "item_1", "item_2", "item_2"],
        "d_num": [1, 2, 1, 2],
        "y": [5.0, 3.0, 10.0, 12.0],
        "yhat": [5.0, 3.0, 10.0, 12.0],
    })
    score = wrmsse(pred, stats)
    assert np.isclose(score, 0.0)


def test_pinball_at_median_equals_half_mae():
    """pinball(q=0.5) is exactly half of the Mean Absolute Error (MAE)."""
    y = np.array([10.0, 20.0, 30.0, 40.0])
    q_pred = np.array([12.0, 18.0, 35.0, 38.0])
    
    mae = np.mean(np.abs(y - q_pred))
    p_loss = pinball(y, q_pred, q=0.5)
    
    assert np.isclose(p_loss, mae / 2.0)


def test_coverage_infinite_bounds():
    """Coverage of [-inf, +inf] must be exactly 1.0."""
    y = np.array([1.0, 2.0, 5.0, 10.0])
    lo = np.array([-np.inf, -np.inf, -np.inf, -np.inf])
    hi = np.array([np.inf, np.inf, np.inf, np.inf])
    
    assert coverage(y, lo, hi) == 1.0


def test_eligible_ids_filtering():
    """Series with scale <= 0 or n_hist < 28 are excluded."""
    stats = pd.DataFrame({
        "id": ["ok_1", "zero_scale", "short_hist"],
        "scale": [1.5, 0.0, 2.0],
        "n_hist": [30, 50, 10],
        "weight": [0.5, 0.25, 0.25],
    })
    valid = eligible_ids(stats)
    assert valid == {"ok_1"}


def test_wrmsse_excludes_ineligible_series():
    """Series with scale=0 or short history shouldn't crash wrmsse calculation."""
    stats = pd.DataFrame({
        "id": ["item_valid", "item_bad_scale", "item_short"],
        "scale": [2.0, 0.0, 1.5],
        "weight": [0.5, 0.3, 0.2],
        "n_hist": [100, 100, 10],
    })
    pred = pd.DataFrame({
        "id": ["item_valid", "item_bad_scale", "item_short"],
        "d_num": [1, 1, 1],
        "y": [10.0, 0.0, 5.0],
        "yhat": [8.0, 1.0, 2.0],
    })
    score = wrmsse(pred, stats)
    # item_valid se = (10-8)^2 = 4, scale = 2.0 -> rmsse = sqrt(4/2) = sqrt(2)
    # bad series are excluded, so weight is renormalised to 1.0
    expected = np.sqrt(4.0 / 2.0)
    assert np.isclose(score, expected)


def test_wape_calculation():
    y = np.array([10.0, 20.0])
    yhat = np.array([8.0, 25.0])
    # abs err = 2 + 5 = 7, total y = 30 -> 7/30
    assert np.isclose(wape(y, yhat), 7.0 / 30.0)
