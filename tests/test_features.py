import numpy as np
import pandas as pd
import pytest

from pyspark.sql import Row
from src.features import build_base, add_lag_features, add_rolling_features

N_DAYS = 450  # > 364 so lag_364 is defined somewhere


def _series(spark, ids=("A", "B"), seed=0, override=None):
    rng = np.random.default_rng(seed)
    rows = []
    for i in ids:
        for d in range(1, N_DAYS + 1):
            rows.append((i, d, int(rng.poisson(3))))
    pdf = pd.DataFrame(rows, columns=["id", "d_num", "sales"])
    if override is not None:
        pdf = override(pdf)
    return spark.createDataFrame(pdf)


def _features(df):
    return add_rolling_features(add_lag_features(df)).toPandas().sort_values(["id", "d_num"])


def test_build_base_smoke(spark):
    """Smoke test for build_base to ensure join and snap logic run cleanly."""
    sales_data = [
        Row(id="A", item_id="item1", dept_id="d1", cat_id="c1", store_id="CA_1", state_id="CA", sales=5, d_num=1),
        Row(id="B", item_id="item2", dept_id="d1", cat_id="c1", store_id="TX_1", state_id="TX", sales=0, d_num=1),
    ]
    cal_data = [
        Row(d_num=1, date="2011-01-29", wm_yr_wk=11101, event_name_1="", event_type_1="", snap_CA=1, snap_TX=0, snap_WI=0),
    ]
    prices_data = [
        Row(store_id="CA_1", item_id="item1", wm_yr_wk=11101, sell_price=9.99),
        Row(store_id="TX_1", item_id="item2", wm_yr_wk=11101, sell_price=4.50),
    ]

    sales_df = spark.createDataFrame(sales_data)
    cal_df = spark.createDataFrame(cal_data)
    prices_df = spark.createDataFrame(prices_data)

    base = build_base(sales_df, cal_df, prices_df)

    assert base.count() == 2
    assert "snap" in base.columns
    assert "snap_CA" not in base.columns


def test_features_do_not_see_the_last_27_days_or_the_future(spark):
    """Changing every sale on or after day t-27 must not change any feature at day t."""
    t = 420
    poison = lambda p: p.assign(sales=np.where(p["d_num"] > t - 28, 10**6, p["sales"]))
    clean = _features(_series(spark))
    dirty = _features(_series(spark, override=poison))
    feat_cols = [c for c in clean.columns if c.startswith(("lag_", "rmean_", "rstd_"))]
    a = clean[clean.d_num == t][feat_cols].reset_index(drop=True)
    b = dirty[dirty.d_num == t][feat_cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_lag_28_is_exactly_28_days_back(spark):
    out = _features(_series(spark, ids=("A",)))
    s = out.set_index("d_num")["sales"]
    for d in (29, 100, 400):
        assert out.set_index("d_num").loc[d, "lag_28"] == s.loc[d - 28]


def test_series_do_not_mix(spark):
    out = _features(_series(spark))
    b = out[out.id == "B"].set_index("d_num")
    assert pd.isna(b.loc[28, "lag_28"])                  # B has no day 0 — must not borrow from A
    assert b.loc[29, "lag_28"] == b.loc[1, "sales"]      # B's lag comes from B's own history


def test_rolling_is_null_until_window_is_full(spark):
    out = _features(_series(spark, ids=("A",))).set_index("d_num")
    # rmean_28 at day d uses days d-55 … d-28 → first full window is d = 56
    assert pd.isna(out.loc[55, "rmean_28"])
    assert not pd.isna(out.loc[56, "rmean_28"])


def test_short_lag_is_rejected(spark):
    with pytest.raises(ValueError):
        add_lag_features(_series(spark), lags=(7,))
