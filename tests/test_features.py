"""
Unit and leakage tests for PySpark feature engineering.
"""

from pyspark.sql import Row
from src.features import build_base


def test_build_base_smoke(spark):
    """Smoke test for build_base to ensure join and snap logic run cleanly."""
    sales_data = [
        Row(id="A", item_id="item1", dept_id="d1", cat_id="c1", store_id="CA_1", state_id="CA", sales=5, d_num=1),
        Row(id="B", item_id="item2", dept_id="d1", cat_id="c1", store_id="TX_1", state_id="TX", sales=0, d_num=1),
    ]
    cal_data = [
        Row(d_num=1, date="2011-01-29", wm_yr_wk=11101, event_name_1=None, event_type_1=None, snap_CA=1, snap_TX=0, snap_WI=0),
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
