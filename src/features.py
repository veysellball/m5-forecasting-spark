"""
PySpark feature engineering logic: lag features, rolling windows, calendar features.
All lags must be >= HORIZON_DAYS to prevent data leakage in multistep forecasting.
"""

from pyspark.sql import DataFrame, Window, functions as F


def build_base(sales_long: DataFrame, calendar: DataFrame, prices: DataFrame) -> DataFrame:
    """One row per (id, d_num) from the item's launch week on, with calendar and price attached.

    The inner join with prices drops pre-launch days: an item with no price that week
    was not on sale, so its zeros are structural, not demand.
    """
    cal = calendar.select(
        "d_num", "date", "wm_yr_wk", "event_name_1", "event_type_1",
        "snap_CA", "snap_TX", "snap_WI",
    )
    return (
        sales_long
        .join(F.broadcast(cal), "d_num", "inner")
        .join(prices, ["store_id", "item_id", "wm_yr_wk"], "inner")
        .withColumn(
            "snap",
            F.when(F.col("state_id") == "CA", F.col("snap_CA"))
             .when(F.col("state_id") == "TX", F.col("snap_TX"))
             .otherwise(F.col("snap_WI")),
        )
        .drop("snap_CA", "snap_TX", "snap_WI")
    )
