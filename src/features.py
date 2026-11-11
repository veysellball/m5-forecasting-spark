"""
PySpark feature engineering logic: lag features, rolling windows, calendar features.
All lags must be >= HORIZON_DAYS to prevent data leakage in multistep forecasting.
"""

from pyspark.sql import DataFrame, Window, functions as F

from src.config import HORIZON, LAGS, ROLL_WINDOWS


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


def add_lag_features(df: DataFrame, lags=LAGS, key: str = "id",
                     time: str = "d_num", target: str = "sales") -> DataFrame:
    """lag_k = target k days earlier. Requires contiguous days per series (checked in Step 3.5)."""
    w = Window.partitionBy(key).orderBy(time)
    for k in lags:
        if k < HORIZON:
            raise ValueError(f"lag {k} < horizon {HORIZON} would leak the future")
        df = df.withColumn(f"lag_{k}", F.lag(target, k).over(w))
    return df


def add_rolling_features(df: DataFrame, windows=ROLL_WINDOWS, shift: int = HORIZON,
                         key: str = "id", time: str = "d_num", target: str = "sales") -> DataFrame:
    """Rolling mean/std over the `wlen` days ending `shift` days before the target day.

    Frame for row d: rows d-(shift+wlen-1) … d-shift. Null until the window is full,
    so early rows don't get a mean of 3 days pretending to be a mean of 28.
    """
    if shift < HORIZON:
        raise ValueError(f"shift {shift} < horizon {HORIZON} would leak the future")
    for wlen in windows:
        w = (Window.partitionBy(key).orderBy(time)
                   .rowsBetween(-(shift + wlen - 1), -shift))
        full = F.count(target).over(w) == wlen
        df = df.withColumn(f"rmean_{wlen}", F.when(full, F.avg(target).over(w)))
        if wlen == 28:
            df = df.withColumn("rstd_28", F.when(full, F.stddev(target).over(w)))
    return df


def add_calendar_features(df: DataFrame) -> DataFrame:
    """Calendar features capturing weekly, monthly, and holiday event patterns."""
    return (
        df
        .withColumn("dow", F.dayofweek("date"))
        .withColumn("dom", F.dayofmonth("date"))
        .withColumn("month", F.month("date"))
        .withColumn("is_event", F.col("event_type_1").isNotNull().cast("int"))
    )


def add_price_features(df: DataFrame) -> DataFrame:
    """Price features reflecting promotions, momentum, and department-level relativity.

    Prices are planned in advance, so the price ON day d is a legitimate input.
    Future price statistics must never be leaked; price_norm uses only historical max.
    """
    upto_now = (
        Window.partitionBy("id").orderBy("d_num")
        .rowsBetween(Window.unboundedPreceding, 0)
    )
    week_dept = Window.partitionBy("store_id", "dept_id", "wm_yr_wk")
    by_item = Window.partitionBy("id").orderBy("d_num")

    return (
        df
        .withColumn("price_norm", F.col("sell_price") / F.max("sell_price").over(upto_now))
        .withColumn("price_change_7d", F.col("sell_price") / F.lag("sell_price", 7).over(by_item) - 1)
        .withColumn("price_rel_dept", F.col("sell_price") / F.avg("sell_price").over(week_dept))
    )


def build_features(base: DataFrame) -> DataFrame:
    """Full feature engineering pipeline composing lags, rolling windows, calendar, and price features."""
    df = add_lag_features(base)
    df = add_rolling_features(df)
    df = add_calendar_features(df)
    return add_price_features(df)
