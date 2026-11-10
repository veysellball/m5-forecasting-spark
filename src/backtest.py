"""Time series cross-validation and rolling-origin backtest split generators."""

import pandas as pd

from src.config import HORIZON


def split_fold(
    df: pd.DataFrame,
    cutoff: int,
    train_start: int = 0,
    horizon: int = HORIZON,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train = [train_start, cutoff], test = (cutoff, cutoff + horizon].

    Args:
        df: DataFrame containing a 'd_num' column.
        cutoff: The last day included in the training set.
        train_start: The first day included in the training set.
        horizon: Number of days in the forecasting horizon.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (train_df, test_df)
    """
    train = df[(df["d_num"] >= train_start) & (df["d_num"] <= cutoff)]
    test = df[(df["d_num"] > cutoff) & (df["d_num"] <= cutoff + horizon)]
    return train, test
