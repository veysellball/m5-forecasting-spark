"""
PySpark feature engineering logic: lag features, rolling windows, calendar features.
All lags must be >= HORIZON_DAYS to prevent data leakage in multistep forecasting.
"""
