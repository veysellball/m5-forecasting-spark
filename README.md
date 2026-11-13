# M5 Forecasting at Scale (PySpark & LightGBM)

![tests](https://github.com/veysellball/m5-forecasting-spark/actions/workflows/tests.yml/badge.svg)

> **Goal:** Forecast 28 days of daily unit sales for every item in every store of the Walmart M5 dataset (30,490 series), build feature engineering pipelines in PySpark on Databricks, test leakage-free lag generation, and produce calibrated prediction intervals for store replenishment.

## Architecture

- **Databricks (PySpark):** Distributed ingestion, reshaping wide sales to long format, feature engineering (lags, rolling stats), and distributed baseline models via `applyInPandas`.
- **Local Machine (LightGBM & DuckDB):** Backtesting on 3 rolling folds, quantile regression for prediction intervals (P10, P50, P90), SHAP explainability, and parity reconciliation with DuckDB.

## Repository Structure

```
Project6_Forecasting_Spark/
├── data/
│   ├── raw/                    # M5 CSV files (gitignored)
│   └── exports/                # Exported Parquet datasets (gitignored)
├── databricks/                 # Notebooks running on Databricks (Spark)
├── notebooks/                  # Local evaluation & modeling notebooks
├── src/                        # Modular pipeline code (tested)
│   ├── config.py
│   ├── features.py
│   ├── baselines.py
│   ├── metrics.py
│   └── backtest.py
├── tests/                      # Pytest suite (feature leakage & metric tests)
├── reports/                    # Validation plan and final forecasting decision report
├── .github/workflows/tests.yml # CI pipeline with Java 17 & PySpark
├── pyproject.toml
├── requirements.txt
└── requirements-dev.txt
```

