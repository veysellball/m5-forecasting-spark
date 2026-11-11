# Project 6: Validation Plan

> **Note:** This validation plan must be fixed and documented before any model training to prevent data leakage and cherry-picking.

## 1. Forecasting Question & Decision Framing
- **What is forecast:** Daily unit sales per `item_id` × `store_id` (30,490 time series in M5).
- **Horizon:** 28 days ahead.
- **Decision:** Store replenishment ordering and stock allocation.
- **Cost asymmetry:** Stockout costs (lost revenue, customer dissatisfaction) vs overstocking costs (perishable food waste, storage).
- **Decision target:** P50 (median estimate) for regular items, P90 for high-margin / low-spoilage items.

## 2. Cross-Validation & Split Strategy (Rolling Origin)
```
day ->   ... 1829 | 1830-1857 | 1858-1885 | 1886-1913 | 1914-1941
TUNE:  train <=1829 | validate |
F1:    train <=1857            |   test    |
F2:    train <=1885                        |   test    |
F3:    train <=1913                                    |  HOLDOUT  |   <- touched ONCE, at the end
```

- **Cutoffs:**
  - `TUNE`: 1829
  - `F1` (Dev): 1857
  - `F2` (Dev): 1885
  - `F3` (Holdout): 1913
- **Development Folds:** `F1`, `F2` (All model selection, feature engineering, and hyperparameter tuning happen here).
- **Holdout:** `F3` (Evaluated exactly once at the end; never used to choose models, features, or parameters).
- **Leakage-Free Feature Design:** All lag and rolling features use a lag of at least 28 days (`lags >= HORIZON`), enabling direct multi-step forecasting without recursive evaluation or data leakage.

## 3. Evaluation Metrics
- **Primary Metric:** **WRMSSE** (Weighted Root Mean Squared Scaled Error) at Level 12 (item × store). Weights correspond to the dollar sales share over the last 28 days of training.
- **Scale-Free Baseline:** **RMSSE** per series relative to the in-sample 1-step naive forecast.
- **Business Metric:** **WAPE** ($\sum |\text{error}| / \sum \text{actual}$), reported both in unit volume and dollar terms.
- **Probabilistic Metrics:** **Pinball loss** and **Empirical Coverage** for the 80% prediction interval ([P10, P90]).

## 4. Success Rule (Pre-registered, before any model training)
- The model **"beats the baseline"** if its WRMSSE is lower than seasonal-naive on **BOTH** dev folds (`F1`, `F2`), and on the holdout `F3` — reported once.
- Primary metric: **WRMSSE (level 12)**. Secondary: **Dollar-WAPE**.
- Intervals: The 80% interval's empirical coverage on `F3` must be within **75%–85%**.
- **Strict Leakage Rule:** No hyperparameter, feature engineering, or model architecture choice may be made by looking at `F3`.


