# Project 6: Validation Plan

> **Note:** This validation plan must be fixed and documented before any model training to prevent data leakage and cherry-picking.

## 1. Forecasting Question & Decision Framing
- **What is forecast:** Daily unit sales per `item_id` × `store_id` (30,490 time series in M5).
- **Horizon:** 28 days ahead.
- **Decision:** Store replenishment ordering and stock allocation.
- **Cost asymmetry:** Stockout costs (lost revenue, customer dissatisfaction) vs overstocking costs (perishable food waste, storage).
- **Decision target:** P50 (median estimate) for regular items, P90 for high-margin / low-spoilage items.

## 2. Cross-Validation & Split Strategy
- **Method:** Rolling-origin backtest (Time-Series Cross Validation).
- **Folds:** 3 evaluation folds of 28 days each.
- **No Random Split:** Strictly respect chronological order; no future leakage.

## 3. Evaluation Metrics
- Primary: **RMSSE / WRMSSE** and **WAPE** (intermittent-safe; MAPE avoids division by zero).
- Probabilistic: **Pinball loss** and **Empirical Coverage** for prediction intervals [P10, P90].

## 4. Benchmark Baselines
- Naive (last observed value)
- Seasonal Naive (value from 7 days ago or 28 days ago)
- Moving Average (trailing window)
