import numpy as np
import pandas as pd


MIN_HISTORY = 28


def eligible_ids(stats: pd.DataFrame) -> set:
    """The series every model is scored on, for one fold (see Step 6.3)."""
    ok = (stats["scale"] > 0) & (stats["n_hist"] >= MIN_HISTORY)
    return set(stats.loc[ok, "id"])


def wrmsse(pred: pd.DataFrame, stats: pd.DataFrame) -> float:
    """pred: id, d_num, y, yhat (one fold). stats: id, scale, weight, n_hist (same fold).
    Only eligible series are scored; weights are renormalised over them."""
    per = (pred.assign(se=(pred["y"] - pred["yhat"]) ** 2)
               .groupby("id", as_index=False)["se"].mean()
               .merge(stats[["id", "scale", "weight", "n_hist"]], on="id", how="inner"))
    per = per[per["id"].isin(eligible_ids(stats))].copy()
    if per.empty or per["weight"].sum() == 0:
        return 0.0
    per["rmsse"] = np.sqrt(per["se"] / per["scale"])
    return float((per["rmsse"] * per["weight"]).sum() / per["weight"].sum())


def wape(y, yhat, price=None) -> float:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    w = np.ones_like(y) if price is None else np.asarray(price, float)
    denom = np.sum(y * w)
    if denom == 0:
        return 0.0
    return float(np.sum(np.abs(y - yhat) * w) / denom)


def pinball(y, q_pred, q: float) -> float:
    d = np.asarray(y, float) - np.asarray(q_pred, float)
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


def coverage(y, lo, hi) -> float:
    y = np.asarray(y, float)
    return float(np.mean((y >= np.asarray(lo)) & (y <= np.asarray(hi))))
