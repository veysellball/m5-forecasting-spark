import pandas as pd

HISTORY_DAYS = 730  # ETS on the last 2 years; older history adds cost, not accuracy

OUT_COLS = ["id", "fold", "d_num", "model", "yhat", "lo80", "hi80"]


def fit_baselines(pdf: pd.DataFrame, cutoffs: dict[str, int], horizon: int = 28) -> pd.DataFrame:
    """For ONE series: seasonal-naive, AutoETS (with 80% intervals) and Croston, for every fold."""
    from statsforecast import StatsForecast
    from statsforecast.models import AutoETS, CrostonOptimized, SeasonalNaive

    pdf = pdf.sort_values("d_num")
    sid = pdf["id"].iloc[0]
    out = []
    for fold, cutoff in cutoffs.items():
        hist = pdf[pdf["d_num"] <= cutoff].tail(HISTORY_DAYS)
        if len(hist) < 28:  # too new to forecast honestly
            continue
        sf_df = pd.DataFrame(
            {
                "unique_id": sid,
                "ds": hist["d_num"].to_numpy(),
                "y": hist["sales"].astype(float).to_numpy(),
            }
        )

        with_pi = StatsForecast(
            models=[SeasonalNaive(season_length=7), AutoETS(season_length=7)],
            freq=1,
            n_jobs=1,
        ).forecast(df=sf_df, h=horizon, level=[80])
        croston = StatsForecast(
            models=[CrostonOptimized()],
            freq=1,
            n_jobs=1,
        ).forecast(df=sf_df, h=horizon)

        fc = with_pi.reset_index() if "unique_id" not in with_pi.columns else with_pi
        cr = croston.reset_index() if "unique_id" not in croston.columns else croston

        for m in ("SeasonalNaive", "AutoETS"):
            out.append(
                pd.DataFrame(
                    {
                        "id": sid,
                        "fold": fold,
                        "d_num": fc["ds"].astype(int),
                        "model": m,
                        "yhat": fc[m].clip(lower=0),
                        "lo80": fc[f"{m}-lo-80"].clip(lower=0),
                        "hi80": fc[f"{m}-hi-80"],
                    }
                )
            )
        out.append(
            pd.DataFrame(
                {
                    "id": sid,
                    "fold": fold,
                    "d_num": cr["ds"].astype(int),
                    "model": "Croston",
                    "yhat": cr["CrostonOptimized"],
                    "lo80": float("nan"),
                    "hi80": float("nan"),
                }
            )
        )
    return pd.concat(out, ignore_index=True)[OUT_COLS] if out else pd.DataFrame(columns=OUT_COLS)
