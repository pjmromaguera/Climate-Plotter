import numpy as np
import pandas as pd
from scipy.stats import gumbel_r


def prepare_frequency_base(df_daily: pd.DataFrame) -> pd.DataFrame:
    if df_daily.empty:
        return pd.DataFrame()

    base = (
        df_daily[["date", "rain_mm"]]
        .dropna()
        .sort_values("date")
        .reset_index(drop=True)
        .copy()
    )

    base["rain_24h_mm"] = base["rain_mm"]
    base["rain_72h_mm"] = base["rain_mm"].rolling(3, min_periods=3).sum()

    # Daily GSOD cannot resolve true 36-hour rainfall exactly.
    base["rain_36h_note"] = "Exact 36-hour analysis requires sub-daily data."

    return base


def compute_return_period_table(series: pd.Series, label: str) -> pd.DataFrame:
    series = series.dropna()
    if len(series) < 3:
        return pd.DataFrame()

    vals = series.astype(float).values
    loc, scale = gumbel_r.fit(vals)

    rows = []
    for rp in [5, 25, 50, 100]:
        prob = 1 - (1 / rp)
        depth = gumbel_r.ppf(prob, loc=loc, scale=scale)
        rows.append(
            {
                "Duration": label,
                "Return Period (years)": rp,
                "Estimated Rainfall (mm)": round(float(depth), 2),
                "Method": "Gumbel EV1",
            }
        )
    return pd.DataFrame(rows)


def compute_frequency_analysis(df_daily: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    base = prepare_frequency_base(df_daily)
    if base.empty:
        return pd.DataFrame(), "No rainfall data available."

    annual_24h = (
        base.assign(year=base["date"].dt.year)
        .groupby("year", as_index=False)["rain_24h_mm"]
        .max()["rain_24h_mm"]
    )

    annual_72h = (
        base.assign(year=base["date"].dt.year)
        .groupby("year", as_index=False)["rain_72h_mm"]
        .max()["rain_72h_mm"]
    )

    out_24 = compute_return_period_table(annual_24h, "24-hour")
    out_72 = compute_return_period_table(annual_72h, "72-hour")

    out = pd.concat([out_24, out_72], ignore_index=True)

    note = (
        "24-hour and 72-hour rainfall frequency can be estimated from daily GSOD totals. "
        "Exact 36-hour frequency analysis is not supported with daily-resolution data and "
        "requires sub-daily observations."
    )
    return out, note