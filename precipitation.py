import pandas as pd
import numpy as np
from scipy.stats import linregress
import plotly.graph_objects as go


def prepare_precip_data(df, ma_option, show_trend):

    monthly = (
        df.set_index("date")
        .resample("MS")["rain_mm"]
        .sum()
        .reset_index()
    )

    if ma_option=="3-month":
        monthly["ma"] = monthly["rain_mm"].rolling(3).mean()

    if ma_option=="6-month":
        monthly["ma"] = monthly["rain_mm"].rolling(6).mean()

    x=np.arange(len(monthly))

    slope=0

    if show_trend:
        r=linregress(x,monthly["rain_mm"])
        monthly["trend"]=r.intercept+r.slope*x
        slope=r.slope*12

    return monthly,slope


def precip_summary(df,slope):

    idx=df["rain_mm"].idxmax()

    return dict(
        period=f"{df.date.min().year}-{df.date.max().year}",
        avg=df["rain_mm"].mean(),
        annual_avg=df["rain_mm"].sum()/df.date.dt.year.nunique(),
        max_label=df.loc[idx,"date"].strftime("%B %Y"),
        max_value=df.loc[idx,"rain_mm"],
        slope=slope
    )


def precip_plot(df,station,lat,lon,ma_option,show_trend):

    fig=go.Figure()

    fig.add_bar(
        x=df.date,
        y=df.rain_mm,
        name="Monthly Rainfall"
    )

    if "ma" in df.columns:

        fig.add_scatter(
            x=df.date,
            y=df.ma,
            name="Moving Avg",
            line=dict(color="orange")
        )

    if show_trend:

        fig.add_scatter(
            x=df.date,
            y=df.trend,
            name="Trend",
            line=dict(dash="dash",color="green")
        )

    fig.update_layout(
        title=f"{station} ({lat},{lon})",
        template="plotly_dark"
    )

    return fig