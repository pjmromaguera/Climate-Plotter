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


def precip_summary(df, slope):
    if df.empty:
        return dict(
            period="N/A",
            avg=float("nan"),
            annual_avg=float("nan"),
            max_label="N/A",
            max_value=float("nan"),
            slope=slope,
        )

    precip = df.dropna(subset=["rain_mm"]).reset_index(drop=True)

    if precip.empty:
        return dict(
            period="N/A",
            avg=float("nan"),
            annual_avg=float("nan"),
            max_label="N/A",
            max_value=float("nan"),
            slope=slope,
        )

    idx = precip["rain_mm"].idxmax()
    max_row = precip.iloc[idx]

    return dict(
        period=f"{precip['date'].min().year}-{precip['date'].max().year}",
        avg=precip["rain_mm"].mean(),
        annual_avg=precip["rain_mm"].sum() / precip["date"].dt.year.nunique(),
        max_label=pd.to_datetime(max_row["date"]).strftime("%B %Y"),
        max_value=max_row["rain_mm"],
        slope=slope,
    )


def build_precip_plot(
    df_plot: pd.DataFrame,
    ma_option: str,
    show_trend: bool,
    station_name: str,
    lat: str,
    lon: str,
) -> go.Figure:
    fig = go.Figure()

    title_text = station_name.upper()
    if lat and lon:
        title_text += f" ({lat},{lon})"

    fig.add_trace(
        go.Bar(
            x=df_plot["date"],
            y=df_plot["monthly_rain_mm"],
            name="Monthly Rainfall",
            marker=dict(
                color="rgba(88,166,255,0.65)",
                line=dict(color="rgba(88,166,255,0.85)", width=0.4),
            ),
            hovertemplate="<b>%{x|%Y-%m}</b><br>Monthly Rainfall: %{y:.1f} mm<extra></extra>",
        )
    )

    if ma_option != "None":
        ma_label = {
            "1-month": "1-Month Moving Average",
            "3-month": "3-Month Moving Average",
            "6-month": "6-Month Moving Average",
        }[ma_option]

        fig.add_trace(
            go.Scatter(
                x=df_plot["date"],
                y=df_plot["moving_avg"],
                mode="lines",
                name=ma_label,
                line=dict(color="#f0883e", width=1.8),
                hovertemplate="<b>%{x|%Y-%m}</b><br>Moving Average: %{y:.2f} mm<extra></extra>",
            )
        )

    if show_trend:
        fig.add_trace(
            go.Scatter(
                x=df_plot["date"],
                y=df_plot["trend"],
                mode="lines",
                name="Linear Trend",
                line=dict(color="#3fb950", width=1.8, dash="dash"),
                hovertemplate="<b>%{x|%Y-%m}</b><br>Trend: %{y:.2f} mm<extra></extra>",
            )
        )

    fig.update_layout(
        title=dict(
            text=title_text,
            x=0.01,
            xanchor="left",
            y=0.98,
            yanchor="top",
            font=dict(size=14, color="#f3f4f6", family="Arial Black"),
        ),
        height=480,
        paper_bgcolor="#161b22",
        plot_bgcolor="#161b22",
        font=dict(color="#e6edf3"),
        margin=dict(l=50, r=20, t=56, b=40),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.02,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="#aab6c8", size=10),
        ),
        hovermode="x unified",
        bargap=0.12,
    )

    fig.update_xaxes(
        title=None,
        showgrid=True,
        gridcolor="rgba(48,54,61,0.35)",
        linecolor="#30363d",
        tickfont=dict(color="#8b949e", size=10),
        tickformat="%Y",
        dtick="M48",
        zeroline=False,
    )

    fig.update_yaxes(
        title="Monthly Rainfall (mm)",
        showgrid=True,
        gridcolor="rgba(48,54,61,0.35)",
        linecolor="#30363d",
        tickfont=dict(color="#8b949e", size=10),
        title_font=dict(size=11, color="#c9d1d9"),
        zeroline=False,
    )

    return fig