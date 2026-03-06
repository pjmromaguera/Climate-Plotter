import pandas as pd
import numpy as np
from scipy.stats import linregress
import plotly.graph_objects as go


def prepare_temp_data(df,ma_option,show_trend):

    temp=df[["date","temp_c"]].dropna()

    if ma_option=="1-month":
        temp["ma"]=temp.temp_c.rolling(30).mean()

    if ma_option=="3-month":
        temp["ma"]=temp.temp_c.rolling(90).mean()

    if ma_option=="6-month":
        temp["ma"]=temp.temp_c.rolling(180).mean()

    x=np.arange(len(temp))

    slope=0

    if show_trend:

        r=linregress(x,temp.temp_c)

        temp["trend"]=r.intercept+r.slope*x

        slope=r.slope*365

    return temp,slope

def temp_summary(df, slope):
    if df.empty:
        return dict(
            period="N/A",
            avg=float("nan"),
            series_avg=float("nan"),
            max_label="N/A",
            max_value=float("nan"),
            slope=slope,
        )

    temp = df.dropna(subset=["temp_c"]).reset_index(drop=True)

    if temp.empty:
        return dict(
            period="N/A",
            avg=float("nan"),
            series_avg=float("nan"),
            max_label="N/A",
            max_value=float("nan"),
            slope=slope,
        )

    idx = temp["temp_c"].idxmax()
    max_row = temp.iloc[idx]

    return dict(
        period=f"{temp['date'].min().year}-{temp['date'].max().year}",
        avg=temp["temp_c"].mean(),
        series_avg=temp["temp_c"].mean(),
        max_label=pd.to_datetime(max_row["date"]).strftime("%d %b %Y"),
        max_value=max_row["temp_c"],
        slope=slope,
    )


def build_temp_plot(
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
        go.Scatter(
            x=df_plot["date"],
            y=df_plot["temp_c"],
            mode="lines",
            name="Daily Temperature",
            line=dict(color="rgba(99,110,250,0.9)", width=1.1),
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Daily Temperature: %{y:.2f} °C<extra></extra>",
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
                hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Moving Average: %{y:.2f} °C<extra></extra>",
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
                hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Trend: %{y:.2f} °C<extra></extra>",
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
    )

    fig.update_xaxes(
        title=None,
        showgrid=True,
        gridcolor="rgba(48,54,61,0.35)",
        linecolor="#30363d",
        tickfont=dict(color="#8b949e", size=10),
        zeroline=False,
    )

    fig.update_yaxes(
        title="Daily Temperature (°C)",
        showgrid=True,
        gridcolor="rgba(48,54,61,0.35)",
        linecolor="#30363d",
        tickfont=dict(color="#8b949e", size=10),
        title_font=dict(size=11, color="#c9d1d9"),
        zeroline=False,
    )

    return fig