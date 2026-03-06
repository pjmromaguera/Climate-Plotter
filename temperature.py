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


def temp_summary(df,slope):

    idx=df.temp_c.idxmax()

    return dict(
        period=f"{df.date.min().year}-{df.date.max().year}",
        avg=df.temp_c.mean(),
        series_avg=df.temp_c.mean(),
        max_label=df.loc[idx,"date"].strftime("%d %b %Y"),
        max_value=df.loc[idx,"temp_c"],
        slope=slope
    )


def temp_plot(df,station,lat,lon,ma_option,show_trend):

    fig=go.Figure()

    fig.add_scatter(
        x=df.date,
        y=df.temp_c,
        name="Daily Temperature"
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
            line=dict(color="green",dash="dash")
        )

    fig.update_layout(
        title=f"{station} ({lat},{lon})",
        template="plotly_dark"
    )

    return fig