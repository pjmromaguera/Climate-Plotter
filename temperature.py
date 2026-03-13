import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import linregress

from theme import (
    TEMP_LINE_DARK,
    MA1_COLOR, MA3_COLOR, MA6_COLOR, TREND_COLOR,
    BG_PAPER, BG_PLOT,
    GRID_COLOR, TICK_COLOR, FONT_COLOR, ANN_COLOR,
    MENU_BG, MENU_BORDER, MENU_FONT,
)

_MA_WINDOWS = {"ma_1m": (30, 15), "ma_3m": (90, 45), "ma_6m": (180, 90)}


# ── Core preparation ──────────────────────────────────────────────────────────

def prepare_daily_temp(df_daily: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    temp = (df_daily[["date", "temp_c"]].dropna(subset=["temp_c"])
            .sort_values("date").reset_index(drop=True))
    if temp.empty:
        return temp, np.nan
    for col, (window, min_p) in _MA_WINDOWS.items():
        temp[col] = temp["temp_c"].rolling(window, min_periods=min_p).mean()
    valid_mask = temp["temp_c"].notna()
    if valid_mask.sum() >= 2:
        x = np.arange(len(temp))[valid_mask]
        y = temp.loc[valid_mask, "temp_c"].astype(float).values
        reg = linregress(x, y)
        temp["trend"] = reg.intercept + reg.slope * np.arange(len(temp))
        slope_per_year = reg.slope * 365.25
    else:
        temp["trend"] = np.nan
        slope_per_year = np.nan
    return temp, slope_per_year


def compute_temp_summary(df_plot: pd.DataFrame, slope_per_year: float) -> dict:
    if df_plot.empty:
        return {"period": "N/A", "avg": np.nan, "series_avg": np.nan,
                "max_label": "N/A", "max_value": np.nan, "slope": np.nan}
    avg = df_plot["temp_c"].mean()
    max_idx = df_plot["temp_c"].idxmax()
    max_row = df_plot.loc[max_idx]
    return {
        "period":     f"{df_plot['date'].min().year} – {df_plot['date'].max().year}",
        "avg":        avg,
        "series_avg": avg,
        "max_label":  pd.to_datetime(max_row["date"]).strftime("%d %b %Y"),
        "max_value":  float(max_row["temp_c"]),
        "slope":      slope_per_year,
    }


# ── Plot ──────────────────────────────────────────────────────────────────────

def build_temp_plot(
    df_plot: pd.DataFrame,
    show_ma1: bool,
    show_ma3: bool,
    show_ma6: bool,
    show_trend: bool,
    station_name: str,
    station_id: str,
    lat: str,
    lon: str,
    summary: dict | None = None,
    height: int = 480,
) -> go.Figure:

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_plot["date"], y=df_plot["temp_c"],
        mode="lines", name="Daily Temp",
        line=dict(color=TEMP_LINE_DARK, width=0.8),
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>%{y:.1f} °C<extra></extra>",
    ))

    ma_defs = [
        ("ma_1m", "1-Mo MA", MA1_COLOR, show_ma1),
        ("ma_3m", "3-Mo MA", MA3_COLOR, show_ma3),
        ("ma_6m", "6-Mo MA", MA6_COLOR, show_ma6),
    ]
    ma_trace_indices = []
    for col, label, color, visible in ma_defs:
        if col in df_plot.columns:
            ma_trace_indices.append(len(fig.data))
            fig.add_trace(go.Scatter(
                x=df_plot["date"], y=df_plot[col],
                mode="lines", name=label,
                line=dict(color=color, width=2.2),
                connectgaps=True, visible=visible,
                hovertemplate=f"<b>%{{x|%Y-%m-%d}}</b><br>{label}: %{{y:.1f}} °C<extra></extra>",
            ))

    trend_trace_idx = None
    if "trend" in df_plot.columns:
        trend_trace_idx = len(fig.data)
        fig.add_trace(go.Scatter(
            x=df_plot["date"], y=df_plot["trend"],
            mode="lines", name="Trend",
            line=dict(color=TREND_COLOR, width=1.8, dash="dash"),
            visible=show_trend,
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Trend: %{y:.1f} °C<extra></extra>",
        ))

    subtitle_text = f"({lat}, {lon})" if lat and lon else ""
    _apply_layout(
        fig,
        title=station_name,
        subtitle=subtitle_text,
        annotation=_temp_annotation(summary),
        yaxis_title="°C",
        height=height,
        n_traces=len(fig.data),
        ma_trace_indices=ma_trace_indices,
        ma_defs=ma_defs,
        trend_trace_idx=trend_trace_idx,
        trend_initially_on=show_trend,
    )
    return fig


def _temp_annotation(summary: dict | None) -> str:
    if not summary:
        return ""
    return (f"Period: {summary['period']}  ·  "
            f"Daily Avg: {summary['avg']:.1f} °C  ·  "
            f"Warmest: {summary['max_label']} ({summary['max_value']:.1f} °C)")


def export_temp_plot_html(fig: go.Figure) -> bytes:
    return fig.to_html(include_plotlyjs="cdn").encode("utf-8")


# ── Shared layout ─────────────────────────────────────────────────────────────

def _apply_layout(
    fig: go.Figure,
    title: str,
    subtitle: str,
    annotation: str,
    yaxis_title: str,
    height: int,
    n_traces: int = 0,
    ma_trace_indices: list | None = None,
    ma_defs: list | None = None,
    trend_trace_idx: int | None = None,
    trend_initially_on: bool = True,
) -> None:
    ann_list = []
    if annotation:
        ann_list.append(dict(
            text=annotation,
            x=0.5, y=-0.22, xref="paper", yref="paper",
            xanchor="center", yanchor="top", showarrow=False,
            font=dict(size=10, color=ANN_COLOR),
        ))

    updatemenus = []

    if ma_trace_indices and ma_defs and n_traces > 0:
        def _actual_vis(i):
            v = fig.data[i].visible
            return True if (v is None or v is True) else False

        base_vis = [_actual_vis(i) for i in range(n_traces)]

        def _ma_vis(show1, show3, show6):
            v = list(base_vis)
            for k, show in zip(ma_trace_indices, [show1, show3, show6]):
                v[k] = show
            return v

        active_pattern = tuple(bool(d[3]) for d in ma_defs)
        options = [
            ("None",    _ma_vis(False, False, False)),
            ("1-Mo MA", _ma_vis(True,  False, False)),
            ("3-Mo MA", _ma_vis(False, True,  False)),
            ("6-Mo MA", _ma_vis(False, False, True )),
            ("All MAs", _ma_vis(True,  True,  True )),
        ]
        pattern_map = {
            (False, False, False): 0,
            (True,  False, False): 1,
            (False, True,  False): 2,
            (False, False, True ): 3,
            (True,  True,  True ): 4,
        }
        active_idx = pattern_map.get(active_pattern, 2)

        updatemenus.append(dict(
            type="dropdown",
            x=0.0, xanchor="left", y=1.13, yanchor="top",
            showactive=True, active=active_idx,
            bgcolor=MENU_BG, bordercolor=MENU_BORDER, borderwidth=1,
            font=dict(color=MENU_FONT, size=10),
            buttons=[
                dict(label=lbl, method="restyle", args=[{"visible": vis}])
                for lbl, vis in options
            ],
        ))

    if trend_trace_idx is not None and n_traces > 0:
        base_vis_t = [_actual_vis(i) for i in range(n_traces)]
        vis_on  = list(base_vis_t); vis_on[trend_trace_idx]  = True
        vis_off = list(base_vis_t); vis_off[trend_trace_idx] = False
        updatemenus.append(dict(
            type="buttons", direction="right",
            x=0.42, xanchor="left", y=1.13, yanchor="top",
            showactive=True, active=0 if trend_initially_on else 1,
            bgcolor=MENU_BG, bordercolor=MENU_BORDER, borderwidth=1,
            font=dict(color=MENU_FONT, size=10),
            buttons=[
                dict(label="☑ Trend", method="restyle", args=[{"visible": vis_on}]),
                dict(label="☐ Trend", method="restyle", args=[{"visible": vis_off}]),
            ],
        ))

    title_text = (f"{title}<br><sup style='color:{ANN_COLOR}'>{subtitle}</sup>"
                  if subtitle else title)

    fig.update_layout(
        title=dict(text=title_text, x=0.5, xanchor="center",
                   font=dict(size=13, color=FONT_COLOR)),
        height=height,
        paper_bgcolor=BG_PAPER,
        plot_bgcolor=BG_PLOT,
        font=dict(color=FONT_COLOR, size=10),
        margin=dict(l=40, r=15, t=80, b=75),
        legend=dict(
            orientation="h", yanchor="top", y=-0.14,
            xanchor="center", x=0.5,
            bgcolor="rgba(0,0,0,0)", font=dict(color=ANN_COLOR, size=9),
        ),
        hovermode="x unified",
        uirevision="station_plot",
        annotations=ann_list,
        updatemenus=updatemenus,
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID_COLOR,
                     linecolor=GRID_COLOR, tickfont=dict(color=TICK_COLOR, size=9))
    fig.update_yaxes(title_text=yaxis_title, showgrid=True, gridcolor=GRID_COLOR,
                     linecolor=GRID_COLOR, tickfont=dict(color=TICK_COLOR, size=9),
                     title_font=dict(size=10, color=TICK_COLOR))