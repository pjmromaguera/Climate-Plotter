import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import linregress, gumbel_r

# ── Colour palette ────────────────────────────────────────────────────────────
BAR_FILL   = "rgba(88,166,255,0.50)"
BAR_LINE   = "rgba(88,166,255,0.80)"
MA1_LINE   = "#f0883e"
MA3_LINE   = "#d2a8ff"
MA6_LINE   = "#ffa657"
TREND_LINE = "#3fb950"
BG_COLOR   = "#161b22"
GRID_COLOR = "rgba(48,54,61,0.55)"
TICK_COLOR = "#8b949e"
FONT_COLOR = "#e6edf3"
ANN_COLOR  = "#aab6c8"


# ── Core preparation ──────────────────────────────────────────────────────────

def prepare_monthly_precip(df_daily: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    if df_daily.empty:
        return pd.DataFrame(), np.nan

    monthly = (
        df_daily.set_index("date")
        .resample("MS")
        .agg(monthly_rain_mm=("rain_mm", "sum"))
        .reset_index()
    )
    full_range = pd.date_range(start=monthly["date"].min(),
                               end=monthly["date"].max(), freq="MS")
    monthly = (monthly.set_index("date").reindex(full_range)
               .rename_axis("date").reset_index())
    monthly["monthly_rain_mm"] = monthly["monthly_rain_mm"].fillna(0.0)

    monthly["ma_1m"] = monthly["monthly_rain_mm"].rolling(1,  min_periods=1).mean()
    monthly["ma_3m"] = monthly["monthly_rain_mm"].rolling(3,  min_periods=2).mean()
    monthly["ma_6m"] = monthly["monthly_rain_mm"].rolling(6,  min_periods=3).mean()

    valid_mask = monthly["monthly_rain_mm"].notna()
    if valid_mask.sum() >= 2:
        x = np.arange(len(monthly))[valid_mask]
        y = monthly.loc[valid_mask, "monthly_rain_mm"].astype(float).values
        reg = linregress(x, y)
        monthly["trend"] = reg.intercept + reg.slope * np.arange(len(monthly))
        slope_per_year = reg.slope * 12.0
    else:
        monthly["trend"] = np.nan
        slope_per_year = np.nan

    monthly["year"] = monthly["date"].dt.year
    return monthly, slope_per_year


def compute_precip_summary(df_plot: pd.DataFrame, slope_per_year: float) -> dict:
    if df_plot.empty:
        return {"period": "N/A", "avg": np.nan, "annual_avg": np.nan,
                "max_label": "N/A", "max_value": np.nan, "slope": np.nan}
    col = "monthly_rain_mm"
    avg = df_plot[col].mean()
    n_years = df_plot["date"].dt.year.nunique()
    annual_avg = df_plot[col].sum() / n_years if n_years > 0 else np.nan
    max_idx = df_plot[col].idxmax()
    max_row = df_plot.loc[max_idx]
    return {
        "period":    f"{df_plot['date'].min().year} – {df_plot['date'].max().year}",
        "avg":        avg,
        "annual_avg": annual_avg,
        "max_label":  pd.to_datetime(max_row["date"]).strftime("%b %Y"),
        "max_value":  float(max_row[col]),
        "slope":      slope_per_year,
    }


# ── Plot ──────────────────────────────────────────────────────────────────────

def build_precip_plot(
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
    height: int = 420,
) -> go.Figure:

    fig = go.Figure()

    # Bar — monthly totals (trace 0)
    fig.add_trace(go.Bar(
        x=df_plot["date"], y=df_plot["monthly_rain_mm"],
        name="Monthly Rainfall",
        marker=dict(color=BAR_FILL, line=dict(color=BAR_LINE, width=0.5)),
        hovertemplate="<b>%{x|%Y-%m}</b><br>%{y:.1f} mm<extra></extra>",
    ))

    # MA traces — always added, visibility controlled by buttons
    ma_defs = [
        ("ma_1m", "1-Mo MA",  MA1_LINE, show_ma1),
        ("ma_3m", "3-Mo MA",  MA3_LINE, show_ma3),
        ("ma_6m", "6-Mo MA",  MA6_LINE, show_ma6),
    ]
    ma_trace_indices = []
    for col, label, color, visible in ma_defs:
        if col in df_plot.columns:
            ma_trace_indices.append(len(fig.data))
            fig.add_trace(go.Scatter(
                x=df_plot["date"], y=df_plot[col],
                mode="lines", name=label,
                line=dict(color=color, width=2.0),
                connectgaps=True, visible=visible,
                hovertemplate=f"<b>%{{x|%Y-%m}}</b><br>{label}: %{{y:.1f}} mm<extra></extra>",
            ))

    # Trendline (always added, visibility toggleable)
    trend_trace_idx = None
    if "trend" in df_plot.columns:
        trend_trace_idx = len(fig.data)
        fig.add_trace(go.Scatter(
            x=df_plot["date"], y=df_plot["trend"],
            mode="lines", name="Trend",
            line=dict(color=TREND_LINE, width=1.8, dash="dash"),
            visible=show_trend,
            hovertemplate="<b>%{x|%Y-%m}</b><br>Trend: %{y:.1f} mm<extra></extra>",
        ))

    subtitle_text = f"({lat}, {lon})" if lat and lon else ""
    summary_text = _precip_annotation(summary)

    _apply_dark_layout(
        fig,
        title=f"{station_name}",
        subtitle=subtitle_text,
        annotation=summary_text,
        yaxis_title="mm",
        height=height,
        n_traces=len(fig.data),
        ma_trace_indices=ma_trace_indices,
        ma_defs=ma_defs,
        trend_trace_idx=trend_trace_idx,
        trend_initially_on=show_trend,
    )
    fig.update_xaxes(tickformat="%Y", dtick="M48")
    return fig


def _precip_annotation(summary: dict | None) -> str:
    if not summary:
        return ""
    parts = [
        f"Period: {summary['period']}",
        f"Monthly Avg: {summary['avg']:.0f} mm",
        f"Annual Avg: {summary['annual_avg']:.0f} mm/yr",
    ]
    return "  ·  ".join(parts)


# ── Frequency analysis ────────────────────────────────────────────────────────

def compute_return_periods(df_daily: pd.DataFrame) -> pd.DataFrame:
    if df_daily.empty:
        return pd.DataFrame()

    base = (df_daily[["date", "rain_mm"]].dropna(subset=["rain_mm"])
            .sort_values("date").reset_index(drop=True).copy())

    def _annual_max(window_days: int) -> pd.Series:
        rolled = (base.set_index("date")["rain_mm"]
                  .rolling(f"{window_days}D", min_periods=1).sum())
        rolled.index = base["date"].values
        return rolled.groupby(rolled.index.year).max().dropna()

    def _fit(annual: pd.Series, return_periods):
        if len(annual) < 3:
            return [np.nan] * len(return_periods)
        loc, scale = gumbel_r.fit(annual)
        return [float(gumbel_r.ppf(1 - 1 / rp, loc=loc, scale=scale))
                for rp in return_periods]

    rps = [2, 5, 10, 25, 50, 100]
    ann_24 = _annual_max(1)
    ann_72 = _annual_max(3)

    if len(ann_24) < 3 and len(ann_72) < 3:
        return pd.DataFrame()

    rows_24 = _fit(ann_24, rps)
    rows_72 = _fit(ann_72, rps)

    return pd.DataFrame({
        "Return Period (years)": rps,
        "24-hr Rainfall (mm)":   [round(v, 1) if not np.isnan(v) else np.nan for v in rows_24],
        "72-hr Rainfall (mm)":   [round(v, 1) if not np.isnan(v) else np.nan for v in rows_72],
    })


def build_frequency_plot(
    freq_df: pd.DataFrame,
    station_name: str,
    station_id: str,
    height: int = 380,
) -> go.Figure:
    fig = go.Figure()
    rp_str = [str(r) for r in freq_df["Return Period (years)"]]
    colors = {"24-hr Rainfall (mm)": "#58a6ff", "72-hr Rainfall (mm)": "#d2a8ff"}
    for col, color in colors.items():
        if col in freq_df.columns:
            fig.add_trace(go.Scatter(
                x=rp_str, y=freq_df[col], mode="lines+markers", name=col,
                line=dict(color=color, width=2),
                marker=dict(size=7, color=color),
                hovertemplate=f"<b>%{{x}}-yr</b><br>{col}: %{{y:.1f}} mm<extra></extra>",
            ))
    fig.update_layout(
        title=dict(text=f"{station_name} — Frequency Analysis",
                   x=0.5, xanchor="center", font=dict(size=14, color=FONT_COLOR)),
        height=height,
        paper_bgcolor=BG_COLOR, plot_bgcolor=BG_COLOR,
        font=dict(color=FONT_COLOR, size=11),
        margin=dict(l=30, r=20, t=55, b=60),
        xaxis=dict(type="category", title="Return Period (years)",
                   tickfont=dict(color=TICK_COLOR), gridcolor=GRID_COLOR),
        yaxis=dict(title="Rainfall (mm)", tickfont=dict(color=TICK_COLOR),
                   gridcolor=GRID_COLOR),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="center", x=0.5, font=dict(color=ANN_COLOR, size=10)),
        hovermode="x unified",
    )
    return fig


def export_precip_plot_html(fig: go.Figure) -> bytes:
    return fig.to_html(include_plotlyjs="cdn").encode("utf-8")


# ── Shared layout helper ──────────────────────────────────────────────────────

def _apply_dark_layout(
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
    annotations = []
    if annotation:
        annotations.append(dict(
            text=annotation,
            x=0.5, y=-0.22, xref="paper", yref="paper",
            xanchor="center", yanchor="top", showarrow=False,
            font=dict(size=10, color=ANN_COLOR),
        ))

    # ── MA dropdown + Trendline toggle ───────────────────────────────────────
    updatemenus = []

    if ma_trace_indices and ma_defs and n_traces > 0:
        # Capture actual initial visibility of every trace from the figure
        def _actual_vis(i):
            v = fig.data[i].visible
            if v is None or v is True:
                return True
            return bool(v)

        base_vis = [_actual_vis(i) for i in range(n_traces)]

        # For each dropdown option, build a full per-trace visibility list.
        # MA indices get the specified show value; every other trace keeps its base.
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
        active_idx = pattern_map.get(active_pattern, 2)  # default 3-Mo

        updatemenus.append(dict(
            type="dropdown",
            x=0.0, xanchor="left",
            y=1.13, yanchor="top",
            showactive=True,
            active=active_idx,
            bgcolor="#1e2a3a", bordercolor="#3d5068", borderwidth=1,
            font=dict(color="#c9d1d9", size=10),
            buttons=[
                dict(label=lbl, method="restyle", args=[{"visible": vis}])
                for lbl, vis in options
            ],
        ))

    # Trendline toggle — full visibility lists, no dict shorthand
    if trend_trace_idx is not None and n_traces > 0:
        base_vis_trend = [_actual_vis(i) for i in range(n_traces)]
        vis_trend_on  = list(base_vis_trend); vis_trend_on[trend_trace_idx]  = True
        vis_trend_off = list(base_vis_trend); vis_trend_off[trend_trace_idx] = False
        updatemenus.append(dict(
            type="buttons", direction="right",
            x=0.42, xanchor="left",
            y=1.13, yanchor="top",
            showactive=True,
            active=0 if trend_initially_on else 1,
            bgcolor="#1e2a3a", bordercolor="#3d5068", borderwidth=1,
            font=dict(color="#c9d1d9", size=10),
            buttons=[
                dict(label="☑ Trend", method="restyle", args=[{"visible": vis_trend_on}]),
                dict(label="☐ Trend", method="restyle", args=[{"visible": vis_trend_off}]),
            ],
        ))

    fig.update_layout(
        title=dict(
            text=f"{title}<br><sup style='color:#8b949e'>{subtitle}</sup>" if subtitle else title,
            x=0.5, xanchor="center",
            font=dict(size=13, color=FONT_COLOR),
        ),
        height=height,
        paper_bgcolor=BG_COLOR, plot_bgcolor=BG_COLOR,
        font=dict(color=FONT_COLOR, size=10),
        margin=dict(l=40, r=15, t=80, b=75),
        legend=dict(
            orientation="h", yanchor="top", y=-0.14,
            xanchor="center", x=0.5,
            bgcolor="rgba(0,0,0,0)", font=dict(color=ANN_COLOR, size=9),
        ),
        hovermode="x unified",
        uirevision="station_plot",
        annotations=annotations,
        updatemenus=updatemenus,
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID_COLOR,
                     linecolor="#30363d", tickfont=dict(color=TICK_COLOR, size=9))
    fig.update_yaxes(title_text=yaxis_title, showgrid=True, gridcolor=GRID_COLOR,
                     linecolor="#30363d", tickfont=dict(color=TICK_COLOR, size=9),
                     title_font=dict(size=10))