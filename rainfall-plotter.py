import io
from datetime import date
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from scipy.stats import linregress, gumbel_r

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="NOAA GSOD Climate Time Series",
    layout="wide",
)

# ============================================================
# STYLES
# ============================================================
st.markdown(
    """
    <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(31,111,235,0.08), transparent 30%),
                radial-gradient(circle at top right, rgba(240,136,62,0.08), transparent 25%),
                #0d1117;
            color: #e6edf3;
        }

        .block-container {
            max-width: 1420px;
            padding-top: 1.2rem;
            padding-bottom: 2rem;
        }

        .eyebrow {
            font-size: 0.72rem;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            color: #58a6ff;
            font-family: monospace;
            margin-bottom: 0.35rem;
        }

        .page-title {
            font-size: 2.15rem;
            font-weight: 800;
            color: #f3f4f6;
            margin-bottom: 0.2rem;
            line-height: 1.1;
        }

        .subtitle {
            font-size: 0.92rem;
            color: #a8b3c7;
            font-family: monospace;
            margin-bottom: 1rem;
        }

        .hero-box {
            background: linear-gradient(180deg, rgba(22,27,34,0.92), rgba(17,22,29,0.92));
            border: 1px solid rgba(240,136,62,0.35);
            border-left: 4px solid #f0883e;
            border-radius: 14px;
            padding: 14px 16px;
            color: #b6c2d1;
            font-size: 0.90rem;
            line-height: 1.65;
            margin-bottom: 1rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.15);
        }

        .hero-box strong {
            color: #ff9d4d;
        }

        .metric-card {
            background:
                linear-gradient(180deg, rgba(17,24,34,0.96), rgba(14,19,27,0.96));
            border: 1px solid rgba(90,103,126,0.28);
            border-radius: 16px;
            padding: 16px 18px;
            min-height: 104px;
            box-shadow:
                inset 0 1px 0 rgba(255,255,255,0.03),
                0 10px 30px rgba(0,0,0,0.18);
        }

        .metric-label {
            font-size: 0.68rem;
            color: #8ea0bc;
            text-transform: uppercase;
            letter-spacing: 0.14em;
            font-family: monospace;
            margin-bottom: 0.55rem;
        }

        .metric-value {
            font-size: 1.55rem;
            font-weight: 800;
            font-family: monospace;
            color: #f3f4f6;
            line-height: 1.1;
        }

        .metric-sub {
            font-size: 0.80rem;
            color: #93a1b5;
            font-family: monospace;
            margin-top: 0.32rem;
        }

        .txt-blue   { color: #58a6ff; }
        .txt-green  { color: #3fb950; }
        .txt-red    { color: #f85149; }
        .txt-orange { color: #ff9d4d; }
        .txt-violet { color: #bc8cff; }

        .panel-wrap {
            background:
                linear-gradient(180deg, rgba(22,27,34,0.96), rgba(16,21,28,0.96));
            border: 1px solid rgba(90,103,126,0.25);
            border-radius: 16px;
            padding: 14px 14px 8px 14px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.16);
        }

        div[data-testid="stSidebar"] {
            background: #0b1016;
        }

        div[data-testid="stDataFrame"] {
            border-radius: 12px;
            overflow: hidden;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# CONSTANTS
# ============================================================
BASE_URL = "https://www.ncei.noaa.gov/data/global-summary-of-the-day/access"
STATIONS_FILE = Path(__file__).parent / "stations.csv"

PRCP_MISSING_FLAG = 99.99
TEMP_MISSING_FLAG = 9999.9

# ============================================================
# HELPERS
# ============================================================
def safe_text(val: object) -> str:
    if pd.isna(val):
        return ""
    return str(val).strip()


def fmt_coord(val) -> str:
    try:
        if pd.isna(val):
            return ""
        return f"{float(val):.3f}"
    except Exception:
        return safe_text(val)


@st.cache_data(show_spinner=False)
def load_station_csv() -> pd.DataFrame:
    if not STATIONS_FILE.exists():
        raise FileNotFoundError(f"Missing station file: {STATIONS_FILE}")

    df = pd.read_csv(STATIONS_FILE)

    if "ID" not in df.columns:
        raise ValueError("stations.csv must contain an 'ID' column.")

    df["ID"] = (
        df["ID"]
        .astype(str)
        .str.strip()
        .str.replace(".0", "", regex=False)
        .str.zfill(11)
    )

    if "NAME" not in df.columns:
        df["NAME"] = df["ID"]

    for col in ["NAME", "COUNTRY", "LATITUDE", "LONGITUDE"]:
        if col not in df.columns:
            df[col] = np.nan

    df["NAME"] = df["NAME"].astype(str).str.strip()
    df = df.sort_values("NAME").reset_index(drop=True)
    return df


def detect_date_column(df: pd.DataFrame) -> str:
    for c in ["DATE", "date"]:
        if c in df.columns:
            return c
    raise ValueError("No DATE column found in NOAA station CSV.")


def detect_required_weather_columns(df: pd.DataFrame) -> tuple[str, str]:
    prcp_candidates = ["PRCP", "prcp", "PRECIPITATION", "precipitation"]
    temp_candidates = ["TEMP", "temp", "TEMPERATURE", "temperature"]

    prcp_col = next((c for c in prcp_candidates if c in df.columns), None)
    temp_col = next((c for c in temp_candidates if c in df.columns), None)

    if prcp_col is None:
        raise ValueError("No precipitation column found.")
    if temp_col is None:
        raise ValueError("No temperature column found.")

    return prcp_col, temp_col


def make_station_year_url(year: int, station_id: str) -> str:
    return f"{BASE_URL}/{year}/{station_id}.csv"


@st.cache_data(show_spinner=False)
def fetch_station_year(year: int, station_id: str, timeout: int = 40) -> Optional[pd.DataFrame]:
    url = make_station_year_url(year, station_id)
    response = requests.get(url, timeout=timeout)

    if response.status_code == 404:
        return None
    response.raise_for_status()

    df = pd.read_csv(io.StringIO(response.text))
    if df.empty:
        return None
    return df


def merge_station_data(
    station_id: str,
    start_date: date,
    end_date: date,
    progress_bar=None,
    status_box=None,
) -> pd.DataFrame:
    years = list(range(start_date.year, end_date.year + 1))
    parts: List[pd.DataFrame] = []

    for i, year in enumerate(years, start=1):
        if status_box is not None:
            status_box.info(f"Downloading {station_id} — {year}")

        part = fetch_station_year(year, station_id)
        if part is not None and not part.empty:
            parts.append(part)

        if progress_bar is not None:
            progress_bar.progress(i / len(years))

    if not parts:
        return pd.DataFrame()

    df = pd.concat(parts, ignore_index=True)

    date_col = detect_date_column(df)
    prcp_col, temp_col = detect_required_weather_columns(df)

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[prcp_col] = pd.to_numeric(df[prcp_col], errors="coerce")
    df[temp_col] = pd.to_numeric(df[temp_col], errors="coerce")

    df.loc[df[prcp_col] == PRCP_MISSING_FLAG, prcp_col] = np.nan
    df.loc[df[temp_col] == TEMP_MISSING_FLAG, temp_col] = np.nan

    df = df.dropna(subset=[date_col]).copy()
    df = df[
        (df[date_col] >= pd.Timestamp(start_date)) &
        (df[date_col] <= pd.Timestamp(end_date))
    ].copy()

    out = pd.DataFrame()
    out["date"] = df[date_col]
    out["prcp_in"] = df[prcp_col]
    out["temp_f"] = df[temp_col]

    # conversions retained in data processing, but not described in methodology text
    out["rain_mm"] = out["prcp_in"] * 25.4
    out["temp_c"] = (out["temp_f"] - 32.0) * 5.0 / 9.0

    if "STATION" in df.columns:
        out["station_id"] = df["STATION"].astype(str).str.strip()
    else:
        out["station_id"] = station_id

    return out.sort_values("date").reset_index(drop=True)


def prepare_monthly_precip(df_daily: pd.DataFrame) -> pd.DataFrame:
    if df_daily.empty:
        return pd.DataFrame()

    monthly = (
        df_daily.set_index("date")
        .resample("MS")
        .agg(monthly_rain_mm=("rain_mm", "sum"))
        .reset_index()
    )

    full_range = pd.date_range(
        start=monthly["date"].min(),
        end=monthly["date"].max(),
        freq="MS",
    )

    monthly = (
        monthly.set_index("date")
        .reindex(full_range)
        .rename_axis("date")
        .reset_index()
    )

    monthly["monthly_rain_mm"] = monthly["monthly_rain_mm"].fillna(0.0)
    monthly["year"] = monthly["date"].dt.year
    monthly["ym"] = monthly["date"].dt.strftime("%Y-%m")
    return monthly


def add_postprocessing(
    df: pd.DataFrame,
    value_col: str,
    ma_option: str,
    show_trend: bool,
    freq: str,
) -> tuple[pd.DataFrame, float]:
    out = df.copy().sort_values("date").reset_index(drop=True)
    out["plot_value"] = out[value_col]

    if ma_option == "None":
        out["moving_avg"] = np.nan
    else:
        if freq == "monthly":
            window_map = {"1-month": 1, "3-month": 3, "6-month": 6}
        else:
            window_map = {"1-month": 30, "3-month": 90, "6-month": 180}

        window = window_map[ma_option]
        min_periods = 1 if window == 1 else max(2, window // 2)
        out["moving_avg"] = out["plot_value"].rolling(window, min_periods=min_periods).mean()

    if show_trend:
        valid = out["plot_value"].notna()
        if valid.sum() >= 2:
            x = np.arange(len(out))[valid]
            y = out.loc[valid, "plot_value"].astype(float).values
            reg = linregress(x, y)
            out["trend"] = reg.intercept + reg.slope * np.arange(len(out))
            slope_per_year = reg.slope * (12.0 if freq == "monthly" else 365.25)
        else:
            out["trend"] = np.nan
            slope_per_year = np.nan
    else:
        out["trend"] = np.nan
        slope_per_year = np.nan

    return out, slope_per_year


def compute_precip_summary(df_plot: pd.DataFrame, slope_per_year: float) -> dict:
    if df_plot.empty:
        return {"period": "N/A", "avg": np.nan, "annual_avg": np.nan, "max_label": "N/A", "max_value": np.nan, "slope": np.nan}

    value_col = "monthly_rain_mm"
    avg = df_plot[value_col].mean()
    n_years = df_plot["date"].dt.year.nunique()
    annual_avg = df_plot[value_col].sum() / n_years if n_years > 0 else np.nan
    max_idx = df_plot[value_col].idxmax()
    max_row = df_plot.loc[max_idx]

    return {
        "period": f"{df_plot['date'].min().year} – {df_plot['date'].max().year}",
        "avg": avg,
        "annual_avg": annual_avg,
        "max_label": max_row["date"].strftime("%B %Y"),
        "max_value": max_row[value_col],
        "slope": slope_per_year,
    }


def compute_temp_summary(df_plot: pd.DataFrame, slope_per_year: float) -> dict:
    if df_plot.empty:
        return {"period": "N/A", "avg": np.nan, "series_avg": np.nan, "max_label": "N/A", "max_value": np.nan, "slope": np.nan}

    value_col = "temp_c"
    avg = df_plot[value_col].mean()
    max_idx = df_plot[value_col].idxmax()
    max_row = df_plot.loc[max_idx]

    return {
        "period": f"{df_plot['date'].min().year} – {df_plot['date'].max().year}",
        "avg": avg,
        "series_avg": avg,
        "max_label": max_row["date"].strftime("%d %b %Y"),
        "max_value": max_row[value_col],
        "slope": slope_per_year,
    }


def make_card(label: str, value: str, sub: str = "", value_class: str = ""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value {value_class}">{value}</div>
            {"<div class='metric-sub'>" + sub + "</div>" if sub else ""}
        </div>
        """,
        unsafe_allow_html=True,
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

    title_text = station_name
    if lat and lon:
        title_text += f"<br><sup>Lat {lat}, Lon {lon}</sup>"

    fig.add_trace(
        go.Bar(
            x=df_plot["date"],
            y=df_plot["monthly_rain_mm"],
            name="Monthly Rainfall",
            marker=dict(
                color="rgba(88,166,255,0.58)",
                line=dict(color="rgba(88,166,255,0.9)", width=0.6),
            ),
            hovertemplate="<b>%{x|%Y-%m}</b><br>Monthly Rainfall: %{y:.1f} mm<extra></extra>",
        )
    )

    if ma_option != "None":
        fig.add_trace(
            go.Scatter(
                x=df_plot["date"],
                y=df_plot["moving_avg"],
                mode="lines",
                name={
                    "1-month": "1-Month Moving Average",
                    "3-month": "3-Month Moving Average",
                    "6-month": "6-Month Moving Average",
                }[ma_option],
                line=dict(color="#f0883e", width=2.4),
                connectgaps=True,
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
                line=dict(color="#3fb950", width=2, dash="dash"),
                hovertemplate="<b>%{x|%Y-%m}</b><br>Trend: %{y:.2f} mm<extra></extra>",
            )
        )

    fig.update_layout(
        title=dict(text=title_text, x=0.01, xanchor="left", font=dict(size=20, color="#f3f4f6")),
        height=610,
        paper_bgcolor="#161b22",
        plot_bgcolor="#161b22",
        font=dict(color="#e6edf3"),
        margin=dict(l=30, r=20, t=80, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0.0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="#aab6c8"),
        ),
        hovermode="x unified",
    )

    fig.update_xaxes(
        title=None,
        showgrid=True,
        gridcolor="rgba(48,54,61,0.5)",
        linecolor="#30363d",
        tickfont=dict(color="#8b949e"),
        tickformat="%Y",
        dtick="M48",
    )

    fig.update_yaxes(
        title="Monthly Rainfall (mm)",
        showgrid=True,
        gridcolor="rgba(48,54,61,0.6)",
        linecolor="#30363d",
        tickfont=dict(color="#8b949e"),
    )

    return fig


def build_temp_plot(
    df_plot: pd.DataFrame,
    ma_option: str,
    show_trend: bool,
    station_name: str,
    lat: str,
    lon: str,
) -> go.Figure:
    fig = go.Figure()

    title_text = station_name
    if lat and lon:
        title_text += f"<br><sup>Lat {lat}, Lon {lon}</sup>"

    fig.add_trace(
        go.Scatter(
            x=df_plot["date"],
            y=df_plot["temp_c"],
            mode="lines",
            name="Daily Temperature",
            line=dict(color="#bc8cff", width=1.4),
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Daily Temperature: %{y:.2f} °C<extra></extra>",
        )
    )

    if ma_option != "None":
        fig.add_trace(
            go.Scatter(
                x=df_plot["date"],
                y=df_plot["moving_avg"],
                mode="lines",
                name={
                    "1-month": "1-Month Moving Average",
                    "3-month": "3-Month Moving Average",
                    "6-month": "6-Month Moving Average",
                }[ma_option],
                line=dict(color="#f0883e", width=2.4),
                connectgaps=True,
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
                line=dict(color="#3fb950", width=2, dash="dash"),
                hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Trend: %{y:.2f} °C<extra></extra>",
            )
        )

    fig.update_layout(
        title=dict(text=title_text, x=0.01, xanchor="left", font=dict(size=20, color="#f3f4f6")),
        height=610,
        paper_bgcolor="#161b22",
        plot_bgcolor="#161b22",
        font=dict(color="#e6edf3"),
        margin=dict(l=30, r=20, t=80, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0.0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="#aab6c8"),
        ),
        hovermode="x unified",
    )

    fig.update_xaxes(
        title=None,
        showgrid=True,
        gridcolor="rgba(48,54,61,0.5)",
        linecolor="#30363d",
        tickfont=dict(color="#8b949e"),
    )

    fig.update_yaxes(
        title="Daily Temperature (°C)",
        showgrid=True,
        gridcolor="rgba(48,54,61,0.6)",
        linecolor="#30363d",
        tickfont=dict(color="#8b949e"),
    )

    return fig


def compute_return_periods(df_daily: pd.DataFrame) -> pd.DataFrame:
    if df_daily.empty:
        return pd.DataFrame()

    annual_max = (
        df_daily.dropna(subset=["rain_mm"])
        .assign(year=lambda x: x["date"].dt.year)
        .groupby("year", as_index=False)["rain_mm"]
        .max()
        .rename(columns={"rain_mm": "annual_max_daily_rain_mm"})
    )

    if len(annual_max) < 3:
        return pd.DataFrame()

    vals = annual_max["annual_max_daily_rain_mm"].astype(float).values
    loc, scale = gumbel_r.fit(vals)

    rows = []
    for rp in [5, 25, 50, 100]:
        prob = 1 - (1 / rp)
        depth = gumbel_r.ppf(prob, loc=loc, scale=scale)
        rows.append(
            {
                "Return Period (years)": rp,
                "Estimated Daily Rainfall (mm)": round(float(depth), 2),
            }
        )

    result = pd.DataFrame(rows)
    result["Method"] = "Gumbel EV1 on annual maximum daily rainfall"
    result["Years Used"] = len(annual_max)
    return result


# ============================================================
# LOAD STATIONS
# ============================================================
try:
    stations_df = load_station_csv()
except Exception as e:
    st.error(f"Failed to load embedded stations.csv: {e}")
    st.stop()

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.markdown("## Station Selection")
station_name_options = stations_df["NAME"].sort_values().tolist()
selected_station_name = st.sidebar.selectbox("Choose station", station_name_options)

selected_row = stations_df.loc[stations_df["NAME"] == selected_station_name].iloc[0]
selected_station_id = selected_row["ID"]
selected_lat = fmt_coord(selected_row.get("LATITUDE", ""))
selected_lon = fmt_coord(selected_row.get("LONGITUDE", ""))

st.sidebar.markdown("## Date Range")
start_date = st.sidebar.date_input(
    "Start date",
    value=date(1973, 1, 1),
    min_value=date(1900, 1, 1),
    max_value=date(2100, 12, 31),
)

end_date = st.sidebar.date_input(
    "End date",
    value=date(2025, 12, 31),
    min_value=date(1900, 1, 1),
    max_value=date(2100, 12, 31),
)

st.sidebar.markdown("## Display Options")
ma_option = st.sidebar.selectbox(
    "Moving average",
    ["None", "1-month", "3-month", "6-month"],
    index=2,
)

show_trend = st.sidebar.checkbox(
    "Show trendline",
    value=True,
)

compute_rp = st.sidebar.checkbox(
    "Compute rainfall return periods",
    value=False,
    help="Based on annual maximum daily rainfall using a Gumbel EV1 fit.",
)

if start_date > end_date:
    st.sidebar.error("Start date must be earlier than end date.")
    st.stop()

run = st.sidebar.button("Process station", type="primary", use_container_width=True)

# ============================================================
# HEADER
# ============================================================
st.markdown('<div class="eyebrow">Illustrative Analysis Only</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="page-title">NOAA GSOD — Climate Time Series</div>',
    unsafe_allow_html=True,
)
subtitle = f"Selected station: {selected_station_name} · ID: {selected_station_id}"
if selected_lat and selected_lon:
    subtitle += f" · ({selected_lat}, {selected_lon})"
st.markdown(f'<div class="subtitle">{subtitle}</div>', unsafe_allow_html=True)

st.markdown(
    """
    <div class="hero-box">
        <strong>Mini-Methodology</strong><br>
        1. Daily station records are retrieved for the selected period from the NOAA Global Summary of the Day archive.<br>
        2. The workflow processes <strong>both precipitation and temperature</strong> for the selected station and period.<br>
        3. For <strong>precipitation</strong>, daily rainfall is aggregated into <strong>monthly accumulated totals</strong>.<br>
        4. For <strong>temperature</strong>, the plotting series uses the <strong>daily temperature record</strong>.<br>
        5. Moving averages and trendlines are optional <strong>display layers</strong> applied after the base series is prepared.<br>
        6. Optional rainfall return periods are estimated from the <strong>annual maximum daily rainfall</strong> series using a Gumbel EV1 fit.<br><br>
        <strong>Data source:</strong> NOAA Global Summary of the Day (GSOD) station archive.<br>
        <strong>Python packages used:</strong> streamlit, pandas, numpy, plotly, requests, scipy.<br><br>
        <strong>Disclaimer:</strong> This tool is not intended to replace expert analysis. It is provided for educational or research purposes only. Proper validation, quality control, and expert review are still needed before any operational, engineering, planning, or decision-making use.
    </div>
    """,
    unsafe_allow_html=True,
)

if not run:
    st.stop()

# ============================================================
# DOWNLOAD + PROCESS
# ============================================================
progress = st.progress(0.0)
status = st.empty()

try:
    daily_df = merge_station_data(
        station_id=selected_station_id,
        start_date=start_date,
        end_date=end_date,
        progress_bar=progress,
        status_box=status,
    )
except requests.HTTPError as e:
    st.error(f"HTTP error while downloading NOAA files: {e}")
    st.stop()
except Exception as e:
    st.error(f"Failed to process NOAA data: {e}")
    st.stop()
finally:
    progress.empty()
    status.empty()

if daily_df.empty:
    st.warning("No records found for the selected station and date range.")
    st.stop()

# Process both series upfront
precip_base = prepare_monthly_precip(daily_df)
temp_base = daily_df[["date", "temp_c"]].dropna().sort_values("date").reset_index(drop=True)

precip_plot_df, precip_slope = add_postprocessing(
    df=precip_base,
    value_col="monthly_rain_mm",
    ma_option=ma_option,
    show_trend=show_trend,
    freq="monthly",
)

temp_plot_df, temp_slope = add_postprocessing(
    df=temp_base,
    value_col="temp_c",
    ma_option=ma_option,
    show_trend=show_trend,
    freq="daily",
)

precip_summary = compute_precip_summary(precip_plot_df, precip_slope)
temp_summary = compute_temp_summary(temp_plot_df, temp_slope)

# ============================================================
# TABS
# ============================================================
tab_precip, tab_temp = st.tabs(["Precipitation", "Temperature"])

with tab_precip:
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        make_card("Period", precip_summary["period"], value_class="txt-orange")
    with c2:
        make_card("Monthly Avg", f"{precip_summary['avg']:.1f} mm", value_class="txt-blue")
    with c3:
        make_card("Annual Avg", f"{precip_summary['annual_avg']:.0f} mm/yr", value_class="txt-blue")
    with c4:
        make_card(
            "Max Month",
            precip_summary["max_label"],
            sub=f"{precip_summary['max_value']:.0f} mm",
            value_class="txt-blue",
        )
    with c5:
        if show_trend and pd.notna(precip_slope):
            trend_class = "txt-green" if precip_slope >= 0 else "txt-red"
            sign = "+" if precip_slope >= 0 else ""
            make_card("Trend", f"{sign}{precip_slope:.2f} mm/yr", value_class=trend_class)
        else:
            make_card("Trend", "Off", sub="Display layer disabled", value_class="txt-orange")

    st.markdown('<div class="panel-wrap">', unsafe_allow_html=True)
    st.plotly_chart(
        build_precip_plot(
            df_plot=precip_plot_df,
            ma_option=ma_option,
            show_trend=show_trend,
            station_name=selected_station_name,
            lat=selected_lat,
            lon=selected_lon,
        ),
        use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if compute_rp:
        st.markdown("### Rainfall Return Period Estimates")
        rp_df = compute_return_periods(daily_df)
        if rp_df.empty:
            st.warning("Not enough daily rainfall data to estimate return periods. At least a few years of valid annual maxima are needed.")
        else:
            st.dataframe(rp_df, use_container_width=True, hide_index=True)

with tab_temp:
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        make_card("Period", temp_summary["period"], value_class="txt-orange")
    with c2:
        make_card("Daily Avg", f"{temp_summary['avg']:.2f} °C", value_class="txt-violet")
    with c3:
        make_card("Series Avg", f"{temp_summary['series_avg']:.2f} °C", value_class="txt-violet")
    with c4:
        make_card(
            "Warmest Day",
            temp_summary["max_label"],
            sub=f"{temp_summary['max_value']:.2f} °C",
            value_class="txt-violet",
        )
    with c5:
        if show_trend and pd.notna(temp_slope):
            trend_class = "txt-green" if temp_slope >= 0 else "txt-red"
            sign = "+" if temp_slope >= 0 else ""
            make_card("Trend", f"{sign}{temp_slope:.2f} °C/yr", value_class=trend_class)
        else:
            make_card("Trend", "Off", sub="Display layer disabled", value_class="txt-orange")

    st.markdown('<div class="panel-wrap">', unsafe_allow_html=True)
    st.plotly_chart(
        build_temp_plot(
            df_plot=temp_plot_df,
            ma_option=ma_option,
            show_trend=show_trend,
            station_name=selected_station_name,
            lat=selected_lat,
            lon=selected_lon,
        ),
        use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# TABLES
# ============================================================
st.markdown("### Daily extracted data preview")
st.dataframe(
    daily_df.assign(date=daily_df["date"].dt.strftime("%Y-%m-%d")),
    use_container_width=True,
    hide_index=True,
)

left, right = st.columns([1.35, 1.0])

with left:
    st.markdown("### Processed series preview")
    preview_cols = st.tabs(["Precipitation Series", "Temperature Series"])

    with preview_cols[0]:
        temp_df = precip_plot_df.copy()
        temp_df["date"] = pd.to_datetime(temp_df["date"]).dt.strftime("%Y-%m-%d")
        st.dataframe(temp_df, use_container_width=True, hide_index=True)

    with preview_cols[1]:
        temp_df = temp_plot_df.copy()
        temp_df["date"] = pd.to_datetime(temp_df["date"]).dt.strftime("%Y-%m-%d")
        st.dataframe(temp_df, use_container_width=True, hide_index=True)

with right:
    st.markdown("### Station / processing info")
    info_df = pd.DataFrame(
        {
            "Field": [
                "Selected station",
                "Station ID",
                "Latitude",
                "Longitude",
                "Date range",
                "Daily records downloaded",
                "Precipitation series",
                "Temperature series",
                "Moving average",
                "Trendline",
            ],
            "Value": [
                selected_station_name,
                selected_station_id,
                selected_lat or "N/A",
                selected_lon or "N/A",
                f"{start_date} to {end_date}",
                f"{len(daily_df):,}",
                "Monthly accumulated rainfall",
                "Daily temperature",
                ma_option,
                "On" if show_trend else "Off",
            ],
        }
    )
    st.dataframe(info_df, use_container_width=True, hide_index=True)

# ============================================================
# DOWNLOADS
# ============================================================
st.markdown("### Downloads")
col_a, col_b, col_c = st.columns(3)

with col_a:
    st.download_button(
        "Download merged daily CSV",
        data=daily_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{selected_station_id}_{start_date}_{end_date}_daily_merged.csv",
        mime="text/csv",
        use_container_width=True,
    )

with col_b:
    st.download_button(
        "Download precipitation series CSV",
        data=precip_plot_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{selected_station_id}_{start_date}_{end_date}_precip_series.csv",
        mime="text/csv",
        use_container_width=True,
    )

with col_c:
    st.download_button(
        "Download temperature series CSV",
        data=temp_plot_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{selected_station_id}_{start_date}_{end_date}_temp_series.csv",
        mime="text/csv",
        use_container_width=True,
    )