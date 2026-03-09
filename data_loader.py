import io
from datetime import date
from pathlib import Path
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
import streamlit as st

BASE_URL = "https://www.ncei.noaa.gov/data/global-summary-of-the-day/access"
STATIONS_FILE = Path(__file__).parent / "stations.csv"

PRCP_MISSING_FLAG = 99.99
TEMP_MISSING_FLAG = 9999.9

# Use a session per thread to avoid race conditions with a shared session
def _make_session() -> requests.Session:
    s = requests.Session()
    adapter = requests.adapters.HTTPAdapter(max_retries=2)
    s.mount("https://", adapter)
    return s


def safe_text(val: object) -> str:
    if pd.isna(val):
        return ""
    return str(val).strip()


def fmt_coord(val) -> str:
    try:
        if pd.isna(val):
            return ""
        return f"{float(val):.5f}"
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

    for col in ["NAME", "COUNTRY", "LATITUDE", "LONGITUDE"]:
        if col not in df.columns:
            df[col] = np.nan

    df["NAME"] = df["NAME"].fillna(df["ID"]).astype(str).str.strip()
    return df.sort_values("NAME").reset_index(drop=True)


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
        raise ValueError("No precipitation column found in NOAA data.")
    if temp_col is None:
        raise ValueError("No temperature column found in NOAA data.")

    return prcp_col, temp_col


def make_station_year_url(year: int, station_id: str) -> str:
    return f"{BASE_URL}/{year}/{station_id}.csv"


def _fetch_year(year: int, station_id: str, timeout: int = 40) -> Optional[pd.DataFrame]:
    """Fetch one year of station data. Not cached — called inside a thread pool."""
    url = make_station_year_url(year, station_id)
    session = _make_session()
    response = session.get(url, timeout=timeout)
    if response.status_code == 404:
        return None
    response.raise_for_status()
    df = pd.read_csv(io.StringIO(response.text))
    return None if df.empty else df


@st.cache_data(show_spinner=False)
def merge_station_data(
    station_id: str,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    years = list(range(start_date.year, end_date.year + 1))
    parts: List[pd.DataFrame] = []
    n_years = len(years)

    progress = st.progress(0, text="Downloading station data…")

    completed = 0
    with ThreadPoolExecutor(max_workers=min(8, n_years)) as executor:
        futures = {executor.submit(_fetch_year, year, station_id): year for year in years}
        for future in as_completed(futures):
            completed += 1
            progress.progress(completed / n_years, text=f"Downloaded {completed}/{n_years} years…")
            try:
                part = future.result()
            except Exception:
                # Skip years that fail (e.g. network hiccup); don't crash the whole load
                continue
            if part is not None and not part.empty:
                parts.append(part)

    progress.empty()

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

    # Deduplicate: keep first record per station/date if duplicates exist
    df = df.drop_duplicates(subset=[date_col], keep="first")

    out = pd.DataFrame()
    out["date"] = df[date_col].values
    out["prcp_in"] = df[prcp_col].values
    out["temp_f"] = df[temp_col].values
    out["rain_mm"] = out["prcp_in"] * 25.4
    out["temp_c"] = (out["temp_f"] - 32.0) * 5.0 / 9.0
    out["station_id"] = station_id

    return out.sort_values("date").reset_index(drop=True)