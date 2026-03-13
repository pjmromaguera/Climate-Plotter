import pandas as pd
import requests
import streamlit as st
from datetime import date

from data_loader import load_station_csv, merge_station_data, fmt_coord
from precipitation import (
    prepare_monthly_precip, compute_precip_summary,
    build_precip_plot, compute_return_periods,
    build_frequency_plot, export_precip_plot_html,
)
from temperature import (
    prepare_daily_temp, compute_temp_summary,
    build_temp_plot, export_temp_plot_html,
)
from ui_components import (
    apply_styles, render_header, render_methodology,
    make_card, build_station_selector_map, PH_PROVINCES,
)
import math

# ── Haversine distance (km) ───────────────────────────────────────────────────
def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def _fmt_dist(km):
    if km < 1.0:
        return f"{km*1000:.0f} m"
    return f"{km:.1f} km"

def _nearest_stations(stations_df, pin_lat, pin_lon, n=3):
    df = stations_df[stations_df["LATITUDE"].notna() & stations_df["LONGITUDE"].notna()].copy()
    df["_dist_km"] = df.apply(
        lambda r: _haversine_km(pin_lat, pin_lon, float(r["LATITUDE"]), float(r["LONGITUDE"])),
        axis=1,
    )
    return df.nsmallest(n, "_dist_km")[["NAME", "ID", "_dist_km"]].reset_index(drop=True)


@st.cache_data(show_spinner=False)
def _geocode(place: str) -> list[dict]:
    """Call Nominatim to resolve a place name → list of {name, lat, lon}."""
    try:
        resp = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": place, "format": "json", "limit": 5,
                    "countrycodes": "ph", "addressdetails": 0},
            headers={"User-Agent": "NOAA-GSOD-Dashboard/1.0"},
            timeout=5,
        )
        if resp.status_code == 200:
            return [
                {"name": r.get("display_name", ""), "lat": float(r["lat"]), "lon": float(r["lon"])}
                for r in resp.json()
            ]
    except Exception:
        pass
    return []

st.set_page_config(
    page_title="NOAA GSOD Climate Time Series",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_styles()

# ── Load stations ─────────────────────────────────────────────────────────────
try:
    stations_df = load_station_csv()
except Exception as e:
    st.error(f"Failed to load stations.csv: {e}")
    st.stop()

station_name_options = stations_df["NAME"].sort_values().tolist()

_today      = date.today()
_prev_month = (_today.replace(day=1) - pd.Timedelta(days=1)).replace(day=1)

# ── Session state ─────────────────────────────────────────────────────────────
if "map_selected_station" not in st.session_state:
    st.session_state["map_selected_station"] = station_name_options[0]
if "map_center" not in st.session_state:
    st.session_state["map_center"] = None
if "results_ready" not in st.session_state:
    st.session_state["results_ready"] = False
if "show_ma1" not in st.session_state:
    st.session_state["show_ma1"] = False
if "show_ma3" not in st.session_state:
    st.session_state["show_ma3"] = True
if "show_ma6" not in st.session_state:
    st.session_state["show_ma6"] = False
if "show_trend" not in st.session_state:
    st.session_state["show_trend"] = True
if "pin_mode" not in st.session_state:
    st.session_state["pin_mode"] = False
if "pin_location" not in st.session_state:
    st.session_state["pin_location"] = None   # (lat, lon) or None
if "geo_results" not in st.session_state:
    st.session_state["geo_results"] = []      # list of {name, lat, lon}
if "geo_query" not in st.session_state:
    st.session_state["geo_query"] = ""

selected_station_name = st.session_state["map_selected_station"]
selected_row          = stations_df.loc[stations_df["NAME"] == selected_station_name].iloc[0]
selected_station_id   = selected_row["ID"]
selected_lat          = fmt_coord(selected_row.get("LATITUDE",  ""))
selected_lon          = fmt_coord(selected_row.get("LONGITUDE", ""))

# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:

    # ── Mode toggle ───────────────────────────────────────────────────────
    mode_col1, mode_col2 = st.columns(2)
    with mode_col1:
        if st.button("🔍 Search", use_container_width=True,
                     type="primary" if not st.session_state["pin_mode"] else "secondary"):
            st.session_state["pin_mode"] = False
            st.rerun()
    with mode_col2:
        if st.button("📌 Pin location", use_container_width=True,
                     type="primary" if st.session_state["pin_mode"] else "secondary"):
            st.session_state["pin_mode"] = True
            st.rerun()

    pin_mode = st.session_state["pin_mode"]

    # ══════════════════════════════════════════════════════════════════════
    # SEARCH MODE — single smart bar
    # ══════════════════════════════════════════════════════════════════════
    if not pin_mode:
        search_query = st.text_input(
            "search",
            placeholder="🔍 City, province, station, or coords (10.123, 123.087)…",
            label_visibility="collapsed", key="search_query",
        )
        sq = search_query.strip()

        # ── Coordinate shortcut: "10.123, 123.087" ───────────────────────
        import re
        coord_match = re.match(
            r"^\s*(-?\d{1,3}(?:\.\d+)?)\s*,\s*(-?\d{1,3}(?:\.\d+)?)\s*$", sq
        )
        if coord_match:
            clat = float(coord_match.group(1))
            clon = float(coord_match.group(2))
            if st.button("📌 Go to coordinates", key="_goto_coords",
                         use_container_width=True, type="primary"):
                st.session_state["map_center"] = (clat, clon, 12)
                st.rerun()
        else:
            sql = sq.lower()
            province_hits, station_hits = [], []
            if sql:
                for place_key, coords in PH_PROVINCES.items():
                    if sql in place_key or place_key.startswith(sql):
                        province_hits.append((f"📍 {place_key.title()}", place_key, coords))
                station_hits = [n for n in station_name_options if sql in n.lower()]

            suggestion_labels = (
                [lbl for lbl, _, _ in province_hits[:5]] +
                [f"🌐 {n}" for n in station_hits[:25]]
            )
            if suggestion_labels:
                selected_suggestion = st.selectbox(
                    "Suggestions", suggestion_labels,
                    key="_unified_suggest", label_visibility="collapsed",
                )
                if st.button("✔ Go", key="_confirm_suggest",
                             use_container_width=True, type="primary"):
                    if selected_suggestion.startswith("📍 "):
                        for place_key, coords in PH_PROVINCES.items():
                            if place_key.title() == selected_suggestion[2:].strip():
                                st.session_state["map_center"] = coords
                                break
                    elif selected_suggestion.startswith("🌐 "):
                        sname = selected_suggestion[2:].strip()
                        st.session_state["map_selected_station"] = sname
                        row = stations_df.loc[stations_df["NAME"] == sname]
                        if not row.empty:
                            st.session_state["map_center"] = (
                                float(row.iloc[0]["LATITUDE"]),
                                float(row.iloc[0]["LONGITUDE"]), 11)
                        st.session_state["_trigger_process"] = True
                    st.rerun()
            elif sql:
                st.caption("*No matches — try a landmark name or enter coords like `14.651, 121.049`*")

        highlight_names = station_hits[:25] if (not coord_match and sql) else None
        pin_loc = st.session_state["pin_location"]

    # ══════════════════════════════════════════════════════════════════════
    # PIN MODE — single bar: place name OR coords, then click map
    # ══════════════════════════════════════════════════════════════════════
    else:
        pin_loc = st.session_state["pin_location"]

        # Single smart input
        gc1, gc2 = st.columns([7, 3])
        with gc1:
            pin_query = st.text_input(
                "pin_search",
                placeholder="Place, landmark, or lat, lon…",
                label_visibility="collapsed", key="_pin_query",
            )
        with gc2:
            pin_search_clicked = st.button("🔎 Find", key="_pin_search_btn",
                                           use_container_width=True, type="primary")

        import re
        pq = pin_query.strip()
        coord_m = re.match(
            r"^\s*(-?\d{1,3}(?:\.\d+)?)\s*,\s*(-?\d{1,3}(?:\.\d+)?)\s*$", pq
        )

        if pin_search_clicked and pq:
            if coord_m:
                clat, clon = float(coord_m.group(1)), float(coord_m.group(2))
                st.session_state["pin_location"] = (clat, clon)
                st.session_state["map_center"]   = (clat, clon, 13)
                st.session_state["geo_results"]  = []
                st.rerun()
            else:
                with st.spinner("Searching…"):
                    results = _geocode(pq)
                st.session_state["geo_results"] = results

        geo_results = st.session_state["geo_results"]
        if geo_results:
            for gi, gr in enumerate(geo_results[:5]):
                short = gr["name"].split(",")[0].strip()
                rc1, rc2 = st.columns([8, 2])
                with rc1:
                    st.markdown(
                        f'<div style="font-size:0.78rem;color:var(--txt-primary);padding:3px 0;">'
                        f'<strong>{short}</strong> '
                        f'<span style="font-size:0.68rem;color:var(--txt-muted);">'
                        f'{gr["lat"]:.4f}, {gr["lon"]:.4f}</span></div>',
                        unsafe_allow_html=True,
                    )
                with rc2:
                    if st.button("📌", key=f"_geo_pin_{gi}", use_container_width=True):
                        st.session_state["pin_location"] = (gr["lat"], gr["lon"])
                        st.session_state["map_center"]   = (gr["lat"], gr["lon"], 13)
                        st.session_state["geo_results"]  = []
                        st.rerun()

        # Instruction / active pin status
        if pin_loc is None:
            st.markdown(
                '<div class="hero-box" style="margin:6px 0;text-align:center;">'
                '👆 Search above <em>or</em> <strong>click anywhere on the map</strong> to drop a pin'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            plat, plon = pin_loc
            pi1, pi2 = st.columns([7, 3])
            with pi1:
                st.markdown(
                    f'<div class="hero-box" style="margin:6px 0;">'
                    f'📌 <strong>{plat:.5f}, {plon:.5f}</strong>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            with pi2:
                st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
                if st.button("✖ Clear", key="_clear_pin", use_container_width=True):
                    st.session_state["pin_location"] = None
                    st.session_state["geo_results"]  = []
                    st.rerun()

            # 3-column nearest-stations table
            nearest = _nearest_stations(stations_df, plat, plon, n=3)
            st.markdown(
                '<div class="ctrl-label" style="margin-top:8px;">NEAREST STATIONS</div>',
                unsafe_allow_html=True,
            )
            hc1, hc2, hc3 = st.columns([6, 3, 2])
            with hc1: st.markdown('<span style="font-size:0.62rem;color:var(--txt-muted);">STATION</span>', unsafe_allow_html=True)
            with hc2: st.markdown('<span style="font-size:0.62rem;color:var(--txt-muted);">DISTANCE</span>', unsafe_allow_html=True)
            with hc3: st.markdown("&nbsp;", unsafe_allow_html=True)

            for i, (_, row) in enumerate(nearest.iterrows()):
                dist_str = _fmt_dist(row["_dist_km"])
                is_sel   = row["NAME"] == st.session_state["map_selected_station"]
                nc1, nc2, nc3 = st.columns([6, 3, 2])
                with nc1:
                    s = "font-weight:700;color:var(--accent-orange2);" if is_sel else "color:var(--txt-primary);"
                    st.markdown(
                        f'<div style="font-size:0.80rem;{s}padding:6px 0;">'
                        f'{row["NAME"]}{"  ✓" if is_sel else ""}</div>',
                        unsafe_allow_html=True,
                    )
                with nc2:
                    st.markdown(
                        f'<div style="font-size:0.78rem;color:var(--txt-blue);'
                        f'font-family:monospace;padding:6px 0;">{dist_str}</div>',
                        unsafe_allow_html=True,
                    )
                with nc3:
                    if st.button("Load", key=f"_load_{i}_{row['NAME']}",
                                 use_container_width=True,
                                 type="primary" if is_sel else "secondary"):
                        st.session_state["map_selected_station"] = row["NAME"]
                        srow = stations_df.loc[stations_df["NAME"] == row["NAME"]].iloc[0]
                        st.session_state["map_center"] = (
                            float(srow["LATITUDE"]), float(srow["LONGITUDE"]), 12)
                        st.session_state["_trigger_process"] = True
                        st.rerun()

        highlight_names = None

    # ── Map ───────────────────────────────────────────────────────────────
    current_center = st.session_state.get("map_center", None)
    selector_fig = build_station_selector_map(
        stations_df, selected_station_name,
        highlight_names=highlight_names,
        map_center=current_center,
        pin_location=pin_loc if pin_mode else None,
        pin_mode=pin_mode,
    )
    event = st.plotly_chart(
        selector_fig, use_container_width=True,
        key="station_selector_map", on_select="rerun",
        config={"scrollZoom": True, "displayModeBar": False},
    )

    # ── Map click handler ─────────────────────────────────────────────────
    if event and event.get("selection") and event["selection"].get("points"):
        pts = event["selection"]["points"]
        if pts:
            pt = pts[0]
            raw = pt.get("customdata", "")
            clicked_val = (raw[0] if isinstance(raw, (list, tuple)) else str(raw)).strip()

            if pin_mode:
                # Grid point click → extract lat/lon from customdata tag
                if clicked_val.startswith("__grid__"):
                    parts = clicked_val.split("__")
                    try:
                        clat = float(parts[2]); clon = float(parts[3])
                        st.session_state["pin_location"] = (clat, clon)
                        st.session_state["map_center"]   = (clat, clon, 11)
                        st.session_state["geo_results"]  = []
                        st.rerun()
                    except Exception:
                        pass
                # Station marker click in pin mode → use station coords as pin
                elif clicked_val and clicked_val != "__pin__" and clicked_val in station_name_options:
                    srow = stations_df.loc[stations_df["NAME"] == clicked_val].iloc[0]
                    clat = float(srow["LATITUDE"]); clon = float(srow["LONGITUDE"])
                    st.session_state["pin_location"] = (clat, clon)
                    st.session_state["map_center"]   = (clat, clon, 11)
                    st.session_state["geo_results"]  = []
                    st.rerun()
            else:
                if clicked_val and clicked_val in station_name_options:
                    st.session_state["map_selected_station"] = clicked_val
                    row = stations_df.loc[stations_df["NAME"] == clicked_val]
                    if not row.empty:
                        st.session_state["map_center"] = (
                            float(row.iloc[0]["LATITUDE"]),
                            float(row.iloc[0]["LONGITUDE"]), 11)
                    st.session_state["_trigger_process"] = True
                    st.rerun()

    # ── Date range ────────────────────────────────────────────────────────
    st.markdown('<div class="ctrl-label" style="margin-top:6px;">DATE RANGE</div>',
                unsafe_allow_html=True)
    dc1, dc2 = st.columns(2)
    with dc1:
        start_date = st.date_input("Start", value=date(1973, 1, 1),
                                   min_value=date(1901, 1, 1), max_value=_prev_month,
                                   key="start_date", label_visibility="collapsed")
    with dc2:
        end_date = st.date_input("End", value=_today,
                                 min_value=date(1901, 2, 1), max_value=_today,
                                 key="end_date", label_visibility="collapsed")
    if start_date > end_date:
        st.error("Start date must be before end date.")
        st.stop()

# ── Resolve state after sidebar widgets render ────────────────────────────────
show_ma1   = st.session_state["show_ma1"]
show_ma3   = st.session_state["show_ma3"]
show_ma6   = st.session_state["show_ma6"]
show_trend = st.session_state["show_trend"]
start_date = st.session_state.get("start_date", date(1973, 1, 1))
end_date   = st.session_state.get("end_date",   _today)

# Re-resolve selected station (may have changed in sidebar)
selected_station_name = st.session_state["map_selected_station"]
selected_row          = stations_df.loc[stations_df["NAME"] == selected_station_name].iloc[0]
selected_station_id   = selected_row["ID"]
selected_lat          = fmt_coord(selected_row.get("LATITUDE",  ""))
selected_lon          = fmt_coord(selected_row.get("LONGITUDE", ""))

# ── Trigger data load ─────────────────────────────────────────────────────────
trigger = st.session_state.pop("_trigger_process", False)

if trigger:
    st.session_state["results_ready"] = True
    data_key = f"{selected_station_id}_{start_date}_{end_date}"
    if st.session_state.get("data_key") != data_key:
        with st.spinner(f"Downloading data for {selected_station_name}…"):
            try:
                st.session_state["daily_df"] = merge_station_data(
                    station_id=selected_station_id,
                    start_date=start_date, end_date=end_date,
                )
                st.session_state["data_key"] = data_key
            except requests.HTTPError as e:
                st.error(f"HTTP error: {e}")
                st.stop()
            except Exception as e:
                st.error(f"Data fetch failed: {e}")
                st.stop()

# ════════════════════════════════════════════════════════════════════════════
# MAIN AREA — Header + Results
# ════════════════════════════════════════════════════════════════════════════
render_header(selected_station_name, selected_station_id, selected_lat, selected_lon)

if not st.session_state["results_ready"]:
    st.markdown(
        "<div style='display:flex;align-items:center;justify-content:center;"
        "height:320px;color:#4a5568;font-size:0.9rem;'>"
        "← Click a station on the map to load data"
        "</div>",
        unsafe_allow_html=True,
    )
    st.stop()

daily_df = st.session_state.get("daily_df", pd.DataFrame())
if daily_df.empty:
    st.warning("No records found for the selected station and date range.")
    st.stop()

# ── Process series ────────────────────────────────────────────────────────────
precip_plot_df, precip_slope = prepare_monthly_precip(daily_df)
temp_plot_df,   temp_slope   = prepare_daily_temp(daily_df)
precip_summary = compute_precip_summary(precip_plot_df, precip_slope)
temp_summary   = compute_temp_summary(temp_plot_df,   temp_slope)

# ── Sub-tabs ──────────────────────────────────────────────────────────────────
res_precip, res_temp, res_freq = st.tabs(
    ["🌧 Precipitation", "🌡 Temperature", "📊 Frequency"]
)

# ── Precipitation ─────────────────────────────────────────────────────────────
with res_precip:
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: make_card("Period",     precip_summary["period"],                                value_class="txt-orange")
    with c2: make_card("Mo. Avg",    f"{precip_summary['avg']:.0f} mm",                      value_class="txt-blue")
    with c3: make_card("Annual Avg", f"{precip_summary['annual_avg']:.0f} mm",               value_class="txt-blue")
    with c4: make_card("Max Month",  precip_summary["max_label"],
                       sub=f"{precip_summary['max_value']:.0f} mm",                          value_class="txt-blue")
    with c5:
        if pd.notna(precip_slope):
            sign = "+" if precip_slope >= 0 else ""
            make_card("Trend", f"{sign}{precip_slope:.2f} mm/yr",
                      value_class="txt-green" if precip_slope >= 0 else "txt-red")
        else:
            make_card("Trend", "N/A", value_class="txt-orange")

    precip_fig = build_precip_plot(
        df_plot=precip_plot_df,
        show_ma1=show_ma1, show_ma3=show_ma3, show_ma6=show_ma6,
        show_trend=show_trend,
        station_name=selected_station_name, station_id=selected_station_id,
        lat=selected_lat, lon=selected_lon, summary=precip_summary,
        height=480,
    )
    st.markdown('<div class="panel-wrap">', unsafe_allow_html=True)
    st.plotly_chart(precip_fig, use_container_width=True,
        key=f"precip_{selected_station_id}_{start_date}_{end_date}")
    st.markdown("</div>", unsafe_allow_html=True)

    render_methodology()
    dc1, dc2 = st.columns(2)
    with dc1:
        st.download_button("⬇ CSV", precip_plot_df.to_csv(index=False).encode(),
            f"{selected_station_id}_{start_date}_{end_date}_precip.csv",
            "text/csv", use_container_width=True)
    with dc2:
        st.download_button("⬇ HTML plot", export_precip_plot_html(precip_fig),
            f"{selected_station_id}_{start_date}_{end_date}_precip.html",
            "text/html", use_container_width=True)

# ── Temperature ───────────────────────────────────────────────────────────────
with res_temp:
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: make_card("Period",      temp_summary["period"],                                value_class="txt-orange")
    with c2: make_card("Daily Avg",   f"{temp_summary['avg']:.1f} °C",                      value_class="txt-violet")
    with c3: make_card("Series Avg",  f"{temp_summary['series_avg']:.1f} °C",               value_class="txt-violet")
    with c4: make_card("Warmest Day", temp_summary["max_label"],
                       sub=f"{temp_summary['max_value']:.1f} °C",                           value_class="txt-violet")
    with c5:
        if pd.notna(temp_slope):
            sign = "+" if temp_slope >= 0 else ""
            make_card("Trend", f"{sign}{temp_slope:.3f} °C/yr",
                      value_class="txt-green" if temp_slope >= 0 else "txt-red")
        else:
            make_card("Trend", "N/A", value_class="txt-orange")

    temp_fig = build_temp_plot(
        df_plot=temp_plot_df,
        show_ma1=show_ma1, show_ma3=show_ma3, show_ma6=show_ma6,
        show_trend=show_trend,
        station_name=selected_station_name, station_id=selected_station_id,
        lat=selected_lat, lon=selected_lon, summary=temp_summary,
        height=480,
    )
    st.markdown('<div class="panel-wrap">', unsafe_allow_html=True)
    st.plotly_chart(temp_fig, use_container_width=True,
        key=f"temp_{selected_station_id}_{start_date}_{end_date}")
    st.markdown("</div>", unsafe_allow_html=True)

    render_methodology()
    dc1, dc2 = st.columns(2)
    with dc1:
        st.download_button("⬇ CSV", temp_plot_df.to_csv(index=False).encode(),
            f"{selected_station_id}_{start_date}_{end_date}_temp.csv",
            "text/csv", use_container_width=True)
    with dc2:
        st.download_button("⬇ HTML plot", export_temp_plot_html(temp_fig),
            f"{selected_station_id}_{start_date}_{end_date}_temp.html",
            "text/html", use_container_width=True)

# ── Frequency Analysis ────────────────────────────────────────────────────────
with res_freq:
    freq_df = compute_return_periods(daily_df)
    if freq_df.empty:
        st.warning("Not enough data for Gumbel fit (need ≥5 years).")
    else:
        freq_fig = build_frequency_plot(
            freq_df, selected_station_name, selected_station_id, height=460)
        st.markdown('<div class="panel-wrap">', unsafe_allow_html=True)
        st.plotly_chart(freq_fig, use_container_width=True,
            key=f"freq_{selected_station_id}_{start_date}_{end_date}")
        st.markdown("</div>", unsafe_allow_html=True)
        st.caption("Gumbel EV1 distribution fitted to annual maxima of 24-hr and 72-hr rainfall.")
        st.dataframe(freq_df.set_index("Return Period (years)"), use_container_width=True)
        st.download_button("⬇ CSV", freq_df.to_csv(index=False).encode(),
            f"{selected_station_id}_{start_date}_{end_date}_freq.csv", "text/csv")