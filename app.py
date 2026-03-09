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

selected_station_name = st.session_state["map_selected_station"]
selected_row          = stations_df.loc[stations_df["NAME"] == selected_station_name].iloc[0]
selected_station_id   = selected_row["ID"]
selected_lat          = fmt_coord(selected_row.get("LATITUDE",  ""))
selected_lon          = fmt_coord(selected_row.get("LONGITUDE", ""))

# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Map + search + date range
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # ── Search bar ────────────────────────────────────────────────────────
    search_query = st.text_input(
        "search",
        placeholder="🔍 Station, province or city…",
        label_visibility="collapsed",
        key="search_query",
    )
    sq = search_query.strip().lower()

    # ── Build unified suggestion list as user types ───────────────────────
    province_hits = []   # list of (display_label, place_key, coords)
    station_hits  = []   # list of station name strings

    if sq:
        # Province/city matches — partial match both ways
        for place_key, coords in PH_PROVINCES.items():
            if sq in place_key or place_key.startswith(sq):
                province_hits.append((f"📍 {place_key.title()}", place_key, coords))
        # Station name matches
        station_hits = [n for n in station_name_options if sq in n.lower()]

    # Combine: provinces first (up to 5), then stations (up to 25)
    suggestion_labels = (
        [lbl for lbl, _, _ in province_hits[:5]] +
        [f"🌐 {n}" for n in station_hits[:25]]
    )

    selected_suggestion = None
    if suggestion_labels:
        selected_suggestion = st.selectbox(
            "Suggestions",
            suggestion_labels,
            key="_unified_suggest",
            label_visibility="collapsed",
        )

        # Confirm button
        if st.button("✔ Go", key="_confirm_suggest",
                     use_container_width=True, type="primary"):
            # Determine if province or station was chosen
            chosen_label = selected_suggestion
            if chosen_label.startswith("📍 "):
                # Province/city zoom
                chosen_place = chosen_label[2:].strip().lower()
                for place_key, coords in PH_PROVINCES.items():
                    if place_key == chosen_place or place_key.title() == chosen_label[2:].strip():
                        st.session_state["map_center"] = coords
                        break
            elif chosen_label.startswith("🌐 "):
                # Station select + zoom
                station_name_chosen = chosen_label[2:].strip()
                st.session_state["map_selected_station"] = station_name_chosen
                # Zoom map to that station
                row = stations_df.loc[stations_df["NAME"] == station_name_chosen]
                if not row.empty:
                    slat = float(row.iloc[0]["LATITUDE"])
                    slon = float(row.iloc[0]["LONGITUDE"])
                    st.session_state["map_center"] = (slat, slon, 11)
                st.session_state["_trigger_process"] = True
            st.rerun()
    elif sq:
        st.caption("*No matches found.*")

    # ── Map center logic ──────────────────────────────────────────────────
    current_center = st.session_state.get("map_center", None)

    # ── Map ───────────────────────────────────────────────────────────────
    # Highlight station hits in the map
    highlight_names = station_hits[:25] if station_hits else None
    selector_fig = build_station_selector_map(
        stations_df, selected_station_name,
        highlight_names=highlight_names,
        map_center=current_center,
    )
    event = st.plotly_chart(
        selector_fig, use_container_width=True,
        key="station_selector_map", on_select="rerun",
        config={"scrollZoom": True},
    )

    # Map click → auto process + zoom to clicked station
    if event and event.get("selection") and event["selection"].get("points"):
        pts = event["selection"]["points"]
        if pts:
            raw = pts[0].get("customdata", "")
            clicked_name = (raw[0] if isinstance(raw, (list, tuple)) else str(raw)).strip()
            if clicked_name and clicked_name in station_name_options:
                st.session_state["map_selected_station"] = clicked_name
                row = stations_df.loc[stations_df["NAME"] == clicked_name]
                if not row.empty:
                    slat = float(row.iloc[0]["LATITUDE"])
                    slon = float(row.iloc[0]["LONGITUDE"])
                    st.session_state["map_center"] = (slat, slon, 11)
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