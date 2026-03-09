import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def apply_styles():
    st.markdown("""
        <style>
            /* ── Sidebar styling ── */
            [data-testid="stSidebar"] {
                background: #0d1117 !important;
                border-right: 1px solid rgba(48,54,61,0.7) !important;
                min-width: 320px !important;
                max-width: 320px !important;
            }
            [data-testid="stSidebar"] > div:first-child {
                padding-top: 0.8rem;
                padding-left: 0.8rem;
                padding-right: 0.8rem;
            }
            /* Hide the sidebar collapse toggle arrow */
            [data-testid="collapsedControl"] { display: none !important; }

            .stApp {
                background:
                    radial-gradient(circle at top left,  rgba(31,111,235,0.07), transparent 30%),
                    radial-gradient(circle at top right, rgba(240,136,62,0.07),  transparent 25%),
                    #0d1117;
                color: #e6edf3;
            }
            .block-container { max-width: 100%; padding-top: 0.8rem; padding-bottom: 1.5rem; }

            .eyebrow { font-size:0.66rem; letter-spacing:0.16em; text-transform:uppercase;
                       color:#58a6ff; font-family:monospace; margin-bottom:0.2rem; }
            .page-title { font-size:1.6rem; font-weight:800; color:#f3f4f6;
                          margin-bottom:0.1rem; line-height:1.1; }
            .subtitle { font-size:0.78rem; color:#a8b3c7; font-family:monospace; margin-bottom:0.4rem; }

            .station-badge {
                background: linear-gradient(135deg,rgba(240,136,62,0.12),rgba(240,136,62,0.04));
                border: 1px solid rgba(240,136,62,0.38);
                border-radius: 10px; padding: 8px 14px; margin-bottom: 6px;
                display: flex; align-items: center; gap: 14px;
            }
            .station-badge-name { font-size:1.0rem; font-weight:700; color:#f0883e; }
            .station-badge-meta { font-size:0.7rem; color:#6b7a90; font-family:monospace; }

            .hero-box {
                background: linear-gradient(180deg,rgba(22,27,34,0.92),rgba(17,22,29,0.92));
                border:1px solid rgba(240,136,62,0.3); border-left:3px solid #f0883e;
                border-radius:10px; padding:10px 14px; color:#b6c2d1;
                font-size:0.72rem; line-height:1.5; margin:6px 0;
            }
            .hero-box strong { color:#ff9d4d; }

            .metric-card {
                background:linear-gradient(180deg,rgba(17,24,34,0.96),rgba(14,19,27,0.96));
                border:1px solid rgba(90,103,126,0.26); border-radius:12px;
                padding:10px 12px; min-height:78px;
                box-shadow:inset 0 1px 0 rgba(255,255,255,0.03),0 6px 18px rgba(0,0,0,0.16);
            }
            .metric-label { font-size:0.60rem; color:#8ea0bc; text-transform:uppercase;
                            letter-spacing:0.13em; font-family:monospace; margin-bottom:0.35rem; }
            .metric-value { font-size:1.2rem; font-weight:800; font-family:monospace;
                            color:#f3f4f6; line-height:1.1; }
            .metric-sub   { font-size:0.70rem; color:#93a1b5; font-family:monospace; margin-top:0.2rem; }

            .txt-blue   { color: #58a6ff; }
            .txt-green  { color: #3fb950; }
            .txt-red    { color: #f85149; }
            .txt-orange { color: #ff9d4d; }
            .txt-violet { color: #bc8cff; }

            .panel-wrap {
                background:linear-gradient(180deg,rgba(22,27,34,0.96),rgba(16,21,28,0.96));
                border:1px solid rgba(90,103,126,0.22); border-radius:12px;
                padding:8px 8px 4px 8px; box-shadow:0 6px 20px rgba(0,0,0,0.14);
            }
            .ctrl-label { font-size:0.60rem; color:#8ea0bc; text-transform:uppercase;
                          letter-spacing:0.13em; font-family:monospace; margin-bottom:0.3rem; }
        </style>
    """, unsafe_allow_html=True)


def render_header(selected_station_name, selected_station_id, selected_lat, selected_lon):
    st.markdown('<div class="eyebrow">Illustrative Analysis Only · NOAA GSOD</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="page-title">NOAA GSOD — Climate Time Series</div>',
                unsafe_allow_html=True)

    # Station badge inline with title
    coords = f" · {selected_lat}, {selected_lon}" if selected_lat else ""
    st.markdown(
        f"""<div class="station-badge">
            <div>
                <div class="station-badge-name">{selected_station_name}</div>
                <div class="station-badge-meta">ID: {selected_station_id}{coords}</div>
            </div>
            <div style="margin-left:auto;font-size:0.65rem;color:#4a5568;">
                Click map to change station
            </div>
        </div>""",
        unsafe_allow_html=True,
    )


def render_methodology():
    st.markdown("""
        <div class="hero-box">
            <strong>Data & Method:</strong>
            Daily NOAA GSOD records → monthly precipitation totals · daily temperature series.
            Moving averages: 1-Mo, 3-Mo, 6-Mo (toggle above chart).
            Trendline: linear regression. Frequency analysis: Gumbel EV1 on annual maxima.<br>
            <strong>Disclaimer:</strong> For research/educational use only. Expert review required before operational use.
        </div>
    """, unsafe_allow_html=True)


def make_card(label: str, value: str, sub: str = "", value_class: str = ""):
    sub_html = f'<div class="metric-sub">{sub}</div>' if sub else ""
    html = (
        f'<div class="metric-card">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value {value_class}">{value}</div>'
        f'{sub_html}'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


# ── Philippines places: provinces + cities → (lat, lon, zoom) ────────────────
PH_PROVINCES: dict[str, tuple[float, float, int]] = {
    # ── NCR / Metro Manila ──
    "metro manila": (14.55, 121.00, 11), "ncr": (14.55, 121.00, 11),
    "manila": (14.59, 120.98, 13), "quezon city": (14.68, 121.04, 12),
    "makati": (14.55, 121.02, 13), "pasig": (14.57, 121.08, 13),
    "taguig": (14.52, 121.05, 13), "marikina": (14.65, 121.10, 13),
    "parañaque": (14.48, 121.02, 13), "pasay": (14.54, 121.00, 13),
    "caloocan": (14.65, 120.97, 13), "malabon": (14.66, 120.96, 13),
    "navotas": (14.66, 120.94, 13), "valenzuela": (14.70, 120.98, 13),
    "las piñas": (14.45, 120.99, 13), "muntinlupa": (14.41, 121.04, 13),
    "mandaluyong": (14.58, 121.03, 13), "san juan": (14.60, 121.03, 14),
    "pateros": (14.55, 121.07, 14),
    # ── Region III ──
    "bulacan": (14.90, 120.90, 10), "pampanga": (15.10, 120.65, 10),
    "tarlac": (15.47, 120.58, 10), "nueva ecija": (15.60, 121.10, 10),
    "zambales": (15.55, 120.05, 10), "bataan": (14.65, 120.35, 10),
    "angeles": (15.15, 120.59, 12), "san fernando pampanga": (15.03, 120.69, 12),
    "olongapo": (14.83, 120.28, 12), "cabanatuan": (15.49, 120.97, 12),
    # ── Region IV-A ──
    "cavite": (14.20, 120.87, 10), "laguna": (14.15, 121.40, 10),
    "batangas": (13.75, 121.05, 10), "rizal": (14.60, 121.30, 10),
    "quezon province": (14.00, 122.00, 10),
    "antipolo": (14.63, 121.18, 12), "bacoor": (14.46, 120.93, 13),
    "calamba": (14.21, 121.17, 12), "lucena": (13.94, 121.61, 12),
    "lipa": (13.94, 121.16, 12), "batangas city": (13.76, 121.06, 12),
    # ── Region IV-B ──
    "palawan": (10.00, 119.00, 9), "occidental mindoro": (13.00, 120.75, 10),
    "oriental mindoro": (13.10, 121.35, 10), "marinduque": (13.40, 121.90, 10),
    "romblon": (12.55, 122.25, 10), "puerto princesa": (9.74, 118.74, 12),
    # ── Region V ──
    "camarines norte": (14.20, 122.75, 10), "camarines sur": (13.65, 123.20, 10),
    "albay": (13.20, 123.68, 10), "sorsogon": (12.90, 124.00, 10),
    "catanduanes": (13.70, 124.30, 10), "masbate": (12.30, 123.60, 10),
    "naga": (13.62, 123.19, 12), "legazpi": (13.14, 123.74, 12),
    # ── Region I ──
    "pangasinan": (16.00, 120.33, 10), "la union": (16.60, 120.32, 10),
    "ilocos norte": (18.10, 120.65, 10), "ilocos sur": (17.30, 120.45, 10),
    "dagupan": (16.04, 120.34, 12), "san fernando la union": (16.62, 120.32, 12),
    "laoag": (18.20, 120.59, 12), "vigan": (17.57, 120.39, 13),
    # ── CAR ──
    "benguet": (16.50, 120.80, 10), "baguio": (16.41, 120.60, 12),
    "ifugao": (16.80, 121.15, 10), "mountain province": (17.00, 121.00, 10),
    "kalinga": (17.50, 121.40, 10), "abra": (17.60, 120.70, 10),
    "apayao": (18.00, 121.15, 10),
    # ── Region II ──
    "cagayan": (17.90, 121.80, 10), "isabela": (16.95, 121.80, 10),
    "nueva vizcaya": (16.20, 121.20, 10), "quirino": (16.30, 121.55, 10),
    "aurora": (15.85, 121.60, 10), "tuguegarao": (17.61, 121.73, 12),
    "ilagan": (17.15, 121.89, 12),
    # ── Region VI ──
    "iloilo": (10.72, 122.55, 10), "capiz": (11.30, 122.65, 10),
    "aklan": (11.62, 122.28, 10), "antique": (11.10, 122.02, 10),
    "guimaras": (10.60, 122.60, 11), "negros occidental": (10.40, 123.00, 10),
    "iloilo city": (10.72, 122.56, 12), "bacolod": (10.68, 122.95, 12),
    "roxas": (11.59, 122.75, 12), "kalibo": (11.71, 122.37, 13),
    # ── Region VII ──
    "cebu": (10.30, 123.85, 10), "bohol": (9.85, 124.15, 10),
    "negros oriental": (9.60, 123.05, 10), "siquijor": (9.20, 123.50, 11),
    "cebu city": (10.32, 123.89, 12), "mandaue": (10.35, 123.93, 13),
    "lapu-lapu": (10.31, 123.95, 13), "tagbilaran": (9.66, 123.85, 12),
    "dumaguete": (9.31, 123.31, 12),
    # ── Region VIII ──
    "leyte": (11.00, 124.65, 10), "southern leyte": (10.20, 125.05, 10),
    "samar": (12.00, 125.00, 10), "eastern samar": (11.50, 125.40, 10),
    "northern samar": (12.50, 124.65, 10), "biliran": (11.55, 124.40, 11),
    "tacloban": (11.24, 125.00, 12), "ormoc": (11.01, 124.61, 12),
    # ── Region IX ──
    "zamboanga del norte": (8.40, 123.30, 10),
    "zamboanga del sur": (7.80, 123.40, 10),
    "zamboanga sibugay": (7.70, 122.80, 10),
    "zamboanga city": (6.92, 122.08, 12),
    # ── Region X ──
    "misamis occidental": (8.30, 123.70, 10),
    "misamis oriental": (8.60, 124.70, 10),
    "lanao del norte": (8.05, 124.25, 10), "bukidnon": (8.20, 125.10, 10),
    "camiguin": (9.17, 124.73, 11),
    "cagayan de oro": (8.47, 124.65, 12), "iligan": (8.23, 124.24, 12),
    "ozamiz": (8.15, 123.85, 12),
    # ── Region XI ──
    "davao del norte": (7.55, 125.75, 10), "davao del sur": (6.80, 125.40, 10),
    "davao oriental": (7.00, 126.40, 10), "davao de oro": (7.50, 126.05, 10),
    "davao city": (7.10, 125.63, 12), "tagum": (7.45, 125.81, 12),
    "digos": (6.75, 125.36, 12),
    # ── Region XII ──
    "south cotabato": (6.25, 124.90, 10), "north cotabato": (7.10, 124.65, 10),
    "sultan kudarat": (6.70, 124.30, 10), "sarangani": (5.90, 124.90, 10),
    "general santos": (6.12, 125.17, 12), "koronadal": (6.50, 124.85, 12),
    "kidapawan": (7.01, 125.09, 12),
    # ── Region XIII / Caraga ──
    "agusan del norte": (8.95, 125.55, 10), "agusan del sur": (8.40, 125.95, 10),
    "surigao del norte": (9.75, 125.55, 10), "surigao del sur": (8.70, 126.15, 10),
    "dinagat islands": (10.10, 125.65, 11),
    "butuan": (8.95, 125.54, 12), "surigao": (9.79, 125.50, 12),
    # ── BARMM ──
    "maguindanao": (6.85, 124.40, 10), "basilan": (6.50, 122.00, 11),
    "sulu": (5.90, 121.10, 10), "tawi-tawi": (5.10, 119.90, 10),
    "cotabato city": (7.21, 124.25, 12), "marawi": (7.99, 124.29, 12),
    # ── Whole country ──
    "philippines": (12.5, 122.5, 5), "luzon": (16.0, 121.0, 7),
    "visayas": (10.5, 123.5, 7), "mindanao": (7.5, 125.0, 7),
}


def build_station_selector_map(
    stations_df: pd.DataFrame,
    selected_name: str,
    highlight_names: list | None = None,
    map_center: tuple[float, float, int] | None = None,
) -> go.Figure:
    has_coords = stations_df["LATITUDE"].notna() & stations_df["LONGITUDE"].notna()
    df = stations_df[has_coords].copy()
    df["lat"] = df["LATITUDE"].astype(float)
    df["lon"] = df["LONGITUDE"].astype(float)
    df["is_selected"] = df["NAME"] == selected_name

    fig = go.Figure()

    if highlight_names is not None:
        highlight_set = set(highlight_names)
        dim = df[~df["is_selected"] & ~df["NAME"].isin(highlight_set)]
        if not dim.empty:
            fig.add_trace(go.Scattermapbox(
                lat=dim["lat"], lon=dim["lon"], mode="markers",
                marker=dict(size=5, color="rgba(88,166,255,0.15)"),
                text=dim["NAME"], customdata=dim["NAME"],
                hovertemplate="<b>%{customdata}</b><br><span style='color:#aab6c8;font-size:10px'>click to select</span><extra></extra>",
                name="Other",
            ))
        bright = df[~df["is_selected"] & df["NAME"].isin(highlight_set)]
        if not bright.empty:
            fig.add_trace(go.Scattermapbox(
                lat=bright["lat"], lon=bright["lon"], mode="markers",
                marker=dict(size=10, color="rgba(88,166,255,0.95)"),
                text=bright["NAME"], customdata=bright["NAME"],
                hovertemplate="<b>%{customdata}</b><br><span style='color:#58a6ff;font-size:10px'>▲ click to select & process</span><extra></extra>",
                name="Match",
            ))
    else:
        unsel = df[~df["is_selected"]]
        if not unsel.empty:
            fig.add_trace(go.Scattermapbox(
                lat=unsel["lat"], lon=unsel["lon"], mode="markers",
                marker=dict(size=7, color="rgba(88,166,255,0.70)"),
                text=unsel["NAME"], customdata=unsel["NAME"],
                hovertemplate="<b>%{customdata}</b><br><span style='color:#aab6c8;font-size:10px'>click to select & process</span><extra></extra>",
                name="Stations",
            ))

    sel = df[df["is_selected"]]
    if not sel.empty:
        fig.add_trace(go.Scattermapbox(
            lat=sel["lat"], lon=sel["lon"], mode="markers+text",
            marker=dict(size=14, color="#f0883e"),
            text=sel["NAME"], textposition="top right",
            textfont=dict(size=10, color="#f0883e"),
            customdata=sel["NAME"],
            hovertemplate="<b>%{customdata}</b> ✓<extra></extra>",
            name="Selected",
        ))

    if map_center is not None:
        centre_lat, centre_lon, zoom = map_center
    else:
        # Always default to full Philippines — user scrolls/zooms to navigate
        centre_lat, centre_lon, zoom = 12.5, 122.5, 5

    fig.update_layout(
        mapbox=dict(style="carto-darkmatter",
                    center=dict(lat=centre_lat, lon=centre_lon), zoom=zoom),
        height=520,
        paper_bgcolor="#0d1117",
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        uirevision=f"map_{selected_name}_{centre_lat:.2f}_{centre_lon:.2f}",
    )
    return fig