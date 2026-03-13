"""
Shared chart theme tokens.
The Plotly charts use transparent paper/plot backgrounds so they
inherit the page background colour naturally in both light and dark
Streamlit themes.  Axis text, grid lines and annotation colours
are chosen to work on both light and dark surfaces by using
mid-grey tones that remain legible in either context.
"""

# ── Data-series colours — vibrant enough for both themes ─────────────────────
BAR_FILL_DARK    = "rgba(88,166,255,0.50)"
BAR_FILL_LIGHT   = "rgba(26,107,191,0.40)"
BAR_LINE_DARK    = "rgba(88,166,255,0.80)"
BAR_LINE_LIGHT   = "rgba(26,107,191,0.70)"

TEMP_LINE_DARK   = "rgba(188,140,255,0.45)"
TEMP_LINE_LIGHT  = "rgba(109,40,217,0.40)"

MA1_COLOR  = "#e07020"   # amber-orange   — good on both
MA3_COLOR  = "#7c3aed"   # violet         — good on both
MA6_COLOR  = "#b45309"   # dark amber     — good on both
TREND_COLOR= "#16a34a"   # green          — good on both

FREQ_24H   = "#2563eb"   # blue
FREQ_72H   = "#7c3aed"   # violet

# ── Layout neutrals — transparent BG, mid-grey grid/ticks ────────────────────
BG_PAPER   = "rgba(0,0,0,0)"   # transparent → inherits container BG
BG_PLOT    = "rgba(0,0,0,0)"
GRID_COLOR = "rgba(128,128,128,0.20)"
TICK_COLOR = "#888888"
FONT_COLOR = "#555555"          # mid-grey renders on both themes
ANN_COLOR  = "#777777"
MENU_BG    = "rgba(128,128,128,0.15)"
MENU_BORDER= "rgba(128,128,128,0.35)"
MENU_FONT  = "#444444"