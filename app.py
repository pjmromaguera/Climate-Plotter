import streamlit as st
from datetime import date

from data_loader import load_station_csv, merge_station_data
from precipitation import prepare_precip_data, precip_plot, precip_summary
from temperature import prepare_temp_data, temp_plot, temp_summary
from ui_components import make_card

st.set_page_config(page_title="NOAA Climate Explorer", layout="wide")

# -----------------------------------
# Load stations
# -----------------------------------
stations = load_station_csv()

station_name = st.sidebar.selectbox(
    "Station",
    stations["NAME"]
)

row = stations.loc[stations["NAME"] == station_name].iloc[0]
station_id = row["ID"]
lat = row["LATITUDE"]
lon = row["LONGITUDE"]

start_date = st.sidebar.date_input("Start date", date(1973,1,1))
end_date = st.sidebar.date_input("End date", date(2025,12,31))

ma_option = st.sidebar.selectbox(
    "Moving Average",
    ["None","1-month","3-month","6-month"]
)

show_trend = st.sidebar.checkbox("Show Trendline", True)

run = st.sidebar.button("Run")

if not run:
    st.stop()

# -----------------------------------
# Download data
# -----------------------------------
daily = merge_station_data(
    station_id,
    start_date,
    end_date
)

# -----------------------------------
# Prepare data
# -----------------------------------
precip_df, precip_trend = prepare_precip_data(daily, ma_option, show_trend)
temp_df, temp_trend = prepare_temp_data(daily, ma_option, show_trend)

precip_sum = precip_summary(precip_df, precip_trend)
temp_sum = temp_summary(temp_df, temp_trend)

# -----------------------------------
# Tabs
# -----------------------------------
tab1, tab2 = st.tabs(["Precipitation","Temperature"])

# -----------------------------------
# PRECIP
# -----------------------------------
with tab1:

    c1,c2,c3,c4,c5 = st.columns(5)

    with c1:
        make_card("Period", precip_sum["period"])

    with c2:
        make_card("Monthly Avg", f"{precip_sum['avg']:.1f} mm")

    with c3:
        make_card("Annual Avg", f"{precip_sum['annual_avg']:.0f} mm/yr")

    with c4:
        make_card(
            "Max Month",
            precip_sum["max_label"],
            f"{precip_sum['max_value']:.0f} mm"
        )

    with c5:
        make_card(
            "Trend",
            f"{precip_sum['slope']:.2f} mm/yr"
        )

    st.plotly_chart(
        precip_plot(
            precip_df,
            station_name,
            lat,
            lon,
            ma_option,
            show_trend
        ),
        use_container_width=True
    )

# -----------------------------------
# TEMP
# -----------------------------------
with tab2:

    c1,c2,c3,c4,c5 = st.columns(5)

    with c1:
        make_card("Period", temp_sum["period"])

    with c2:
        make_card("Daily Avg", f"{temp_sum['avg']:.2f} °C")

    with c3:
        make_card("Series Avg", f"{temp_sum['series_avg']:.2f} °C")

    with c4:
        make_card(
            "Warmest Day",
            temp_sum["max_label"],
            f"{temp_sum['max_value']:.2f} °C"
        )

    with c5:
        make_card(
            "Trend",
            f"{temp_sum['slope']:.2f} °C/yr"
        )

    st.plotly_chart(
        temp_plot(
            temp_df,
            station_name,
            lat,
            lon,
            ma_option,
            show_trend
        ),
        use_container_width=True
    )