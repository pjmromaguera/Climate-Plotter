import pandas as pd
import requests
import io
from pathlib import Path

BASE_URL = "https://www.ncei.noaa.gov/data/global-summary-of-the-day/access"
STATIONS_FILE = Path("stations.csv")

def load_station_csv():
    df = pd.read_csv(STATIONS_FILE)
    df["ID"] = df["ID"].astype(str).str.zfill(11)
    return df


def fetch_year(station_id, year):

    url = f"{BASE_URL}/{year}/{station_id}.csv"

    r = requests.get(url)

    if r.status_code != 200:
        return None

    return pd.read_csv(io.StringIO(r.text))


def merge_station_data(station_id,start_date,end_date):

    dfs = []

    for year in range(start_date.year, end_date.year+1):

        df = fetch_year(station_id,year)

        if df is None:
            continue

        dfs.append(df)

    df = pd.concat(dfs)

    df["DATE"] = pd.to_datetime(df["DATE"])

    df["PRCP"] = pd.to_numeric(df["PRCP"],errors="coerce")
    df["TEMP"] = pd.to_numeric(df["TEMP"],errors="coerce")

    df.loc[df["PRCP"]==99.99,"PRCP"]=None
    df.loc[df["TEMP"]==9999.9,"TEMP"]=None

    df["rain_mm"] = df["PRCP"]*25.4
    df["temp_c"] = (df["TEMP"]-32)*5/9

    return df[["DATE","rain_mm","temp_c"]].rename(
        columns={"DATE":"date"}
    )