import pandas as pd
from datetime import datetime
import numpy as np
import holidays
from process.coordinate_ops import match_station

FINAL_COLUMNS = [
    "received_dttm",
    "call_type_group",
    "call_type",
    "on_scene_dttm",
    "incident_response_time",
    "station_area",
    "station_long",
    "station_lat",
    "original_priority",
    "longitude",
    "latitude",
    "unit_type",
    "is_weekend",
    "is_holiday"
]


def remove_no_scene_arrivals(df):
    return df[df["on_scene_dttm"].notna()]


def keep_only_first_arrival(df):
    """
    Each row in SF data is a unit dispatch. Keep only the first unit to arrive per incident.
    """
    df = df.sort_values(["incident_number", "on_scene_dttm"], na_position="last")
    df = df.drop_duplicates(subset="incident_number", keep="first")
    return df.sort_values("received_dttm", ascending=False)


def extract_coordinates(df):
    coords = pd.Series(
        [
            loc.get("coordinates") if isinstance(loc, dict) else None
            for loc in df["case_location"]
        ],
        index=df.index,
    )
    df["longitude"] = coords.str[0]
    df["latitude"] = coords.str[1]

    missing = df["longitude"].isna()
    if missing.any():
        print(f"{missing.sum()} of {len(df)} incidents have no case_location:")
        print(
            df.loc[
                missing, ["address", "zipcode_of_incident", "station_area", "call_type"]
            ]
            .head(20)
            .to_string()
        )

    return df


def handle_missing_values(df):
    df = df.dropna(subset=["longitude", "latitude", "station_area"])
    df["call_type_group"] = df["call_type_group"].fillna("Unknown")
    df["original_priority"] = df["original_priority"].fillna("Unknown")
    return df


def calculate_response_time(received, on_scene):
    received = pd.to_datetime(received)
    on_scene = pd.to_datetime(on_scene)
    return (on_scene - received).dt.total_seconds()

def is_weekend(dates):
    """ Additional column to check if a given date is the weekend https://www.geeksforgeeks.org/python/ways-to-apply-an-if-condition-in-pandas-dataframe/"""
    dates = pd.to_datetime(dates)
    return np.where(dates.dt.weekday > 4, 1, 0)

def is_holiday(dates):
    """ Additional column to check if a given date is a recognised public holiday in california"""
    dates = pd.to_datetime(dates)
    SF_holidays = holidays.US(state="CA", years=2025, observed=True)
    dates = dates.dt.date
    return np.where(dates.isin(SF_holidays), 1, 0)
    
def process_sf_data(df):
    df = remove_no_scene_arrivals(df)
    df = keep_only_first_arrival(df)
    df = extract_coordinates(df)
    df['incident_response_time'] = calculate_response_time(df['received_dttm'], df['on_scene_dttm'])
    df['is_weekend'] = is_weekend(df['received_dttm'])
    df['is_holiday'] = is_holiday(df['received_dttm'])
    df = handle_missing_values(df)
    df["incident_response_time"] = calculate_response_time(
        df["received_dttm"], df["on_scene_dttm"]
    )
    match_station(df, df["station_area"])

    return df[FINAL_COLUMNS]
