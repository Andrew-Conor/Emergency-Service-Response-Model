import pandas as pd
from datetime import datetime
import numpy as np
import holidays
from process.coordinate_ops import match_station, distance_between_coords

FINAL_COLUMNS = [
    "received_dttm",
    "call_type_group",
    "call_type",
    "on_scene_dttm",
    "incident_response_time",
    "station_area",
    "original_priority",
    "distance_km",
    "unit_type",
    "is_weekend",
    "is_holiday",
    "outcome",
    "call_number",
    "unit_id",
    "incident_number",
    "call_date",
    "watch_date",
    "entry_dttm",
    "dispatch_dttm",
    "response_dttm",
    "transport_dttm",
    "hospital_dttm",
    "call_final_disposition",
    "available_dttm",
    "address",
    "city",
    "zipcode_of_incident",
    "battalion",
    "box",
    "priority",
    "final_priority",
    "als_unit",
    "number_of_alarms",
    "unit_sequence_in_call_dispatch",
    "fire_prevention_district",
    "supervisor_district",
    "neighborhoods_analysis_boundaries",
    "rowid",
    "data_as_of",
    "data_loaded_at"
]

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
    years = dates.dt.year.unique().tolist()
    SF_holidays = holidays.US(state="CA", years=years, observed=True)
    dates = dates.dt.date
    return np.where(dates.isin(SF_holidays), 1, 0)

def outcome(times):
    return np.where(times.isna(), np.nan, np.where(times > 539, 1, 0))
    
def process_sf_data(df):
    df = extract_coordinates(df)
    df['incident_response_time'] = calculate_response_time(df['received_dttm'], df['on_scene_dttm'])
    df['is_weekend'] = is_weekend(df['received_dttm'])
    df['is_holiday'] = is_holiday(df['received_dttm'])
    df["incident_response_time"] = calculate_response_time(
        df["received_dttm"], df["on_scene_dttm"]
    )
    match_station(df, df["station_area"])
    distance_between_coords(df)
    df['outcome'] = outcome(df['incident_response_time'])

    return df[FINAL_COLUMNS]
