import pandas as pd
from datetime import datetime
import numpy as np
import holidays

FINAL_COLUMNS = [
    'received_dttm', 'call_type_group', 'call_type', 'response_dttm',
    'on_scene_dttm', 'incident_response_time', 'station_area',
    'original_priority', 'longitude', 'latitude', 'is_weekend',
    'is_holiday'
]

def extract_coordinates(df):
    longitude, latitude = [], []
    for data in df['case_location']:
        coord = data.get('coordinates')
        longitude.append(coord[0])
        latitude.append(coord[1])

    df['longitude'] = longitude
    df['latitude'] = latitude

    return df

def keep_only_first_arrival(df):
    """
    Each Row in SF data is everytime a unit is dispatched - Some never arrive on scene (we just want first response),.
    """
    df = df.sort_values(['incident_number', 'on_scene_dttm'], na_position='last')
    df = df.drop_duplicates(subset='incident_number', keep='first')
    return df.sort_values('received_dttm', ascending=False)

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
    df = keep_only_first_arrival(df)
    df = extract_coordinates(df)
    df['incident_response_time'] = calculate_response_time(df['received_dttm'], df['on_scene_dttm'])
    df['is_weekend'] = is_weekend(df['received_dttm'])
    df['is_holiday'] = is_holiday(df['received_dttm'])

    return df[FINAL_COLUMNS]