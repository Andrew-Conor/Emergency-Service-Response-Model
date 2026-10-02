import pandas as pd
import requests

URL_FIRE = "https://data.sf.gov/resource/nuek-vuh3.json"

FINAL_COLUMNS = [
    'received_dttm', 'call_type_group', 'call_type', 'response_dttm',
    'on_scene_dttm', 'incident_response_time_s', 'station_area',
    'original_priority', 'longitude', 'latitude',
]

def fetch_sf_data(limit=100, order="received_dttm DESC", where="received_dttm >= '2025-09-29T00:00:00'"):

    api_params = {
        "$limit": limit,
        "$order": order,
        "$where": where
    }

    sf_data = requests.get(url=URL_FIRE, params=api_params)

    return pd.DataFrame.from_records(sf_data.json())

def extract_coordinates(df):
    longitude, latitude = [], []
    for data in df['case_location']:
        coord = data.get('coordinates')
        longitude.append(coord[0])
        latitude.append(coord[1])

    df['longitude'] = longitude
    df['latitude'] = latitude

    return df

def calculate_response_time(received, on_scene):
    received = pd.to_datetime(received)
    on_scene = pd.to_datetime(on_scene)
    return (on_scene - received).dt.total_seconds()

def scrape_sf_data():
    results_df = fetch_sf_data()
    results_df = extract_coordinates(results_df)
    results_df['incident_response_time_s'] = calculate_response_time(results_df['received_dttm'], results_df['on_scene_dttm'])

    return results_df[FINAL_COLUMNS]

# final_df.to_csv('SF-data.csv', index=False)