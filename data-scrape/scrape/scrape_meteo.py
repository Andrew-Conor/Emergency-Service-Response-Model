import requests
import pandas as pd

URL_METEO = 'https://archive-api.open-meteo.com/v1/archive'
METEO_COLUMNS = 'temperature_2m,rain,snowfall,wind_speed_10m'

def scrape_meteo_data(start_date='2025-09-29', end_date='2026-09-29'):

    params = {
        'latitude': 37.7749,
        'longitude': -122.4194,
        'start_date': start_date,
        'end_date': end_date,
        'hourly': METEO_COLUMNS,
        'timezone': 'America/Los_Angeles',
    }

    resp = requests.get(url=URL_METEO, params=params, timeout=30)

    df = pd.DataFrame(resp.json()['hourly'])
    df['time'] = pd.to_datetime(df['time']) # matches sf's response column

    return df
