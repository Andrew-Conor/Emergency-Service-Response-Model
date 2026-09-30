import requests
import pandas as pd

url = 'https://archive-api.open-meteo.com/v1/archive'

params = {
    'latitude': 37.7749,
    'longitude': -122.4194,
    'start_date': '2025-09-29',
    'end_date': '2026-09-29',
    'hourly': 'temperature_2m,rain,snowfall,wind_speed_10m',
}

resp = requests.get(url=url, params=params)
data = resp.json()


meteo_df = pd.DataFrame(data['hourly'])
meteo_units = data['hourly_units']
