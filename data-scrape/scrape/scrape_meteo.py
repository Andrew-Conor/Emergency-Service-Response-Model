import requests
import pandas as pd

URL_METEO = 'https://archive-api.open-meteo.com/v1/archive'
METEO_COLUMNS = 'temperature_2m,relative_humidity_2m,dew_point_2m,apparent_temperature,surface_pressure,precipitation,cloud_cover,cloud_cover_low,cloud_cover_mid,cloud_cover_high,shortwave_radiation,direct_radiation,direct_normal_irradiance,diffuse_radiation,global_tilted_irradiance,sunshine_duration,wind_speed_100m,wind_gusts_10m,et0_fao_evapotranspiration,weather_code,vapour_pressure_deficit,soil_temperature_0_to_7cm,soil_temperature_7_to_28cm,soil_temperature_28_to_100cm,soil_temperature_100_to_255cm,soil_moisture_0_to_7cm,soil_moisture_7_to_28cm,soil_moisture_28_to_100cm,soil_moisture_100_to_255cm,rain,snowfall,wind_speed_10m'

def scrape_meteo_data(start_date='2024-09-28', end_date='2026-09-29'):

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
