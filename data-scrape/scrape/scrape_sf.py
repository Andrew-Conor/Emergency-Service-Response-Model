import pandas as pd
import requests

URL_FIRE = "https://data.sf.gov/resource/nuek-vuh3.json"

def scrape_sf_data(limit=100, order="received_dttm DESC", where="received_dttm < '2026-09-29T00:00:00'"):

    api_params = {
        "$limit": limit,
        "$order": order,
        "$where": where
    }

    sf_data = requests.get(url=URL_FIRE, params=api_params, timeout=30)

    return pd.DataFrame.from_records(sf_data.json())
