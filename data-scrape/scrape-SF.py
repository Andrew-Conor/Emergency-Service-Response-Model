import pandas as pd
import requests


url_fire = "https://data.sf.gov/resource/nuek-vuh3.json"

# Let the requests library perfectly encode your parameters to avoid 400 errors
api_params = {
    "$limit": 100,
    "$order": "received_dttm DESC",
    "$where": "received_dttm >= '2025-09-29T00:00:00'"
}

response_fire = requests.get(url_fire, params=api_params)

results_df = pd.DataFrame.from_records(response_fire.json())

longitude, latitude = [], []
for data in results_df['case_location']:
    coord = data.get('coordinates')
    longitude.append(coord[0])
    latitude.append(coord[1])

results_df['longitude'] = longitude
results_df['latitude'] = latitude

results_df['received_dttm'] = pd.to_datetime(results_df['received_dttm'])
results_df['on_scene_dttm'] = pd.to_datetime(results_df['on_scene_dttm'])

results_df['incident_response_time_s'] = (results_df['on_scene_dttm'] - results_df['received_dttm']).dt.total_seconds()

final_df = results_df[['received_dttm', 'call_type_group', 'call_type', 'received_dttm', 'response_dttm', 'on_scene_dttm', 'incident_response_time_s', 'station_area', 'original_priority', 'longitude', 'latitude']]

final_df.to_csv('SF-data.csv', index=False)