import pandas as pd

# Codes and IDs: read as text so "01" stays "01"
TEXT_COLUMNS = [
    "station_area",
    "battalion",
    "box",
    "zipcode_of_incident",
    "supervisor_district",
    "fire_prevention_district",
    "incident_number",
    "call_number",
    "unit_id",
    "rowid",
    "original_priority",
    "priority",
    "final_priority",
]


def load_data(path):
    # Codes read as text so "01" stays "01" instead of becoming the number 1
    return pd.read_csv(path, dtype={c: str for c in TEXT_COLUMNS}, low_memory=False)