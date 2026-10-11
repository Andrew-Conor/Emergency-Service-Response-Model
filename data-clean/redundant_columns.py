import pandas as pd
from pathlib import Path
from load_data import load_data

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / "csvs" / "Raw-Fire-and-EMS-Response-Data.csv"
OUTPUT_PATH = PROJECT_ROOT / "csvs" / "Redundant-Columns-Fire-and-EMS-Response-Data.csv"

REDUNDANT_SF_COLUMNS = [
    "call_number",
    "unit_id",
    "rowid",
    "call_final_disposition",
    "address",
    "final_priority",
    "data_as_of",
    "data_loaded_at",
    "call_date",
    "watch_date",
    "entry_dttm",
    "time",
    "dispatch_dttm",
    "response_dttm",
    "transport_dttm",
    "hospital_dttm",
    "available_dttm",
    "incident_response_time",  # outcome covers it
    "priority",  # original_priority covers it
    "number_of_alarms",  # Could be changed after first arrival - also all seem to just be 1 anyways
    "city",  # 99% San Fran, there are others but too small to affect model and location calculations aren't always there
    "box",  # 2006 possible values with no real correlation
    "unit_sequence_in_call_dispatch",  # Not relevant to model
]

REDUNDANT_WEATHER_COLUMNS = [
    "soil_temperature_0_to_7cm",
    "soil_temperature_7_to_28cm",
    "soil_temperature_28_to_100cm",
    "soil_temperature_100_to_255cm",
    "soil_moisture_0_to_7cm",
    "soil_moisture_7_to_28cm",
    "soil_moisture_28_to_100cm",
    "soil_moisture_100_to_255cm",
    "snowfall",  # Only appears in 40 rows
]


def drop_sf_columns(df):
    return df.drop(columns=REDUNDANT_SF_COLUMNS, errors="ignore")


def drop_weather_columns(df):
    return df.drop(columns=REDUNDANT_WEATHER_COLUMNS, errors="ignore")


def main():
    df = load_data(INPUT_PATH)
    original_columns = df.columns.tolist()
    print(f"Loaded {INPUT_PATH.name}: {len(df):,} rows, {df.shape[1]} columns")

    df = drop_sf_columns(df)
    df = drop_weather_columns(df)

    dropped_columns = set(original_columns) - set(df.columns.tolist())
    print(f"Dropped {len(dropped_columns)} redundant columns:")
    for col in sorted(dropped_columns):
        print(f"  - {col}")

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {OUTPUT_PATH.name}: {len(df):,} rows, {df.shape[1]} columns")


if __name__ == "__main__":
    main()
