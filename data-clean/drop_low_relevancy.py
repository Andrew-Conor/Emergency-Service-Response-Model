import pandas as pd
from pathlib import Path
from load_data import load_data

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = (
    PROJECT_ROOT
    / "csvs"
    / "Categorical-Redundancy-Removed-Fire-and-EMS-Response-Data.csv"
)
OUTPUT_PATH = (
    PROJECT_ROOT / "csvs" / "Low-Relevance-Removed-Fire-and-EMS-Response-Data.csv"
)

LOW_RELEVANCE = [
    "dew_point_2m",
    "wind_speed_10m",
    "shortwave_radiation",
    "diffuse_radiation",
    "sunshine_duration",
    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "weather_code",
    "surface_pressure",
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "als_unit",
    "is_holiday",
    "hour_of_day",
    "day_of_week",
]


def main():
    df = load_data(INPUT_PATH)
    original_columns = df.columns.tolist()
    print(f"Loaded {INPUT_PATH.name}: {len(df):,} rows, {df.shape[1]} columns")

    df = df.drop(columns=LOW_RELEVANCE)

    dropped_columns = set(original_columns) - set(df.columns.tolist())
    print(f"Dropped {len(dropped_columns)} low-relevance columns:")
    for col in sorted(dropped_columns):
        print(f"  - {col}")

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {OUTPUT_PATH.name}: {len(df):,} rows, {df.shape[1]} columns")


if __name__ == "__main__":
    main()
