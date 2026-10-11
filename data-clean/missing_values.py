import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / "csvs" / "Low-Relevance-Removed-Fire-and-EMS-Response-Data.csv"
OUTPUT_PATH = PROJECT_ROOT / "csvs" / "Missing-Values-Handled-Fire-and-EMs-Response-Data.csv"

def drop_rows(df):
    df = df.dropna(subset=['outcome', 'original_priority', 'station_area', 'neighborhoods_analysis_boundaries', 'distance_km']).copy()
    df['call_type_group'] = df['call_type_group'].fillna("Unknown")
    return df

def main():
    df = pd.read_csv(INPUT_PATH)
    df = drop_rows(df)
    df.to_csv(OUTPUT_PATH, index=False)


if __name__ == "__main__":
    main()