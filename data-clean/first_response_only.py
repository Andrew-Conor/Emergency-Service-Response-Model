import pandas as pd
from pathlib import Path
from load_data import load_data

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / "csvs" / "Redundant-Columns-Fire-and-EMS-Response-Data.csv"
OUTPUT_PATH = (
    PROJECT_ROOT / "csvs" / "First-Response-Only-Fire-and-EMS-Response-Data.csv"
)

REDUNDANT_SF_COLUMNS = [
    "incident_number",
    "on_scene_dttm",
    "received_dttm",
]


def keep_only_first_arrival(df):
    df = df.assign(
        _arrival=pd.to_datetime(df["on_scene_dttm"]),
        _received=pd.to_datetime(df["received_dttm"]),
    )  # because kept as strng when read in from csv
    df = df.sort_values(["incident_number", "_arrival"], na_position="last")
    df = df.drop_duplicates(subset="incident_number", keep="first")
    return df.sort_values("_received", ascending=False).drop(
        columns=["_arrival", "_received"]
    )


def main():
    df = load_data(INPUT_PATH)
    original_rows = len(df)
    print(f"Loaded {INPUT_PATH.name}: {original_rows:,} rows, {df.shape[1]} columns")

    df = keep_only_first_arrival(df)
    print(
        f"Kept first arrival per incident: {original_rows:,} -> {len(df):,} rows "
        f"({original_rows - len(df):,} removed)"
    )

    df = df.drop(columns=REDUNDANT_SF_COLUMNS)

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {OUTPUT_PATH.name}: {len(df):,} rows, {df.shape[1]} columns")


if __name__ == "__main__":
    main()
