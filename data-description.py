"""
Prints a full description of the raw Fire and EMS dataset produced by main.py.

Place this file in the project root (next to the csvs, data-scrape and data-clean folders).

Usage:
    python describe_data.py
    python describe_data.py csvs/other.csv
    python describe_data.py > description.txt      # save the output to a file
"""

import sys
from pathlib import Path
import pandas as pd

# Resolved from this file's location, so it works whichever folder you run it from
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PATH = PROJECT_ROOT / "csvs" / "Raw-Fire-and-EMS-Response-Data.csv"
# DEFAULT_PATH = PROJECT_ROOT / "csvs" / "Redundant-Fire-and-EMS-Response-Data.csv"
MAX_CATEGORIES = (
    50  # columns with this many unique values or fewer get full value counts
)
TOP_N = 10  # higher-cardinality text columns show only their most common values

# Codes and IDs: read as text so "01" stays "01" and they aren't averaged as numbers
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
DATE_COLUMNS_EXTRA = ("call_date", "watch_date", "time")


def is_date_column(name):
    return name.endswith("_dttm") or name in DATE_COLUMNS_EXTRA


pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)


def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def overview(df):
    section("1. SIZE")
    print(f"Rows:    {len(df):,}")
    print(f"Columns: {df.shape[1]}")


def column_summary(df):
    section("2. COLUMNS (name, type, missing values, unique values)")
    summary = pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "non_null": df.notna().sum(),
            "missing": df.isna().sum(),
            "missing_%": (df.isna().mean() * 100).round(2),
            "unique": df.nunique(dropna=True),
        }
    )
    summary.index.name = "column"
    print(summary.to_string())


def missing_per_row(df):
    section("3. MISSING VALUES PER ROW")
    per_row = df.isna().sum(axis=1)
    print(
        f"Rows with no missing values: {(per_row == 0).sum():,} "
        f"({(per_row == 0).mean() * 100:.2f}%)"
    )
    print(f"Mean missing values per row: {per_row.mean():.2f}")
    print(f"Max missing values in a row: {per_row.max()}\n")
    dist = per_row.value_counts().sort_index()
    dist.index.name = "missing_in_row"
    print(
        pd.DataFrame({"rows": dist, "%": (dist / len(df) * 100).round(2)}).to_string()
    )


def duplicates(df):
    section("4. DUPLICATES")
    print(f"Fully duplicated rows: {df.duplicated().sum():,}")
    if "incident_number" in df.columns:
        dup = df["incident_number"].duplicated().sum()
        print(f"Rows sharing an incident_number with an earlier row: {dup:,}")
        print(f"Unique incidents: {df['incident_number'].nunique():,}")


def constant_columns(df):
    section("5. CONSTANT COLUMNS (one value only)")
    constant = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    if constant:
        for c in constant:
            print(f"{c}: {df[c].dropna().unique().tolist()}")
    else:
        print("None")


def numeric_summary(df):
    section("6. NUMERIC COLUMNS (central tendency and dispersion)")
    num = df.select_dtypes(include="number")
    if num.empty:
        print("No numeric columns")
        return
    stats = num.describe().T
    stats["median"] = num.median()
    stats["IQR"] = stats["75%"] - stats["25%"]
    stats["skew"] = num.skew()
    print(stats.round(3).to_string())


def date_ranges(df):
    section("7. DATE/TIME RANGES")
    date_cols = [c for c in df.columns if is_date_column(c)]
    for c in date_cols:
        dates = pd.to_datetime(df[c], errors="coerce")
        print(
            f"{c:<20} {str(dates.min()):<22} -> {str(dates.max()):<22} ({dates.isna().sum():,} missing)"
        )


def value_counts(df):
    section("8. VALUE COUNTS PER COLUMN")
    print(
        f"(full counts for columns with <= {MAX_CATEGORIES} unique values, "
        f"top {TOP_N} for other text columns, high-cardinality numeric columns skipped)"
    )
    for c in df.columns:
        if is_date_column(c) or c == "outcome":  # dates covered in 7, outcome in 9
            continue
        n_unique = df[c].nunique(dropna=True)
        is_numeric = pd.api.types.is_numeric_dtype(df[c])

        if n_unique > MAX_CATEGORIES and is_numeric:
            continue

        counts = df[c].value_counts(dropna=False)
        table = pd.DataFrame({"count": counts, "%": (counts / len(df) * 100).round(2)})

        print(f"\n--- {c} ({n_unique} unique) ---")
        if n_unique > MAX_CATEGORIES:
            print(table.head(TOP_N).to_string())
            print(f"... {n_unique - TOP_N:,} other values")
        else:
            print(table.to_string())


def target_balance(df):
    if "outcome" not in df.columns:
        return
    section("9. TARGET CLASS BALANCE (outcome: 1 = missed 8:59 target)")
    counts = df["outcome"].value_counts(dropna=False)
    print(
        pd.DataFrame(
            {"count": counts, "%": (counts / len(df) * 100).round(2)}
        ).to_string()
    )


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH
    header = pd.read_csv(path, nrows=0).columns
    dtypes = {c: str for c in TEXT_COLUMNS if c in header}
    df = pd.read_csv(path, dtype=dtypes, low_memory=False)
    print(f"Description of: {path}")

    overview(df)
    column_summary(df)
    missing_per_row(df)
    duplicates(df)
    constant_columns(df)
    numeric_summary(df)
    date_ranges(df)
    value_counts(df)
    target_balance(df)


if __name__ == "__main__":
    main()
