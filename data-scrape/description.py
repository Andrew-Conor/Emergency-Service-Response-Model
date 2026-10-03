import pandas as pd

CATEGORICAL_COLUMNS = [
    "call_type_group",
    "call_type",
    "original_priority",
    "station_area",
]

df = pd.read_csv(
    "csvs/Fire-and-EMS-Response-Data.csv",
    dtype={col: str for col in CATEGORICAL_COLUMNS},
)

for col in CATEGORICAL_COLUMNS:
    print(f"\n{col} ({df[col].nunique()} unique values):")
    print(df[col].value_counts(dropna=False).to_string())

df["response_time_min"] = df["incident_response_time"] / 60

response_by_call_type = (
    df.groupby("call_type")["response_time_min"]
    .agg(count="count", mean="mean", median="median")
    .round(2)
    .sort_values("count", ascending=False)
)
print("\nResponse time by call type (minutes):")
print(response_by_call_type.to_string())

response_by_priority = (
    df.groupby("original_priority")["response_time_min"]
    .agg(count="count", mean="mean", median="median")
    .round(2)
    .sort_values("count", ascending=False)
)
print("\nResponse time by original priority (minutes):")
print(response_by_priority.to_string())
