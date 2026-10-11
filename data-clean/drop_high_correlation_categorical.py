from pathlib import Path
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats.contingency import association
from load_data import load_data

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = (
    PROJECT_ROOT / "csvs" / "Numeric-Redundancy-Removed-Fire-and-EMS-Response-Data.csv"
)
OUTPUT_PATH = (
    PROJECT_ROOT
    / "csvs"
    / "Categorical-Redundancy-Removed-Fire-and-EMS-Response-Data.csv"
)
FIGURES_DIR = PROJECT_ROOT / "data-clean" / "figures"

THRESHOLD = 0.8
CATEGORICAL_COLUMNS = [
    "station_area",
    "battalion",
    "zipcode_of_incident",
    "supervisor_district",
    "fire_prevention_district",
    "neighborhoods_analysis_boundaries",
    "call_type",
    "call_type_group",
    "original_priority",
    "day_of_week",
    "is_weekend",
    "is_holiday",
]
CATEGORICAL_REDUNDANT = [
    "is_weekend",
    "fire_prevention_district",
    "supervisor_district",
    "zipcode_of_incident",
]


def unify_zipcodes(df):
    df["zipcode_of_incident"] = df["zipcode_of_incident"].str.replace(
        r"\.0$", "", regex=True
    )
    return df


def cramers_v_matrix(df, columns):
    matrix = pd.DataFrame(1.0, index=columns, columns=columns)
    for i, a in enumerate(columns):
        for b in columns[i + 1 :]:
            table = pd.crosstab(df[a], df[b])
            matrix.loc[a, b] = matrix.loc[b, a] = association(table, method="cramer")
    return matrix


def print_pairs(matrix, label):
    print(f"\n{label}: pairs with Cramér's V >= {THRESHOLD}")
    cols = matrix.columns
    pairs = sorted(
        [
            (a, b, matrix.loc[a, b])
            for i, a in enumerate(cols)
            for b in cols[i + 1 :]
            if matrix.loc[a, b] >= THRESHOLD
        ],
        key=lambda p: -p[2],
    )
    for a, b, v in pairs:
        print(f"  {v:.3f}  {a}  <->  {b}")
    if not pairs:
        print("  None")


def save_heatmap(matrix, path, title):
    fig, ax = plt.subplots(figsize=(10, 9))
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(matrix)), matrix.columns, rotation=90)
    ax.set_yticks(range(len(matrix)), matrix.index)
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            value = matrix.iat[i, j]
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if value > 0.6 else "black",
            )
    fig.colorbar(image, ax=ax, fraction=0.046, label="Cramér's V")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    df = unify_zipcodes(load_data(INPUT_PATH))
    FIGURES_DIR.mkdir(exist_ok=True)

    columns = [c for c in CATEGORICAL_COLUMNS if c in df.columns]
    before = cramers_v_matrix(df, columns)
    save_heatmap(
        before,
        FIGURES_DIR / "cramers_v_before.png",
        f"Before: {len(columns)} categorical columns",
    )
    print_pairs(before, "Before")

    if not CATEGORICAL_REDUNDANT:
        print("\nAdd the columns to drop to CATEGORICAL_REDUNDANT and run again.")
        return

    df = df.drop(columns=CATEGORICAL_REDUNDANT)
    columns = [c for c in columns if c not in CATEGORICAL_REDUNDANT]
    after = cramers_v_matrix(df, columns)
    save_heatmap(
        after,
        FIGURES_DIR / "cramers_v_after.png",
        f"After: {len(columns)} categorical columns",
    )
    print_pairs(after, "After")

    df.to_csv(OUTPUT_PATH, index=False)
    print(
        f"\nDropped {len(CATEGORICAL_REDUNDANT)} columns. "
        f"Saved {OUTPUT_PATH.name}: {len(df):,} rows, {df.shape[1]} columns"
    )


if __name__ == "__main__":
    main()
