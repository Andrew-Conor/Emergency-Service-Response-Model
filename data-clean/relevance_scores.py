from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif
from load_data import load_data

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = (
    PROJECT_ROOT
    / "csvs"
    / "Categorical-Redundancy-Removed-Fire-and-EMS-Response-Data.csv"
)
FIGURES_DIR = PROJECT_ROOT / "data-clean" / "figures"

TARGET = "outcome"
SHUFFLES = 5  # chance level = average score over this many shuffled outcomes
BINS = 10  # numeric columns are split into this many equal-frequency bins

# Columns not scored
LEAKAGE_COLUMNS = []


def prepare(df):
    """Rows with a known outcome only. Missing values are filled for scoring only"""
    df = df[df[TARGET].notna()].copy()
    if "zipcode_of_incident" in df.columns:  # '94103.0' and '94103' are the same zip
        df["zipcode_of_incident"] = df["zipcode_of_incident"].str.replace(
            r"\.0$", "", regex=True
        )

    features = df.drop(columns=[TARGET])
    is_categorical = [
        not pd.api.types.is_numeric_dtype(features[c]) for c in features.columns
    ]

    # Numeric columns are put into equal-frequency bins (textbook: discretization by
    # binning), so every column is scored by the same counting method
    encoded = pd.DataFrame(index=features.index)
    for column, categorical in zip(features.columns, is_categorical):
        values = features[column]
        if not categorical:
            values = pd.qcut(values, q=BINS, duplicates="drop").astype(str)
        encoded[column] = values.fillna("missing").astype("category").cat.codes
    return encoded, is_categorical, df[TARGET].astype(int).to_numpy()


def mutual_information(encoded, is_categorical, y):
    return mutual_info_classif(encoded, y, discrete_features=True)


def score(encoded, is_categorical, y):
    real = mutual_information(encoded, is_categorical, y)
    rng = np.random.default_rng(1)
    chance = np.mean(
        [
            mutual_information(encoded, is_categorical, rng.permutation(y))
            for _ in range(SHUFFLES)
        ],
        axis=0,
    )
    table = pd.DataFrame({"mi": real, "chance": chance}, index=encoded.columns)
    table["relevance"] = (table["mi"] - table["chance"]).clip(lower=0)
    table["type"] = ["categorical" if c else "numeric" for c in is_categorical]
    return table.sort_values("relevance", ascending=False)


def save_chart(table, path):
    table = table.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * len(table))))
    ax.barh(table.index, table["relevance"], color="#2a6fb0", height=0.7)
    ax.set_xlabel("Mutual information with outcome, above chance")
    ax.set_title("Relevance of each column to missing the 8:59 target")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\nSaved chart: {path.relative_to(PROJECT_ROOT)}")


def main():
    df = load_data(INPUT_PATH)
    encoded, is_categorical, y = prepare(df)
    print(
        f"Scoring {encoded.shape[1]} columns on {len(y):,} rows with a known outcome\n"
    )

    table = score(encoded, is_categorical, y)
    print(table.round(4).to_string())

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    save_chart(table, FIGURES_DIR / "relevance_scores.png")


if __name__ == "__main__":
    main()
