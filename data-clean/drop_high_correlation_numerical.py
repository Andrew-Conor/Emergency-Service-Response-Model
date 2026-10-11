from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from load_data import load_data

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = (
    PROJECT_ROOT / "csvs" / "First-Response-Only-Fire-and-EMS-Response-Data.csv"
)
OUTPUT_PATH = (
    PROJECT_ROOT / "csvs" / "Numeric-Redundancy-Removed-Fire-and-EMS-Response-Data.csv"
)
FIGURES_DIR = PROJECT_ROOT / "data-clean" / "figures"

THRESHOLD = 0.8
NUMERIC_REDUNDANT = [
    "global_tilted_irradiance",
    "et0_fao_evapotranspiration",
    "direct_radiation",
    "direct_normal_irradiance",
    "rain",
    "apparent_temperature",
    "wind_speed_100m",
    "wind_gusts_10m",
    "vapour_pressure_deficit",
]
EXCLUDE = [
    "outcome",
    "weather_code",
    "is_weekend",
    "is_holiday",
    "unit_sequence_in_call_dispatch",
]
SCATTER_PAIRS = [
    ("precipitation", "rain"),
    ("temperature_2m", "wind_speed_10m"),
]


def correlation_matrix(df):
    numeric = df.select_dtypes("number").drop(columns=EXCLUDE, errors="ignore")
    return numeric.corr(method="pearson")


def redundant_pairs(corr):
    pairs = [
        (a, b, corr.loc[a, b])
        for i, a in enumerate(corr.columns)
        for b in corr.columns[i + 1 :]
        if abs(corr.loc[a, b]) >= THRESHOLD
    ]
    return sorted(pairs, key=lambda p: -abs(p[2]))


def print_pairs(corr, label):
    print(f"\n{label}: pairs with |r| >= {THRESHOLD}")
    pairs = redundant_pairs(corr)
    for a, b, r in pairs:
        print(f"  {r:+.3f}  {a}  <->  {b}")
    if not pairs:
        print("  None")


def save_heatmap(corr, path, title):
    size = max(8, 0.5 * len(corr))
    fig, ax = plt.subplots(figsize=(size + 2, size))
    image = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=90)
    ax.set_yticks(range(len(corr)), corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            value = corr.iat[i, j] + 0.0
            ax.text(
                j, i, f"{value:.2f}", ha="center", va="center", fontsize=6,
                color="white" if abs(value) > 0.6 else "black",
            )
    fig.colorbar(image, ax=ax, fraction=0.046, label="Pearson's r")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_scatter(df, x, y, path):
    data = df[[x, y]].dropna()
    r = data[x].corr(data[y])
    sample = data.sample(min(5000, len(data)), random_state=1)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(sample[x], sample[y], s=8, alpha=0.3, color="#2a6fb0")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"{x} vs {y}\nr = {r:.2f}")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    df = load_data(INPUT_PATH)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    before = correlation_matrix(df)
    save_heatmap(
        before,
        FIGURES_DIR / "correlation_heatmap_before.png",
        f"Before: {len(before)} numeric columns",
    )
    print_pairs(before, "Before")

    for x, y in SCATTER_PAIRS:
        save_scatter(df, x, y, FIGURES_DIR / f"scatter_{x}_vs_{y}.png")

    df = df.drop(columns=NUMERIC_REDUNDANT)

    after = correlation_matrix(df)
    save_heatmap(
        after,
        FIGURES_DIR / "correlation_heatmap_after.png",
        f"After: {len(after)} numeric columns",
    )
    print_pairs(after, "After")

    df.to_csv(OUTPUT_PATH, index=False)
    print(
        f"\nDropped {len(NUMERIC_REDUNDANT)} columns. "
        f"Saved {OUTPUT_PATH.name}: {len(df):,} rows, {df.shape[1]} columns"
    )
    print(f"Figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()