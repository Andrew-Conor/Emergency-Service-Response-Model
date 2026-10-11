from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from load_data import load_data

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = (
    PROJECT_ROOT / "csvs" / "First-Response-Only-Fire-and-EMS-Response-Data.csv"
)
FIGURES_DIR = PROJECT_ROOT / "data-clean" / "figures"

THRESHOLD = 0.80
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


def save_heatmap(corr, path):
    fig, ax = plt.subplots(figsize=(10, 9))
    image = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=90)
    ax.set_yticks(range(len(corr)), corr.columns)
    fig.colorbar(image, ax=ax, fraction=0.046, label="Pearson's r")
    ax.set_title("Correlation between numeric columns")
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
    FIGURES_DIR.mkdir(exist_ok=True)

    corr = correlation_matrix(df)
    save_heatmap(corr, FIGURES_DIR / "correlation_heatmap.png")

    print(f"Pairs with |r| >= {THRESHOLD}:")
    for a, b, r in redundant_pairs(corr):
        print(f"  {r:+.3f}  {a}  <->  {b}")

    for x, y in SCATTER_PAIRS:
        save_scatter(df, x, y, FIGURES_DIR / f"scatter_{x}_vs_{y}.png")

    print(f"\nFigures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
