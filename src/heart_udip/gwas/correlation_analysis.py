import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def mean_sd_abs_corr(corr_matrix: pd.DataFrame) -> tuple[float, float]:
    abs_corr = corr_matrix.abs()
    upper_triangle = abs_corr.where(np.triu(np.ones(abs_corr.shape), k=1).astype(bool))
    values = upper_triangle.stack().values
    return float(values.mean()), float(values.std())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize pairwise latent-feature correlations.")
    parser.add_argument("--input-csv", required=True, help="Latent feature CSV.")
    parser.add_argument("--output-heatmap", default=None, help="Optional PNG/PDF path for the heatmap.")
    parser.add_argument("--title", default="Correlation matrix of latent features")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    df = pd.read_csv(args.input_csv)
    feature_columns = [column for column in df.columns if column.lower().startswith("feature_")]
    correlation_matrix = df[feature_columns].corr(method="pearson")

    mean_corr, std_corr = mean_sd_abs_corr(correlation_matrix)
    print(f"Mean absolute correlation: {mean_corr:.4f}")
    print(f"Standard deviation of absolute correlation: {std_corr:.4f}")

    plt.figure(figsize=(16, 14))
    sns.heatmap(
        correlation_matrix,
        cmap="coolwarm",
        center=0,
        square=True,
        xticklabels=50,
        yticklabels=50,
        cbar_kws={"shrink": 0.5},
    )
    plt.title(args.title)
    plt.xlabel("Feature index")
    plt.ylabel("Feature index")
    plt.tight_layout()

    if args.output_heatmap:
        output_path = Path(args.output_heatmap)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300)
    else:
        plt.show()


if __name__ == "__main__":
    main()
