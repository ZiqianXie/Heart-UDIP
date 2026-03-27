import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def combine_minp(gwas_files: list[Path], p_column: str) -> pd.DataFrame:
    min_table = None
    for gwas_file in gwas_files:
        table = pd.read_table(gwas_file)
        if p_column not in table.columns:
            raise ValueError(f"{gwas_file} does not contain p-value column `{p_column}`.")
        subset = table[["CHR", "SNP", "POS", "A1", "A2", "AF1", "N", p_column]].copy()
        subset = subset.rename(columns={p_column: f"P__{gwas_file.stem}"})
        min_table = subset if min_table is None else min_table.merge(
            subset,
            on=["CHR", "SNP", "POS", "A1", "A2", "AF1", "N"],
            how="inner",
        )

    pvalue_columns = [column for column in min_table.columns if column.startswith("P__")]
    min_table["P"] = min_table[pvalue_columns].min(axis=1)
    return min_table[["CHR", "SNP", "POS", "A1", "A2", "AF1", "N", "P"]]


def plot_manhattan(table: pd.DataFrame, output_path: Path | None, threshold: float, title: str) -> None:
    table = table.copy()
    table["minus_log10_p"] = -np.log10(table["P"])
    table["ind"] = range(len(table))
    grouped = table.groupby("CHR")

    fig, ax = plt.subplots(figsize=(25, 15))
    colors = ["#0000ff", "#4682b4"]
    x_labels = []
    x_positions = []
    for index, (chromosome, group) in enumerate(grouped):
        group.plot(kind="scatter", x="ind", y="minus_log10_p", color=colors[index % len(colors)], ax=ax, s=6)
        x_labels.append(chromosome)
        x_positions.append((group["ind"].iloc[0] + group["ind"].iloc[-1]) / 2)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=90, fontsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.axhline(y=-np.log10(threshold), color="grey", linestyle="--")
    ax.set_xlabel("Chromosome", fontsize=18)
    ax.set_ylabel("-log10(p-value)", fontsize=18)
    ax.set_title(title, fontsize=20)
    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300)
    else:
        plt.show()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate multiple fastGWA outputs with a min-P strategy.")
    parser.add_argument("--gwas-dir", required=True, help="Directory containing fastGWA result files.")
    parser.add_argument("--pattern", default="*.fastGWA", help="Glob pattern for GWAS result files.")
    parser.add_argument("--output-tsv", required=True, help="Path for the combined min-P table.")
    parser.add_argument("--p-column", default="P")
    parser.add_argument("--threshold", type=float, default=5e-8)
    parser.add_argument("--output-manhattan", default=None, help="Optional path for Manhattan plot.")
    parser.add_argument("--title", default="Min-P Manhattan plot")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    gwas_files = sorted(Path(args.gwas_dir).glob(args.pattern))
    if not gwas_files:
        raise ValueError("No GWAS files matched the requested pattern.")

    combined = combine_minp(gwas_files, p_column=args.p_column)
    output_tsv = Path(args.output_tsv)
    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_tsv, sep="\t", index=False)
    significant_hits = combined.loc[combined["P"] < args.threshold]
    print(f"Saved min-P results to {output_tsv}")
    print(f"Variants below {args.threshold}: {len(significant_hits)}")

    plot_manhattan(
        combined,
        output_path=Path(args.output_manhattan) if args.output_manhattan else None,
        threshold=args.threshold,
        title=args.title,
    )


if __name__ == "__main__":
    main()
