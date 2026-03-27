import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from scipy.stats import gaussian_kde
from sklearn.preprocessing import StandardScaler


def get_density(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    xy = np.vstack([x, y])
    return gaussian_kde(xy)(xy)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute a 2D UMAP embedding of latent phenotypes.")
    parser.add_argument("--input-csv", required=True, help="Latent feature CSV.")
    parser.add_argument("--output-plot", default=None, help="Optional path for the UMAP figure.")
    parser.add_argument("--neighbors", type=int, default=30)
    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--title", default="UMAP visualization of latent features")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    df = pd.read_csv(args.input_csv)
    id_column = "patient_id" if "patient_id" in df.columns else df.columns[0]
    feature_columns = [column for column in df.columns if column.lower().startswith("feature_")]
    features = df[feature_columns].values

    features_scaled = StandardScaler().fit_transform(features)
    embedding = umap.UMAP(
        n_components=2,
        random_state=args.seed,
        n_neighbors=args.neighbors,
        min_dist=args.min_dist,
    ).fit_transform(features_scaled)

    density = get_density(embedding[:, 0], embedding[:, 1])
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=density, cmap="Spectral", alpha=0.6, s=4)
    plt.colorbar(scatter, label="Density")
    plt.title(args.title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.tight_layout()

    if args.output_plot:
        output_path = Path(args.output_plot)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300)
    else:
        plt.show()

    preview = pd.DataFrame(
        {
            id_column: df[id_column],
            "umap_1": embedding[:, 0],
            "umap_2": embedding[:, 1],
        }
    )
    print(preview.head().to_string(index=False))


if __name__ == "__main__":
    main()
