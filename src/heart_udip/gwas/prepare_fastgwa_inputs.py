import argparse
import subprocess
from pathlib import Path

import pandas as pd


def canonicalize_phenotypes(feature_csv: Path) -> pd.DataFrame:
    table = pd.read_csv(feature_csv)
    if "Patient_ID" in table.columns:
        table = table.rename(columns={"Patient_ID": "FID"})
    elif "patient_id" in table.columns:
        table = table.rename(columns={"patient_id": "FID"})
    else:
        raise ValueError("Feature CSV must contain `Patient_ID` or `patient_id`.")

    table = table[~table["FID"].duplicated(keep=False)].copy()
    table["IID"] = table["FID"]
    return table


def write_fastgwa_phenotypes(table: pd.DataFrame, output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_columns = [column for column in table.columns if column.lower().startswith("feature_")]
    for feature_name in feature_columns:
        table[["FID", "IID", feature_name]].to_csv(output_dir / feature_name, sep=" ", index=False)
    return feature_columns


def run_fastgwa(feature_columns: list[str], args: argparse.Namespace) -> None:
    results_dir = Path(args.fastgwa_output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    for feature_name in feature_columns:
        cmd = [
            args.fastgwa_binary,
            "--maf",
            str(args.maf),
            "--bfile",
            args.bfile,
            "--grm-sparse",
            args.grm_sparse,
            "--fastGWA-mlm",
            "--pheno",
            str(Path(args.pheno_dir) / feature_name),
            "--covar",
            args.covar,
            "--qcovar",
            args.qcovar,
            "--thread-num",
            str(args.threads),
            "--seed",
            str(args.seed),
            "--out",
            str(results_dir / feature_name),
        ]
        subprocess.run(cmd, check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare fastGWA phenotype files and optionally launch GCTA.")
    parser.add_argument("--feature-csv", required=True, help="CSV generated from latent feature extraction.")
    parser.add_argument("--pheno-dir", required=True, help="Output directory for one-feature-per-file phenotypes.")
    parser.add_argument("--fastgwa-binary", default=None, help="Optional path to the GCTA binary.")
    parser.add_argument("--fastgwa-output-dir", default=None, help="Optional directory for fastGWA outputs.")
    parser.add_argument("--bfile", default=None)
    parser.add_argument("--grm-sparse", default=None)
    parser.add_argument("--covar", default=None)
    parser.add_argument("--qcovar", default=None)
    parser.add_argument("--maf", type=float, default=0.01)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    table = canonicalize_phenotypes(Path(args.feature_csv))
    feature_columns = write_fastgwa_phenotypes(table, Path(args.pheno_dir))
    print(f"Wrote {len(feature_columns)} phenotype files to {args.pheno_dir}")

    if args.fastgwa_binary:
        required = ["fastgwa_output_dir", "bfile", "grm_sparse", "covar", "qcovar"]
        missing = [name for name in required if getattr(args, name) is None]
        if missing:
            raise ValueError(f"Missing arguments for fastGWA execution: {', '.join(missing)}")
        run_fastgwa(feature_columns, args)


if __name__ == "__main__":
    main()





