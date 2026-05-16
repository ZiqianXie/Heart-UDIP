#!/usr/bin/env python3
"""
example_compute_weights.py
--------------------------
Demonstrates how to compute variant-specific PerDI weights w = R⁻¹z.

Steps covered
  1. Load covariate-residualised latent features (N × 256 array).
  2. Compute the phenotypic correlation matrix R.
  3. For each variant of interest, extract GWAS z-scores from summary-statistic
     files and solve  w = (R + λI)⁻¹ z.
  4. Save the weights and z-score vectors.

Usage
-----
  # Adjust the placeholder paths at the top of this script, then:
  python example_compute_weights.py

All paths in this script are intentional placeholders.
Do NOT hard-code real data paths here.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.linalg import solve
from tqdm import tqdm

# --- add the repo root to sys.path so `PerDI` can be imported ---
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from PerDI.core import compute_phenotypic_correlation_matrix, compute_variant_weights

# ================================================================
# CONFIGURATION — replace all placeholder paths before running
# ================================================================
RESIDUALS_NPY   = Path("/path/to/output/residuals.npy")        # (N, 256) float32
NOVEL_SNPS_TXT  = Path("/path/to/novel_snps.txt")              # rsid per line (or space-sep)
SUMSTAT_DIR     = Path("/path/to/gwas/duin_sumstats/")         # Feature_*.fastGWA files
OUTPUT_DIR      = Path("/path/to/output")
TAG             = "example"                                     # prefix for output file names
N_FEATURES      = 256
RIDGE           = 1e-4
# ================================================================


def load_novel_rsids(snp_file: Path) -> list[str]:
    """
    Read rsids from a plain text file.
    Accepts one rsid per line or rows with multiple space-separated fields
    where the rsid is in column index 3 (0-based).
    """
    rsids = []
    with open(snp_file) as fh:
        for line in fh:
            parts = line.strip().split()
            if not parts:
                continue
            # heuristic: use column 3 if it starts with "rs", else column 0
            rsid = parts[3] if (len(parts) > 3 and parts[3].startswith("rs")) else parts[0]
            if rsid.startswith("rs"):
                rsids.append(rsid)
    # deduplicate while preserving order
    seen: set[str] = set()
    return [r for r in rsids if not (r in seen or seen.add(r))]  # type: ignore[func-returns-value]


def grep_sumstats(
    sumstat_dir: Path,
    rsids: list[str],
    n_features: int = N_FEATURES,
) -> pd.DataFrame:
    """
    Extract GWAS z-scores for ``rsids`` from all Feature_*.fastGWA files.

    Returns a DataFrame with columns: rsid, feature_idx, beta, se, af1.
    Uses a fast subprocess grep; falls back to pandas scan if needed.
    """
    import subprocess
    from multiprocessing import Pool, cpu_count

    rsid_set   = set(rsids)
    gwa_files  = sorted(sumstat_dir.glob("Feature_*.fastGWA"),
                        key=lambda p: int(p.stem.split("_")[1]))
    if not gwa_files:
        raise FileNotFoundError(f"No Feature_*.fastGWA files found in {sumstat_dir}")
    if len(gwa_files) != n_features:
        print(f"[WARN] found {len(gwa_files)} sumstat files, expected {n_features}")

    # write rsids to a temp file for grep -f
    tmp_rsid_file = OUTPUT_DIR / "_tmp_rsids.txt"
    tmp_rsid_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_rsid_file.write_text("\n".join(rsids) + "\n")

    def _grep_one(args):
        fi, fpath_str = args
        result = subprocess.run(
            ["grep", "-F", "-f", str(tmp_rsid_file), fpath_str],
            capture_output=True, text=True,
        )
        if result.returncode not in (0, 1):
            return []
        records = []
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) < 9:
                continue
            rsid_found = parts[1]
            if rsid_found not in rsid_set:
                continue
            try:
                records.append({
                    "rsid":        rsid_found,
                    "feature_idx": fi,
                    "beta":        float(parts[7]),
                    "se":          float(parts[8]),
                    "af1":         float(parts[6]),
                })
            except ValueError:
                pass
        return records

    workers = min(32, cpu_count())
    all_records: list[dict] = []
    with Pool(workers) as pool:
        for batch in tqdm(
            pool.imap_unordered(_grep_one, [(fi, str(fp)) for fi, fp in enumerate(gwa_files)]),
            total=len(gwa_files), desc="grep sumstats",
        ):
            all_records.extend(batch)

    tmp_rsid_file.unlink(missing_ok=True)
    return pd.DataFrame(all_records)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 1. Load residuals and compute R ----
    print(f"Loading residuals from {RESIDUALS_NPY} …")
    residuals = np.load(str(RESIDUALS_NPY))           # (N, 256)
    print(f"  shape: {residuals.shape}")

    print("Computing phenotypic correlation matrix R …")
    R = compute_phenotypic_correlation_matrix(residuals, ridge=RIDGE)
    print(f"  R shape: {R.shape}  (ridge={RIDGE})")

    # ---- 2. Load novel SNPs ----
    rsids = load_novel_rsids(NOVEL_SNPS_TXT)
    print(f"Novel SNPs to process: {len(rsids)}")

    # ---- 3. Extract z-scores ----
    print("Extracting GWAS z-scores …")
    df = grep_sumstats(SUMSTAT_DIR, rsids)
    df["z"] = df["beta"] / df["se"]
    df["z"]  = df["z"].replace([float("inf"), float("-inf")], float("nan"))

    # ---- 4. Compute weights per SNP ----
    weights_dict:  dict[str, np.ndarray] = {}
    zscores_dict:  dict[str, np.ndarray] = {}
    af_records:    list[dict] = []

    for rsid in tqdm(rsids, desc="compute w=R⁻¹z"):
        sub = df[df["rsid"] == rsid]
        if sub.empty:
            print(f"  [WARN] {rsid}: no z-score entries found, skipping.")
            continue

        z_vec = np.zeros(N_FEATURES, dtype=np.float64)
        af_vals: list[float] = []
        for _, row in sub.iterrows():
            fi = int(row["feature_idx"])
            if 0 <= fi < N_FEATURES and not np.isnan(row["z"]):
                z_vec[fi] = row["z"]
            af_vals.append(row["af1"])

        pop_af = float(np.nanmedian(af_vals)) if af_vals else float("nan")
        af_records.append({
            "rsid": rsid,
            "pop_af": pop_af,
            "n_features_found": int(np.count_nonzero(z_vec)),
        })

        w = compute_variant_weights(R, z_vec, ridge=0.0)   # ridge already in R
        weights_dict[rsid] = w
        zscores_dict[rsid] = z_vec.astype(np.float32)

    # ---- 5. Save ----
    out_weights = OUTPUT_DIR / f"{TAG}_snp_weights.npz"
    out_zscores = OUTPUT_DIR / f"{TAG}_snp_zscores.npz"
    out_af      = OUTPUT_DIR / f"{TAG}_snp_af.csv"

    np.savez(str(out_weights), **weights_dict)
    np.savez(str(out_zscores), **zscores_dict)
    pd.DataFrame(af_records).to_csv(str(out_af), index=False)

    print(f"\nSaved weights for {len(weights_dict)} SNPs → {out_weights}")
    print(f"Saved z-scores → {out_zscores}")
    print(f"Saved AF table → {out_af}")


if __name__ == "__main__":
    main()
