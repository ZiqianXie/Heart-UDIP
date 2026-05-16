#!/usr/bin/env python3
"""
example_perdi_map.py
--------------------
Demonstrates how to compute a variant-specific PerDI (decoder perturbation)
map and save it as a NIfTI file.

This script assumes ``example_compute_weights.py`` has already been run so
that the SNP weight vectors are available.

Steps covered
  1. Load the trained autoencoder.
  2. For each target variant, load its precomputed weight vector w.
  3. Sample subjects with both genotype dosages and imaging data.
  4. Run the two-pass PerDI algorithm:
       Pass 1 — estimate perturbation scale α = SD(ŵ · latent_i).
       Pass 2 — accumulate genotype-weighted decoder difference maps.
  5. Save the PerDI map as a signed float32 NIfTI file.
  6. Print convergence statistics.

Usage
-----
  python example_perdi_map.py [--snp rsXXXXXXX]

All paths in this script are intentional placeholders.
Do NOT hard-code real data paths or subject IDs here.
"""

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --- add repo root to sys.path ---
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from PerDI.model import load_model, select_gpu
from PerDI.core  import compute_variant_perdi_map, save_perdi_nifti

# ================================================================
# CONFIGURATION — replace all placeholder paths before running
# ================================================================
MODEL_CKPT      = Path("/path/to/model_epoch_10.pth")
IMAGING_DIR     = Path("/path/to/registered_volumes/")
IMG_GLOB        = "*.nii.gz"              # filename pattern inside IMAGING_DIR
WEIGHTS_NPZ     = Path("/path/to/output/example_snp_weights.npz")
DOSAGES_NPZ     = Path("/path/to/output/example_dosages.npz")  # {rsid → (N,) dosages}
SAMPLED_IIDS_NPZ = Path("/path/to/output/example_sampled_iids.npz")  # {rsid → (N,) IIDs}
OUTPUT_DIR      = Path("/path/to/output/perturbation_maps/")
N_SAMPLE        = 500       # subjects to use per SNP
CONV_CHECKPOINT = 450       # convergence check point
# ================================================================


def build_imaging_index(imaging_dir: Path, pattern: str = "*.nii.gz") -> dict[str, Path]:
    """
    Build a mapping from subject IID to NIfTI file path.

    Expects filenames of the form  *_{IID}_*.nii.gz  where IID is a
    numeric or alphanumeric identifier.  Adjust the IID extraction
    logic below to match your file-naming convention.
    """
    index: dict[str, Path] = {}
    for p in imaging_dir.glob(pattern):
        # Example filename: registered_cropped_1000001_20208_2_0.nii.gz
        # IID = part after the second underscore before the study code.
        # Adjust this to your filename convention.
        parts = p.stem.split("_")
        # Heuristic: the IID is the first purely numeric token of ≥ 6 chars.
        for part in parts:
            if part.isdigit() and len(part) >= 6:
                index[part] = p
                break
    return index


def main(target_snp: str | None = None) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- load model ----
    device = select_gpu()
    print(f"Loading model from {MODEL_CKPT} on {device} …")
    model  = load_model(MODEL_CKPT, device)

    # ---- load precomputed weights ----
    wt_data  = np.load(str(WEIGHTS_NPZ),       allow_pickle=True)
    iid_data = np.load(str(SAMPLED_IIDS_NPZ),  allow_pickle=True)
    dos_data = np.load(str(DOSAGES_NPZ),        allow_pickle=True)

    rsids = [target_snp] if target_snp else list(wt_data.files)
    if not rsids:
        print("No SNPs found.  Check WEIGHTS_NPZ path.")
        return

    # ---- build imaging index once ----
    print(f"Scanning {IMAGING_DIR} for images …")
    img_idx = build_imaging_index(IMAGING_DIR, IMG_GLOB)
    print(f"  {len(img_idx)} subjects in index.")

    conv_records: list[dict] = []

    for rsid in rsids:
        out_path = OUTPUT_DIR / f"{rsid}.nii.gz"
        if out_path.exists():
            print(f"  [SKIP] {rsid}: output already exists.")
            continue

        if rsid not in wt_data.files:
            print(f"  [WARN] {rsid}: no weight vector — skipping.")
            continue
        if rsid not in iid_data.files or rsid not in dos_data.files:
            print(f"  [WARN] {rsid}: no sampled IIDs / dosages — skipping.")
            continue

        w         = wt_data[rsid].astype(np.float32)     # (256,)
        iids      = iid_data[rsid]                        # (N,) str
        dosages   = dos_data[rsid].astype(np.float32)     # (N,)

        # resolve image paths from IIDs
        img_paths = [img_idx[str(iid)] for iid in iids if str(iid) in img_idx]
        matched_dosages = np.array(
            [dosages[i] for i, iid in enumerate(iids) if str(iid) in img_idx],
            dtype=np.float32,
        )

        if len(img_paths) < 10:
            print(f"  [WARN] {rsid}: only {len(img_paths)} matched images — skipping.")
            continue

        print(f"\n  Processing {rsid}  ({len(img_paths)} subjects) …")

        try:
            perdi_map, affine, meta = compute_variant_perdi_map(
                model=model,
                img_paths=img_paths,
                dosages=matched_dosages,
                w=w,
                device=device,
                conv_checkpoint=CONV_CHECKPOINT,
            )
        except RuntimeError as e:
            print(f"  [ERROR] {rsid}: {e}")
            continue

        save_perdi_nifti(perdi_map, affine, out_path)

        status = "CONVERGED" if meta["converged"] else "not converged"
        print(
            f"  {rsid}: n={meta['n_loaded']}, α={meta['alpha']:.4f}, "
            f"r={meta['pearson_r']:.4f} [{status}]  → {out_path.name}"
        )
        conv_records.append({"rsid": rsid, **meta})

    # ---- save convergence table ----
    if conv_records:
        conv_df  = pd.DataFrame(conv_records)
        conv_csv = OUTPUT_DIR.parent / "perturbation_convergence.csv"
        conv_df.to_csv(str(conv_csv), index=False)
        n_conv   = int(conv_df["converged"].sum())
        print(f"\nConvergence: {n_conv}/{len(conv_df)} SNPs converged (r > 0.70)")
        print(f"Saved convergence table → {conv_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute PerDI maps for GWAS variants.")
    parser.add_argument(
        "--snp", type=str, default=None,
        help="Single rsid to process.  Omit to process all SNPs in WEIGHTS_NPZ.",
    )
    args = parser.parse_args()
    main(target_snp=args.snp)
