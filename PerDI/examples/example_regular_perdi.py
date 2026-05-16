#!/usr/bin/env python3
"""
example_regular_perdi.py
------------------------
Demonstrates how to compute regular PerDI maps — one map per latent dimension,
with no GWAS data required.

Regular PerDI perturbs each of the D latent dimensions independently by its
natural cross-subject standard deviation σ_k, then averages the resulting
decoder difference maps across subjects:

    diff_i_k = Decoder(enc2_i, latent_i + σ_k · e_k) − Decoder(enc2_i, latent_i)
    PerDI_k  = (1/N) Σ_i  diff_i_k

This produces a generative atlas of the decoder: 256 maps that together
describe every mode of variation the model has learned to represent.

Variant-specific PerDI then produces a single composite map by weighting
these dimensions with the GWAS-derived vector w = R⁻¹z.

Options
-------
--mode fast   Compute all 256 maps in a single batched pass (recommended).
--mode loop   Compute maps dimension by dimension (lower GPU memory).

Usage
-----
  python example_regular_perdi.py --mode fast [--n-subjects 300]

All paths in this script are intentional placeholders.
Do NOT hard-code real data paths or subject IDs here.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import nibabel as nib

# --- add repo root to sys.path ---
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from PerDI.model import load_model, select_gpu
from PerDI.core  import (
    estimate_latent_sigmas,
    compute_dimension_perdi_map,
    compute_all_dimension_perdi_maps,
    save_perdi_nifti,
)

# ================================================================
# CONFIGURATION — replace all placeholder paths before running
# ================================================================
MODEL_CKPT   = Path("/path/to/model_epoch_10.pth")
IMAGING_DIR  = Path("/path/to/registered_volumes/")
IMG_GLOB     = "*.nii.gz"
OUTPUT_DIR   = Path("/path/to/output/regular_perdi/")
N_SUBJECTS   = 300          # subjects for sigma estimation and map averaging
DIM_BATCH    = 32           # GPU batch size (lower if OOM)
# ================================================================


def collect_img_paths(imaging_dir: Path, glob: str, n: int | None = None) -> list[Path]:
    paths = sorted(imaging_dir.glob(glob))
    if not paths:
        raise FileNotFoundError(f"No files matching '{glob}' in {imaging_dir}")
    if n is not None:
        import random
        random.seed(42)
        paths = random.sample(paths, min(n, len(paths)))
    return paths


def main(mode: str = "fast", n_subjects: int = N_SUBJECTS) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = select_gpu()
    print(f"Loading model from {MODEL_CKPT} on {device} …")
    model = load_model(MODEL_CKPT, device)
    latent_dim = model.encoder_fc.out_features
    print(f"  Latent dimension: {latent_dim}")

    print(f"Collecting {n_subjects} subject images from {IMAGING_DIR} …")
    img_paths = collect_img_paths(IMAGING_DIR, IMG_GLOB, n=n_subjects)
    print(f"  {len(img_paths)} images found.")

    # ---- estimate per-dimension SDs ----
    print("Estimating per-dimension latent standard deviations …")
    sigma = estimate_latent_sigmas(model, img_paths, device)
    np.save(str(OUTPUT_DIR / "latent_sigma.npy"), sigma)
    print(f"  σ: min={sigma.min():.4f}  mean={sigma.mean():.4f}  max={sigma.max():.4f}")

    if mode == "fast":
        # ---- single-pass batched computation ----
        print(f"\nComputing all {latent_dim} regular PerDI maps (batched, dim_batch={DIM_BATCH}) …")
        perdi_maps, affine, n_loaded = compute_all_dimension_perdi_maps(
            model, img_paths, sigma, device, dim_batch=DIM_BATCH,
        )
        print(f"  {n_loaded} subjects contributed.")

        # save one NIfTI per dimension
        maps_dir = OUTPUT_DIR / "maps"
        maps_dir.mkdir(exist_ok=True)
        for k in range(latent_dim):
            save_perdi_nifti(perdi_maps[k], affine, maps_dir / f"dim_{k:03d}.nii.gz")
        print(f"  Saved {latent_dim} maps → {maps_dir}/")

        # save the full (D, H, W, Depth) stack as a single NPZ for downstream use
        np.savez_compressed(str(OUTPUT_DIR / "perdi_maps_all.npz"), maps=perdi_maps)
        print(f"  Saved full stack → {OUTPUT_DIR}/perdi_maps_all.npz")

    else:  # mode == "loop"
        # ---- dimension-by-dimension computation (lower memory) ----
        maps_dir = OUTPUT_DIR / "maps"
        maps_dir.mkdir(exist_ok=True)
        print(f"\nComputing {latent_dim} regular PerDI maps one by one …")
        import tqdm
        for k in tqdm.trange(latent_dim, desc="dimensions"):
            out_path = maps_dir / f"dim_{k:03d}.nii.gz"
            if out_path.exists():
                continue
            perdi_map, affine, n_loaded = compute_dimension_perdi_map(
                model, img_paths, dim=k, sigma=sigma, device=device,
            )
            save_perdi_nifti(perdi_map, affine, out_path)
        print(f"  Saved {latent_dim} maps → {maps_dir}/")

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute regular PerDI maps (one per latent dimension)."
    )
    parser.add_argument(
        "--mode", choices=["fast", "loop"], default="fast",
        help="'fast' = batched single pass (recommended); "
             "'loop' = one dimension at a time (lower GPU memory).",
    )
    parser.add_argument(
        "--n-subjects", type=int, default=N_SUBJECTS,
        help=f"Number of subjects (default: {N_SUBJECTS}).",
    )
    args = parser.parse_args()
    main(mode=args.mode, n_subjects=args.n_subjects)
