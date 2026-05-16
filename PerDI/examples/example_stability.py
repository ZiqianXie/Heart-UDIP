#!/usr/bin/env python3
"""
example_stability.py
--------------------
Demonstrates how to run the four PerDI stability experiments (A, B, D, E)
on a random sample of subjects using two independently trained model folds.

Experiments
  A — Scale linearity:      Pearson r between α-scaled and reference maps.
  B — Cross-subject:        Mean pairwise r across subjects for each latent dim.
  D — Hessian ratio:        ‖H_k‖ / ‖J_k‖  (second-order non-linearity).
  E — Cross-fold Jacobian:  CCA between Jacobian column spaces of fold-1 and fold-2.

Expected results (from the Heart-UDIP study, N=500 subjects)
  Exp A (α=2):  mean r ≈ 0.999  — decoder is in a linear regime
  Exp B:        mean pairwise r ≈ 0.89  — maps are subject-independent
  Exp D:        mean ratio ≈ 0.08  — locally linear decoder
  Exp E:        mean canonical r ≈ 0.67  — folds span similar image subspaces

Usage
-----
  python example_stability.py [--tag 2ch] [--n-subjects 500]

All paths in this script are intentional placeholders.
Do NOT hard-code real data paths or subject IDs here.
"""

import argparse
import random
import sys
from pathlib import Path

# --- add repo root to sys.path ---
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from PerDI.model     import load_model, select_gpu
from PerDI.stability import run_stability, save_stability_results

# ================================================================
# CONFIGURATION — replace all placeholder paths before running
# ================================================================
CONFIGS = {
    "2ch": {
        "model_ckpt":       Path("/path/to/2ch/model_epoch_10.pth"),
        "fold2_ckpt":       Path("/path/to/2ch/fold2_model_epoch_10.pth"),
        "imaging_dir":      Path("/path/to/registered_2ch_volumes/"),
        "img_glob":         "*.nii.gz",
    },
    "4ch": {
        "model_ckpt":       Path("/path/to/4ch/model_epoch_10.pth"),
        "fold2_ckpt":       Path("/path/to/4ch/fold2_model_epoch_10.pth"),
        "imaging_dir":      Path("/path/to/registered_4ch_volumes/"),
        "img_glob":         "*.nii.gz",
    },
}
OUTPUT_DIR = Path("/path/to/output/stability/")
# ================================================================


def collect_img_paths(imaging_dir: Path, glob_pattern: str) -> list[Path]:
    paths = sorted(imaging_dir.glob(glob_pattern))
    if not paths:
        raise FileNotFoundError(
            f"No files matching '{glob_pattern}' in {imaging_dir}"
        )
    return paths


def main(tag: str = "2ch", n_subjects: int = 500, seed: int = 42) -> None:
    cfg    = CONFIGS[tag]
    device = select_gpu()
    print(f"\n=== PerDI Stability Experiments [{tag}] ===")
    print(f"Device : {device}")

    # ---- load both model folds ----
    print(f"Loading fold-1 model from {cfg['model_ckpt']} …")
    model1 = load_model(cfg["model_ckpt"],  device)
    print(f"Loading fold-2 model from {cfg['fold2_ckpt']} …")
    model2 = load_model(cfg["fold2_ckpt"], device)

    # ---- collect image paths ----
    print(f"Scanning {cfg['imaging_dir']} …")
    all_paths = collect_img_paths(cfg["imaging_dir"], cfg["img_glob"])
    print(f"  {len(all_paths)} images found.")

    # ---- run all four experiments ----
    results = run_stability(
        model1      = model1,
        model2      = model2,
        img_paths   = all_paths,
        device      = device,
        n_subjects  = n_subjects,
        seed        = seed,
    )

    # ---- save results ----
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_stability_results(results, OUTPUT_DIR, tag)

    # ---- concise printout ----
    import numpy as np
    print("\n--- Quick summary ---")
    r_scale = results["r_scale"]
    alphas  = results["alphas"]
    ref_idx = results["ref_alpha_idx"]
    for ai, a in enumerate(alphas):
        if a == 1.0:
            continue
        vals = r_scale[ai]
        print(
            f"  Exp A  α={a:+.1f}  mean r={float(np.nanmean(vals)):.4f}  "
            f"5th={float(np.nanpercentile(vals, 5)):.4f}"
        )
    print(f"  Exp B  mean pairwise r = {float(np.nanmean(results['r_cross_subj'])):.4f}")
    print(f"  Exp D  mean Hessian ratio = {float(np.nanmean(results['hess_ratio'])):.4f}")
    print(f"  Exp E  mean Jacobian CCA  = {float(np.nanmean(results['cos_sim_jacob'])):.4f}")
    print(f"\nFull results saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run PerDI stability experiments."
    )
    parser.add_argument(
        "--tag", choices=list(CONFIGS.keys()), default="2ch",
        help="Which view to analyse (default: 2ch).",
    )
    parser.add_argument(
        "--n-subjects", type=int, default=500,
        help="Number of subjects to sample (default: 500).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for subject sampling (default: 42).",
    )
    args = parser.parse_args()
    main(tag=args.tag, n_subjects=args.n_subjects, seed=args.seed)
