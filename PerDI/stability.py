"""
stability.py — Stability and reproducibility experiments for PerDI maps.

Addresses the reviewer concern:
  "There is no demonstration of stability, uniqueness, or reproducibility
   of these mappings across resampling, folds, or atlas choices."

Strategy
--------
Instead of validating every variant-specific PerDI map individually, we
perturb each of the 256 latent dimensions INDEPENDENTLY by ±σ_k (the
per-dimension natural standard deviation).  Because PerDI maps are linear
combinations of these per-dimension maps, any stability property that holds
for every individual dimension automatically holds for the variant-level map.

Experiments
-----------
A — Scale linearity
    Compares the perturbation map at scale α to the map at 2α via Pearson r.
    r → 1 means the decoder operates in a locally linear regime and maps
    are uniquely determined up to a scalar factor.

B — Cross-subject consistency
    For each latent dimension, computes the mean pairwise Pearson r across
    subjects' perturbation maps.  High r means the map encodes decoder
    geometry (a property of the model), not individual subject anatomy.

D — Hessian ratio
    Measures second-order non-linearity via:
        H_k = D(z + σ_k e_k) + D(z − σ_k e_k) − 2 D(z)
    The Hessian ratio ‖H_k‖ / ‖J_k‖ ≪ 1 confirms local linearity.
    (Experiment C — per-dimension cross-fold r — is omitted because
     independently trained autoencoders have rotationally ambiguous
     latent spaces, making direct dimension comparison uninformative.
     Experiment E is the correct cross-fold comparison.)

E — Cross-fold Jacobian CCA
    Computes the full (D_voxels × 256) Jacobian for each subject and each
    model fold, then measures how well the two Jacobian column spaces agree
    via canonical correlation analysis (CCA).  High mean canonical r means
    both folds decode the same image-space variation patterns, irrespective
    of latent-space rotation.

All experiments use the same randomly sampled set of subjects.

Outputs
-------
  output/stability/scale_linearity_{tag}.npz   — r_scale  (n_alphas, 256, N)
  output/stability/hessian_{tag}.npz           — hess_ratio (256, N)
  output/stability/cross_subject_{tag}.npz     — r_cross_subj (256,), N_cs (256,)
  output/stability/cross_fold_{tag}.npz        — cos_sim_jacob (N,)
  output/stability/summary_{tag}.txt           — human-readable summary
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Sequence

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .model import CNN3D, load_model, select_gpu

# ------------------------------------------------------------------ #
#  Default hyper-parameters                                            #
# ------------------------------------------------------------------ #

N_SUBJECTS  = 500          # number of subjects to sample
DIM_BATCH   = 32           # latent dims processed per GPU forward pass
JACOB_TOP_K = 10           # canonical correlations to average in Exp E
SEED        = 42

# Scale factors for Exp A.  α=1.0 is the reference (1-SD perturbation);
# α=−1.0 is reused by Exp D (needed for the Hessian numerator).
ALPHAS            = [-1.0, 0.5, 1.0, 1.5, 2.0, 3.0]
REF_ALPHA_IDX     = ALPHAS.index(1.0)    # index 2  (compare everything to this)
NEG_ALPHA_IDX     = ALPHAS.index(-1.0)   # index 0  (used by Hessian)


# ------------------------------------------------------------------ #
#  Image loading                                                       #
# ------------------------------------------------------------------ #

def _load_znorm(path: str | Path, device: torch.device) -> torch.Tensor:
    """Load NIfTI, z-normalise per sample, return (1,1,H,W,D) tensor."""
    img = nib.load(str(path)).get_fdata(dtype=np.float32)
    m, s = float(img.mean()), float(img.std())
    if s <= 0:
        s = 1.0
    return torch.tensor((img - m) / s, dtype=torch.float32,
                        device=device).unsqueeze(0).unsqueeze(0)


# ------------------------------------------------------------------ #
#  CCA utility (Exp E)                                                 #
# ------------------------------------------------------------------ #

def jacobian_cca(J1: np.ndarray, J2: np.ndarray, top_k: int = JACOB_TOP_K) -> float:
    """
    Mean of the top-K canonical correlations between the column spaces of J1 and J2.

    This is the principal-angle–based measure of Jacobian column-space similarity.
    It is automatically sign-flip–invariant because the SVD aligns signs optimally.

    Algorithm (Gram-matrix trick; avoids materialising D×D left singular vectors):

    1.  G11 = J1ᵀ J1,   G12 = J1ᵀ J2,   G22 = J2ᵀ J2   — all (latent_dim, latent_dim)
    2.  Eigendecompose  G_ii = V_i Λ_i V_iᵀ
    3.  C = diag(1/√Λ₁) V₁ᵀ G₁₂ V₂ diag(1/√Λ₂)         — cross-correlation (latent_dim, latent_dim)
        (C[i,j] = ⟨u₁_i, u₂_j⟩ where u are left singular vectors of J_fold)
    4.  SVD(C) → singular values = canonical correlations between column spaces
    5.  Return mean(σ₁, …, σ_K).

    Complexity: O(latent_dim² × D_voxels) — same as one matrix multiply.

    Parameters
    ----------
    J1, J2 : np.ndarray  (D_voxels, latent_dim)
    top_k  : int  — number of canonical correlations to average.

    Returns
    -------
    float  — mean of the top-K canonical correlations, clipped to [0, 1].
    """
    G11 = J1.T @ J1
    G12 = J1.T @ J2
    G22 = J2.T @ J2

    lam1, V1 = np.linalg.eigh(G11)
    lam2, V2 = np.linalg.eigh(G22)
    s1 = np.sqrt(np.clip(lam1, 1e-10, None))
    s2 = np.sqrt(np.clip(lam2, 1e-10, None))

    C     = (V1.T @ G12 @ V2) / (s1[:, None] * s2[None, :])
    svals = np.linalg.svd(C, compute_uv=False)
    return float(np.clip(svals[:top_k], 0.0, 1.0).mean())


# ------------------------------------------------------------------ #
#  Main stability runner                                               #
# ------------------------------------------------------------------ #

def run_stability(
    model1: CNN3D,
    model2: CNN3D,
    img_paths: Sequence[str | Path],
    device: torch.device,
    n_subjects: int = N_SUBJECTS,
    dim_batch: int  = DIM_BATCH,
    seed: int       = SEED,
    alphas: list[float] = ALPHAS,
    jacob_top_k: int    = JACOB_TOP_K,
) -> dict:
    """
    Run all four stability experiments (A, B, D, E) on a random sample of subjects.

    Parameters
    ----------
    model1, model2 : CNN3D
        Fold-1 and fold-2 models (both in eval mode).
    img_paths   : list of NIfTI paths.  The function will randomly sample
                  ``n_subjects`` from this list.
    device      : torch.device
    n_subjects  : number of subjects to sample.  Default: 500.
    dim_batch   : how many latent dimensions to process per GPU batch.
    seed        : random seed for subject sampling.
    alphas      : perturbation scale factors for Exp A.
    jacob_top_k : number of canonical correlations to average in Exp E.

    Returns
    -------
    results : dict with keys:
        "r_scale"       — np.ndarray (n_alphas, latent_dim, n_loaded)
                          Exp A: Pearson r between α·map and ref-α·map per (dim, subject).
        "hess_ratio"    — np.ndarray (latent_dim, n_loaded)
                          Exp D: ‖H_k‖ / ‖J_k‖ per (dim, subject).
        "r_cross_subj"  — np.ndarray (latent_dim,)
                          Exp B: mean pairwise Pearson r across subjects per dim.
        "N_cs"          — np.ndarray (latent_dim,)  — valid subject count per dim.
        "cos_sim_jacob" — np.ndarray (n_loaded,)
                          Exp E: Jacobian CCA per subject.
        "sigma"         — np.ndarray (latent_dim,)  — per-dim latent SDs.
        "n_loaded"      — int  — subjects with valid images.
        "latent_dim"    — int
    """
    n_features = model1.encoder_fc.out_features
    n_alphas   = len(alphas)
    ref_idx    = alphas.index(1.0)
    neg_idx    = alphas.index(-1.0)

    # ---- sample subjects ----
    rng   = random.Random(seed)
    paths = rng.sample(list(img_paths), min(n_subjects, len(img_paths)))
    N     = len(paths)

    # ---- pass 1: estimate per-dimension latent SDs ----
    print("  [Pass 1] Estimating latent standard deviations ...")
    latents = np.zeros((N, n_features), dtype=np.float32)
    n_ok    = 0
    for si, path in enumerate(tqdm(paths, desc="  encode")):
        try:
            x = _load_znorm(path, device)
        except Exception as e:
            print(f"  [WARN] {path}: {e}")
            continue
        with torch.no_grad():
            _, z = model1.encode_with_skip(x)
        latents[n_ok] = z.squeeze(0).cpu().numpy()
        n_ok += 1
    latents = latents[:n_ok]
    sigma   = latents.std(axis=0).clip(min=1e-6)   # (n_features,)
    print(f"  σ — min={sigma.min():.4f}  mean={sigma.mean():.4f}  max={sigma.max():.4f}")

    # ---- result arrays ----
    r_scale       = np.full((n_alphas, n_features, n_ok), np.nan, dtype=np.float32)
    hess_ratio    = np.full((n_features, n_ok), np.nan, dtype=np.float32)
    cos_sim_jacob = np.zeros(n_ok, dtype=np.float32)

    # Exp B streaming sums
    D_full: int | None = None
    S_cs: np.ndarray | None = None     # (n_features, D_full) running unit-sum
    N_cs  = np.zeros(n_features, dtype=np.int32)

    # ---- pass 2: perturbation maps ----
    print("  [Pass 2] Computing perturbation maps ...")
    si_ok = 0
    for path in tqdm(paths, desc="  subjects"):
        try:
            x = _load_znorm(path, device)
        except Exception as e:
            print(f"  [WARN] {path}: skip — {e}")
            continue

        with torch.no_grad():
            enc2_1, z1 = model1.encode_with_skip(x)
            enc2_2, z2 = model2.encode_with_skip(x)

        z1_0 = z1.squeeze(0)    # (n_features,)
        z2_0 = z2.squeeze(0)

        J1_subj: np.ndarray | None = None
        J2_subj: np.ndarray | None = None

        for d_start in range(0, n_features, dim_batch):
            d_end = min(d_start + dim_batch, n_features)
            B     = d_end - d_start
            dims  = list(range(d_start, d_end))

            e1_B = enc2_1.expand(B, -1, -1, -1, -1)
            e2_B = enc2_2.expand(B, -1, -1, -1, -1)

            # ---- Exp A: maps at all alpha values (model 1) ----
            # Batch all alpha × B dimensions into one set of forward passes.
            # perturbed latents: shape (n_alphas × B, n_features)
            perturbed_rows: list[torch.Tensor] = []
            for a in alphas:
                delta = torch.zeros(B, n_features, device=device)
                for bi, d in enumerate(dims):
                    delta[bi, d] = a * float(sigma[d])
                perturbed_rows.append(z1_0.unsqueeze(0).expand(B, -1) + delta)

            # Stack into (n_alphas*B, n_features), repeat enc2 accordingly
            lat_stack = torch.cat(perturbed_rows, dim=0)   # (n_alphas*B, n_features)
            enc2_rep  = enc2_1.expand(n_alphas * B, -1, -1, -1, -1)
            with torch.no_grad():
                recon_stack = model1.decode_from_latent(enc2_rep, lat_stack)
                recon_ref1  = model1.decode_from_latent(
                    enc2_1.expand(1, -1, -1, -1, -1), z1)

            recon_ref_np  = recon_ref1.squeeze(0).cpu().numpy().ravel()
            recon_np      = recon_stack.cpu().numpy()
            # recon_np: (n_alphas*B, 1, H, W, D)

            D_full_here = recon_ref_np.size

            if D_full is None:
                D_full = D_full_here
                S_cs   = np.zeros((n_features, D_full), dtype=np.float64)

            for bi, d in enumerate(dims):
                maps_at_alphas = []   # one per alpha
                for ai in range(n_alphas):
                    row_idx = ai * B + bi
                    diff    = recon_np[row_idx].ravel() - recon_ref_np
                    maps_at_alphas.append(diff)

                ref_map = maps_at_alphas[ref_idx]   # diff at alpha=1

                # Exp A: Pearson r between each α and the reference (α=1)
                for ai in range(n_alphas):
                    a_map = maps_at_alphas[ai]
                    rc    = _pearson(a_map, ref_map)
                    r_scale[ai, d, si_ok] = rc

                # Exp D: Hessian ratio
                # H_k = D(z+σe_k) + D(z−σe_k) − 2D(z)
                j_map    = ref_map                    # D(z+σe_k) − D(z)
                neg_map  = maps_at_alphas[neg_idx]    # D(z−σe_k) − D(z)
                h_vec    = j_map + neg_map            # H_k (centered)
                j_norm   = float(np.linalg.norm(j_map)) + 1e-12
                h_ratio  = float(np.linalg.norm(h_vec)) / j_norm
                hess_ratio[d, si_ok] = h_ratio

                # Exp B: cross-subject streaming unit-norm accumulation
                if np.linalg.norm(ref_map) > 1e-12:
                    unit_map = ref_map / np.linalg.norm(ref_map)
                    S_cs[d]  += unit_map
                    N_cs[d]  += 1

                # Exp E: collect Jacobian column (fold-1 and fold-2)
                # We store the maps for one subject at a time.
                if J1_subj is None:
                    J1_subj = np.zeros((D_full, n_features), dtype=np.float32)
                    J2_subj = np.zeros((D_full, n_features), dtype=np.float32)
                J1_subj[:, d] = ref_map.astype(np.float32)

            # ---- Exp E fold-2 Jacobian ----
            delta2 = torch.zeros(B, n_features, device=device)
            for bi, d in enumerate(dims):
                delta2[bi, d] = float(sigma[d])
            lat2_pert = z2_0.unsqueeze(0).expand(B, -1) + delta2
            enc2_2_rep = enc2_2.expand(B, -1, -1, -1, -1)
            with torch.no_grad():
                recon2_pert = model2.decode_from_latent(enc2_2_rep, lat2_pert)
                recon2_ref  = model2.decode_from_latent(
                    enc2_2.expand(1, -1, -1, -1, -1), z2)
            ref2_np = recon2_ref.squeeze(0).cpu().numpy().ravel()
            for bi, d in enumerate(dims):
                J2_subj[:, d] = (recon2_pert[bi].cpu().numpy().ravel() - ref2_np).astype(np.float32)

        # Exp E: compute CCA for this subject
        if J1_subj is not None and J2_subj is not None:
            cos_sim_jacob[si_ok] = jacobian_cca(J1_subj, J2_subj, top_k=jacob_top_k)

        si_ok += 1
        del x, enc2_1, enc2_2, z1, z2

    # ---- Exp B: convert streaming sums to mean pairwise r ----
    # mean pairwise r = (‖S_k‖² − N) / [N(N−1)]
    r_cross_subj = np.zeros(n_features, dtype=np.float32)
    for d in range(n_features):
        n = int(N_cs[d])
        if n < 2:
            continue
        s_norm_sq = float(np.dot(S_cs[d], S_cs[d]))
        r_cross_subj[d] = float((s_norm_sq - n) / (n * (n - 1)))

    return {
        "r_scale":        r_scale[:, :, :si_ok],
        "hess_ratio":     hess_ratio[:, :si_ok],
        "r_cross_subj":   r_cross_subj,
        "N_cs":           N_cs,
        "cos_sim_jacob":  cos_sim_jacob[:si_ok],
        "sigma":          sigma,
        "n_loaded":       si_ok,
        "latent_dim":     n_features,
        "alphas":         alphas,
        "ref_alpha_idx":  ref_idx,
    }


# ------------------------------------------------------------------ #
#  Save and summarise results                                          #
# ------------------------------------------------------------------ #

def save_stability_results(
    results: dict,
    out_dir: str | Path,
    tag: str,
    alphas: list[float] = ALPHAS,
) -> None:
    """
    Save per-experiment NPZ files and a human-readable summary text.

    Parameters
    ----------
    results : dict returned by ``run_stability``.
    out_dir : directory in which to write outputs.
    tag     : short label, e.g. "2ch" or "4ch".
    alphas  : scale factors used (needed for labelling summary).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez(str(out_dir / f"scale_linearity_{tag}.npz"),
             r_scale=results["r_scale"],
             alphas=np.array(results["alphas"]))
    np.savez(str(out_dir / f"hessian_{tag}.npz"),
             hess_ratio=results["hess_ratio"])
    np.savez(str(out_dir / f"cross_subject_{tag}.npz"),
             r_cross_subj=results["r_cross_subj"],
             N_cs=results["N_cs"])
    np.savez(str(out_dir / f"cross_fold_{tag}.npz"),
             cos_sim_jacob=results["cos_sim_jacob"])

    # ---- summary text ----
    r_ref = results["r_scale"][results["ref_alpha_idx"]]   # (n_features, N)
    # pick α=2 comparison: index where alphas == 2.0
    alpha_2_idx = alphas.index(2.0) if 2.0 in alphas else None
    r2_mean = float(np.nanmean(results["r_scale"][alpha_2_idx])) if alpha_2_idx is not None else float("nan")

    lines = [
        f"=== PerDI Stability Report [{tag}] ===",
        f"  N subjects loaded   : {results['n_loaded']}",
        f"  Latent dimensions   : {results['latent_dim']}",
        "",
        "Exp A — Scale linearity (Pearson r between α·map and 1·map)",
    ]
    for ai, a in enumerate(results["alphas"]):
        if a == 1.0:
            continue
        vals = results["r_scale"][ai]
        lines.append(
            f"  α={a:+.1f}  mean r={float(np.nanmean(vals)):.4f}  "
            f"5th-pct={float(np.nanpercentile(vals, 5)):.4f}"
        )
    lines += [
        "",
        "Exp B — Cross-subject consistency",
        f"  mean pairwise r  : {float(np.nanmean(results['r_cross_subj'])):.4f}",
        f"  5th percentile   : {float(np.nanpercentile(results['r_cross_subj'], 5)):.4f}",
        "",
        "Exp D — Hessian ratio ‖H_k‖/‖J_k‖",
        f"  mean ratio       : {float(np.nanmean(results['hess_ratio'])):.4f}",
        f"  95th percentile  : {float(np.nanpercentile(results['hess_ratio'], 95)):.4f}",
        "",
        "Exp E — Cross-fold Jacobian CCA (mean top-10 canonical r)",
        f"  mean per subject : {float(np.nanmean(results['cos_sim_jacob'])):.4f}",
        f"  5th percentile   : {float(np.nanpercentile(results['cos_sim_jacob'], 5)):.4f}",
    ]
    summary = "\n".join(lines)
    print(summary)
    (out_dir / f"summary_{tag}.txt").write_text(summary + "\n")


# ------------------------------------------------------------------ #
#  Internal helpers                                                    #
# ------------------------------------------------------------------ #

def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r between two 1-D arrays."""
    ac = a - a.mean()
    bc = b - b.mean()
    denom = (np.linalg.norm(ac) * np.linalg.norm(bc)) + 1e-12
    return float(np.dot(ac, bc) / denom)
