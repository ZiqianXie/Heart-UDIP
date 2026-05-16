"""
core.py — Core PerDI (Perturbation-based Decoder Interpretation) functions.

Two modes of PerDI
------------------
Regular PerDI (per-dimension)
    Perturbs each of the D latent dimensions independently by ±σ_k (its
    natural cross-subject standard deviation).  Produces D maps — one per
    dimension — that together describe the full generative atlas of the decoder.
    No GWAS data required.

    For dimension k and a population of N subjects:
        diff_i_k = Decoder(enc2_i, latent_i + σ_k · e_k) − Decoder(enc2_i, latent_i)
        PerDI_k  = (1/N) Σ_i  diff_i_k

Variant-specific PerDI
    For a target SNP, computes the optimal linear combination of dimensions:
        w  = (R + λI)⁻¹ z          (R = phenotypic correlation matrix, z = GWAS z-scores)
        ŵ  = w / ‖w‖
        α  = SD({ ŵ · latent_i })  (1-SD natural spread along ŵ)
        diff_i = Decoder(enc2_i, latent_i + α · ŵ) − Decoder(enc2_i, latent_i)
        PerDI  = (1/N) Σ_i  weight_i · diff_i    (weight_i = dosage_i − mean dosage)

See README.md for detailed mathematical formulation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import nibabel as nib
import torch
from scipy.linalg import solve

from .model import CNN3D

# ------------------------------------------------------------------ #
#  Step 1 — Phenotypic correlation matrix                              #
# ------------------------------------------------------------------ #


def compute_phenotypic_correlation_matrix(
    residuals: np.ndarray,
    ridge: float = 1e-4,
) -> np.ndarray:
    """
    Compute the (D × D) Pearson correlation matrix from an (N × D) residual matrix.

    Parameters
    ----------
    residuals : np.ndarray  (N, D)
        Covariate-residualised latent features.  N = subjects, D = latent dim.
    ridge : float
        Small ridge term added to the diagonal of R before returning.
        Ensures the matrix is invertible for the R⁻¹z solve in step 2.
        Set to 0.0 to return the raw correlation matrix.

    Returns
    -------
    R : np.ndarray  (D, D)  float64 — regularised correlation matrix.
    """
    if residuals.ndim != 2:
        raise ValueError(f"residuals must be 2-D (N, D); got shape {residuals.shape}")

    X = residuals.astype(np.float64)
    # Standardise each column (feature) to zero mean, unit variance
    mean = X.mean(axis=0, keepdims=True)
    std  = X.std(axis=0, keepdims=True)
    std  = np.where(std < 1e-12, 1.0, std)
    X    = (X - mean) / std

    R = (X.T @ X) / (X.shape[0] - 1)           # (D, D) correlation matrix
    R += np.eye(R.shape[0]) * ridge             # ridge regularisation
    return R


# ------------------------------------------------------------------ #
#  Step 2 — Variant-specific optimal weights w = R⁻¹z                 #
# ------------------------------------------------------------------ #


def compute_variant_weights(
    R: np.ndarray,
    z_scores: np.ndarray,
    ridge: float = 1e-4,
) -> np.ndarray:
    """
    Compute the optimal linear combination weights  w = (R + λI)⁻¹ z.

    ``w`` defines the direction in the 256-dimensional latent space that is
    maximally associated with the target variant.  It solves the multivariate
    association maximisation problem  max_{w}  z^T R⁻¹ z.

    Parameters
    ----------
    R : np.ndarray  (D, D)
        Phenotypic correlation matrix from ``compute_phenotypic_correlation_matrix``.
        If R already has a ridge term baked in (the default), set ``ridge=0.0``
        here to avoid double-regularisation.
    z_scores : np.ndarray  (D,)
        Per-feature GWAS z-scores (β / SE) for the target variant.
        Missing features should be imputed as 0.
    ridge : float
        Additional ridge added to R before solving.  Ignored if already
        added in ``compute_phenotypic_correlation_matrix``.  Default: 1e-4.

    Returns
    -------
    w : np.ndarray  (D,)  float32
    """
    if R.shape[0] != R.shape[1]:
        raise ValueError(f"R must be square; got {R.shape}")
    if z_scores.shape[0] != R.shape[0]:
        raise ValueError(
            f"z_scores length {z_scores.shape[0]} must match R dimension {R.shape[0]}"
        )

    R_reg = R.astype(np.float64) + np.eye(R.shape[0]) * ridge
    z     = np.where(np.isnan(z_scores), 0.0, z_scores).astype(np.float64)
    w     = solve(R_reg, z, assume_a="sym")
    return w.astype(np.float32)


# ------------------------------------------------------------------ #
#  Image loading helper                                                #
# ------------------------------------------------------------------ #


def load_nifti_znorm(
    img_path: str | Path,
    device: torch.device,
) -> torch.Tensor:
    """
    Load a NIfTI volume, apply per-sample z-normalisation, and return a
    (1, 1, H, W, D) float32 tensor on *device*.

    Per-sample z-normalisation (subtract mean, divide by std) is consistent
    with how subjects were normalised during model training.
    """
    img = nib.load(str(img_path)).get_fdata(dtype=np.float32)
    mean, std = float(img.mean()), float(img.std())
    if std <= 0:
        std = 1.0
    img = (img - mean) / std
    return torch.tensor(img, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)


# ------------------------------------------------------------------ #
#  Step 3 — Perturbation scale α                                       #
# ------------------------------------------------------------------ #


def estimate_perturbation_alpha(
    model: CNN3D,
    img_paths: Sequence[str | Path],
    w: np.ndarray,
    device: torch.device,
) -> float:
    """
    Estimate the perturbation scale: 1 SD of subjects along the w direction.

    The perturbation for variant-specific PerDI moves the latent **along w**
    by the natural 1-SD spread of subjects in that direction.

    Concretely we compute:

        alpha  =  SD({ w · latent_i })          (1-SD spread along w)

    and the actual perturbation vector is  ``(alpha / ‖w‖) · ŵ``  so that the
    w-direction score shifts by exactly ``alpha``.  The return value is the
    normalised form ``alpha_hat = alpha / ‖w‖ = SD({ ŵ · latent_i })``,
    which is passed directly to ``compute_single_perturbation_map`` together
    with the unit vector ``ŵ``.

    Parameters
    ----------
    model     : CNN3D  — loaded and in eval mode.
    img_paths : sequence of NIfTI paths (one per subject).
    w         : np.ndarray (D,)  — variant weights from ``compute_variant_weights``.
    device    : torch.device

    Returns
    -------
    alpha_hat : float  — ``SD({ ŵ · latent_i })``, the normalised perturbation scale.
    """
    w_norm = float(np.linalg.norm(w)) + 1e-12
    w_hat  = (w / w_norm).astype(np.float32)
    w_hat_t = torch.tensor(w_hat, dtype=torch.float32, device=device)

    scores: list[float] = []
    with torch.no_grad():
        for path in img_paths:
            try:
                x = load_nifti_znorm(path, device)
            except Exception:
                continue
            _, latent = model.encode_with_skip(x)
            scores.append(float((latent.squeeze() * w_hat_t).sum()))
            del x, latent

    if len(scores) < 2:
        raise RuntimeError(
            f"Only {len(scores)} valid images loaded; need at least 2 to estimate alpha."
        )
    return float(np.std(scores)) + 1e-8


# ------------------------------------------------------------------ #
#  Step 4 — Single-subject perturbation map                            #
# ------------------------------------------------------------------ #


def compute_single_perturbation_map(
    model: CNN3D,
    img_tensor: torch.Tensor,
    w_hat: torch.Tensor,
    alpha: float,
) -> np.ndarray:
    """
    Compute  D(enc2, latent + α·ŵ) − D(enc2, latent)  for one subject.

    This is a pure forward-pass operation (no gradients required).

    Parameters
    ----------
    model      : CNN3D in eval mode.
    img_tensor : torch.Tensor (1, 1, H, W, D) — z-normalised subject image.
    w_hat      : torch.Tensor (latent_dim,)    — unit-normalised weight direction.
    alpha      : float                         — perturbation magnitude (1-SD scale).

    Returns
    -------
    diff : np.ndarray (H, W, D)  float32
    """
    with torch.no_grad():
        enc2, latent    = model.encode_with_skip(img_tensor)
        recon_orig      = model.decode_from_latent(enc2, latent)
        latent_pert     = latent + alpha * w_hat.unsqueeze(0)
        recon_pert      = model.decode_from_latent(enc2, latent_pert)
        diff            = (recon_pert - recon_orig).squeeze().cpu().numpy()
    return diff.astype(np.float32)


# ------------------------------------------------------------------ #
#  Full PerDI map for a single variant                                 #
# ------------------------------------------------------------------ #


def compute_variant_perdi_map(
    model: CNN3D,
    img_paths: Sequence[str | Path],
    dosages: np.ndarray,
    w: np.ndarray,
    device: torch.device,
    conv_checkpoint: int = 450,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Compute the full PerDI map for a single genetic variant.

    Two-pass algorithm
    ------------------
    Pass 1 : Collect latent projections to estimate perturbation scale α.
    Pass 2 : Compute per-subject perturbation maps, weight by mean-centred
             dosage, and accumulate a running weighted sum.

    Parameters
    ----------
    model           : CNN3D in eval mode.
    img_paths       : sequence of NIfTI paths, one per sampled subject.
    dosages         : np.ndarray (N,)  — genotype dosage (0/1/2 hard-call or
                      posterior mean).  Must match ``img_paths`` order.
    w               : np.ndarray (D,)  — variant weights from ``compute_variant_weights``.
    device          : torch.device
    conv_checkpoint : int
                      Subject count at which to snapshot the running map for a
                      convergence check.  Default: 450 (of 500 total).

    Returns
    -------
    perdi_map  : np.ndarray (H, W, D)  float32  — genotype-weighted mean map.
    affine     : np.ndarray (4, 4)               — NIfTI affine of the first
                 successfully loaded image (for saving the map as NIfTI).
    meta       : dict with keys:
                   "n_loaded"      — number of subjects with a valid image,
                   "alpha"         — estimated perturbation scale,
                   "pearson_r"     — convergence Pearson r (nan if no checkpoint),
                   "converged"     — bool (r > 0.70).

    Notes
    -----
    Mean-centring the dosage weights ensures sign consistency (the map
    reflects the direction of the genetic effect) and cancels any
    population-mean shift.
    """
    if len(img_paths) != len(dosages):
        raise ValueError("img_paths and dosages must have the same length.")

    # ---- normalise weight vector ----
    w_np   = np.asarray(w, dtype=np.float32)
    w_norm = float(np.linalg.norm(w_np)) + 1e-12
    w_hat  = (w_np / w_norm).astype(np.float32)
    w_hat_t = torch.tensor(w_hat, dtype=torch.float32, device=device)

    # ---- pass 1: collect latent projections ----
    latent_scores: list[float] = []
    with torch.no_grad():
        for path in img_paths:
            try:
                x = load_nifti_znorm(path, device)
            except Exception:
                continue
            _, latent = model.encode_with_skip(x)
            latent_scores.append(float((latent.squeeze() * w_hat_t).sum()))
            del x, latent

    if len(latent_scores) < 2:
        raise RuntimeError("Too few valid images to estimate perturbation scale.")

    alpha = float(np.std(latent_scores)) + 1e-8

    # ---- pass 2: accumulate weighted perturbation maps ----
    dosages_f  = np.asarray(dosages, dtype=np.float32)
    dose_mean  = float(dosages_f.mean())

    weighted_sum: np.ndarray | None = None
    weighted_sum_early: np.ndarray | None = None
    affine: np.ndarray | None = None
    img_shape: tuple | None = None
    n_loaded = 0

    for path, dosage in zip(img_paths, dosages_f):
        try:
            nii = nib.load(str(path))
            img = nii.get_fdata(dtype=np.float32)
        except Exception:
            continue

        if img_shape is None:
            img_shape = img.shape
            affine    = nii.affine
            weighted_sum = np.zeros(img_shape, dtype=np.float64)

        mean_v, std_v = float(img.mean()), float(img.std())
        if std_v <= 0:
            continue

        x    = torch.tensor((img - mean_v) / std_v, dtype=torch.float32,
                             device=device).unsqueeze(0).unsqueeze(0)
        diff = compute_single_perturbation_map(model, x, w_hat_t, alpha)
        del x

        if diff.shape != img_shape:
            continue

        weight = float(dosage) - dose_mean
        weighted_sum += weight * diff.astype(np.float64)
        n_loaded += 1

        if n_loaded == conv_checkpoint:
            weighted_sum_early = weighted_sum.copy()

    if n_loaded == 0 or weighted_sum is None or affine is None:
        raise RuntimeError("No valid images loaded.")

    perdi_map = (weighted_sum / n_loaded).astype(np.float32)

    # ---- convergence check ----
    pearson_r = float("nan")
    converged = False
    if weighted_sum_early is not None and n_loaded >= conv_checkpoint:
        early = (weighted_sum_early / conv_checkpoint).flatten()
        final = perdi_map.flatten().astype(np.float64)
        ec    = early - early.mean()
        fc    = final - final.mean()
        denom = np.linalg.norm(ec) * np.linalg.norm(fc)
        if denom > 0:
            pearson_r = float(np.dot(ec, fc) / denom)
            converged = pearson_r > 0.70

    meta = {
        "n_loaded":  n_loaded,
        "alpha":     alpha,
        "pearson_r": pearson_r,
        "converged": converged,
    }
    return perdi_map, affine, meta


# ------------------------------------------------------------------ #
#  Regular PerDI — per-dimension perturbation maps                     #
# ------------------------------------------------------------------ #


def estimate_latent_sigmas(
    model: CNN3D,
    img_paths: Sequence[str | Path],
    device: torch.device,
) -> np.ndarray:
    """
    Estimate the per-dimension latent standard deviations across a subject pool.

    These are used as the perturbation scale for regular PerDI:
        σ_k = SD({ latent_i[k]  for all subjects i })

    Parameters
    ----------
    model     : CNN3D in eval mode.
    img_paths : NIfTI paths for the subjects to use (typically 200–500).
    device    : torch.device

    Returns
    -------
    sigma : np.ndarray (D,)  float32  — per-dimension SD, clipped to a minimum of 1e-6.
    """
    latent_dim = model.encoder_fc.out_features
    latents: list[np.ndarray] = []
    with torch.no_grad():
        for path in img_paths:
            try:
                x = load_nifti_znorm(path, device)
            except Exception:
                continue
            _, z = model.encode_with_skip(x)
            latents.append(z.squeeze(0).cpu().numpy())
            del x, z
    if not latents:
        raise RuntimeError("No valid images loaded for sigma estimation.")
    return np.stack(latents, axis=0).std(axis=0).clip(min=1e-6).astype(np.float32)


def compute_dimension_perdi_map(
    model: CNN3D,
    img_paths: Sequence[str | Path],
    dim: int,
    sigma: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Compute the regular PerDI map for a **single** latent dimension.

    The map is the unweighted mean decoder difference across subjects:

        diff_i = Decoder(enc2_i, latent_i + σ_k · e_k) − Decoder(enc2_i, latent_i)
        PerDI_k = (1/N) Σ_i  diff_i

    No GWAS data or genotypes are needed.

    Parameters
    ----------
    model     : CNN3D in eval mode.
    img_paths : NIfTI paths for the subjects to include.
    dim       : Index of the latent dimension to perturb (0-based).
    sigma     : np.ndarray (D,) — per-dimension SDs from ``estimate_latent_sigmas``.
    device    : torch.device

    Returns
    -------
    perdi_map : np.ndarray (H, W, D)  float32
    affine    : np.ndarray (4, 4)  — spatial affine from the first loaded image.
    n_loaded  : int  — number of subjects contributing to the map.
    """
    latent_dim = model.encoder_fc.out_features
    if not (0 <= dim < latent_dim):
        raise ValueError(f"dim={dim} is out of range [0, {latent_dim})")

    alpha_k = float(sigma[dim])

    running_sum: np.ndarray | None = None
    affine: np.ndarray | None = None
    img_shape: tuple | None = None
    n_loaded = 0

    with torch.no_grad():
        for path in img_paths:
            try:
                nii = nib.load(str(path))
                img = nii.get_fdata(dtype=np.float32)
            except Exception:
                continue

            if img_shape is None:
                img_shape   = img.shape
                affine      = nii.affine
                running_sum = np.zeros(img_shape, dtype=np.float64)

            mean_v, std_v = float(img.mean()), float(img.std())
            if std_v <= 0:
                continue

            x = torch.tensor((img - mean_v) / std_v, dtype=torch.float32,
                              device=device).unsqueeze(0).unsqueeze(0)

            enc2, latent  = model.encode_with_skip(x)
            recon_orig    = model.decode_from_latent(enc2, latent)

            latent_pert        = latent.clone()
            latent_pert[0, dim] += alpha_k
            recon_pert    = model.decode_from_latent(enc2, latent_pert)

            diff = (recon_pert - recon_orig).squeeze().cpu().numpy()
            del x, enc2, latent, latent_pert, recon_orig, recon_pert

            if diff.shape != img_shape:
                continue
            running_sum += diff.astype(np.float64)
            n_loaded    += 1

    if n_loaded == 0 or running_sum is None or affine is None:
        raise RuntimeError(f"No valid images loaded for dim={dim}.")

    return (running_sum / n_loaded).astype(np.float32), affine, n_loaded


def compute_all_dimension_perdi_maps(
    model: CNN3D,
    img_paths: Sequence[str | Path],
    sigma: np.ndarray,
    device: torch.device,
    dim_batch: int = 32,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Compute regular PerDI maps for **all** latent dimensions in a single pass.

    Uses batched GPU forward passes (``dim_batch`` dimensions at once) to avoid
    repeated image loading.  Memory usage scales with ``dim_batch``.

    Parameters
    ----------
    model     : CNN3D in eval mode.
    img_paths : NIfTI paths for the subjects to include.
    sigma     : np.ndarray (D,) — per-dimension SDs from ``estimate_latent_sigmas``.
    device    : torch.device
    dim_batch : number of latent dimensions to process per GPU batch.

    Returns
    -------
    perdi_maps : np.ndarray (D, H, W, Depth)  float32
                 Stack of maps; dimension k is at index k.
    affine     : np.ndarray (4, 4)  — spatial affine.
    n_loaded   : int  — subjects included.
    """
    latent_dim = model.encoder_fc.out_features
    sigma_t    = torch.tensor(sigma, dtype=torch.float32, device=device)  # (D,)

    running_sum: np.ndarray | None = None   # (D, H, W, Depth)  float64
    affine: np.ndarray | None = None
    img_shape: tuple | None = None
    n_loaded = 0

    with torch.no_grad():
        for path in img_paths:
            try:
                nii = nib.load(str(path))
                img = nii.get_fdata(dtype=np.float32)
            except Exception:
                continue

            if img_shape is None:
                img_shape   = img.shape
                affine      = nii.affine
                running_sum = np.zeros((latent_dim, *img_shape), dtype=np.float64)

            mean_v, std_v = float(img.mean()), float(img.std())
            if std_v <= 0:
                continue

            x = torch.tensor((img - mean_v) / std_v, dtype=torch.float32,
                              device=device).unsqueeze(0).unsqueeze(0)
            enc2, latent = model.encode_with_skip(x)
            recon_orig   = model.decode_from_latent(enc2, latent).squeeze(0)  # (1,H,W,D)

            # process in batches of dim_batch dimensions
            for d_start in range(0, latent_dim, dim_batch):
                d_end = min(d_start + dim_batch, latent_dim)
                B     = d_end - d_start

                # latent perturbed for each of the B dimensions: (B, latent_dim)
                lat_batch = latent.expand(B, -1).clone()
                for bi, d in enumerate(range(d_start, d_end)):
                    lat_batch[bi, d] += float(sigma[d])

                enc2_batch  = enc2.expand(B, -1, -1, -1, -1)
                recon_pert  = model.decode_from_latent(enc2_batch, lat_batch)  # (B,1,H,W,D)
                recon_orig_b = recon_orig.unsqueeze(0).expand(B, -1, -1, -1, -1)
                diff_batch  = (recon_pert - recon_orig_b).squeeze(1).cpu().numpy()  # (B,H,W,D)

                running_sum[d_start:d_end] += diff_batch.astype(np.float64)

            del x, enc2, latent, recon_orig
            n_loaded += 1

    if n_loaded == 0 or running_sum is None or affine is None:
        raise RuntimeError("No valid images loaded.")

    return (running_sum / n_loaded).astype(np.float32), affine, n_loaded


# ------------------------------------------------------------------ #
#  Convenience: save PerDI map as NIfTI                                #
# ------------------------------------------------------------------ #


def save_perdi_nifti(
    perdi_map: np.ndarray,
    affine: np.ndarray,
    output_path: str | Path,
) -> None:
    """
    Save a PerDI map (H, W, D) as a signed float32 NIfTI file.

    Parameters
    ----------
    perdi_map   : np.ndarray (H, W, D)  float32
    affine      : np.ndarray (4, 4)  — spatial affine from the source images
    output_path : destination path (will be created with ``.nii.gz`` if needed)
    """
    nii = nib.Nifti1Image(perdi_map.astype(np.float32), affine)
    nib.save(nii, str(output_path))
