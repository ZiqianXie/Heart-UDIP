"""
PerDI — Perturbation-based Decoder Interpretation for cardiac MRI autoencoders.

Public API
----------
from PerDI.model     import CNN3D, load_model, select_gpu
from PerDI.core      import (
    # regular PerDI (per-dimension, no GWAS needed)
    estimate_latent_sigmas,
    compute_dimension_perdi_map,
    compute_all_dimension_perdi_maps,
    # variant-specific PerDI (requires GWAS z-scores)
    compute_phenotypic_correlation_matrix,
    compute_variant_weights,
    estimate_perturbation_alpha,
    compute_single_perturbation_map,
    compute_variant_perdi_map,
    # shared
    save_perdi_nifti,
)
from PerDI.stability import run_stability, save_stability_results
"""

from .model import CNN3D, load_model, select_gpu
from .core import (
    estimate_latent_sigmas,
    compute_dimension_perdi_map,
    compute_all_dimension_perdi_maps,
    compute_phenotypic_correlation_matrix,
    compute_variant_weights,
    estimate_perturbation_alpha,
    compute_single_perturbation_map,
    compute_variant_perdi_map,
    save_perdi_nifti,
)
from .stability import run_stability, save_stability_results

__all__ = [
    "CNN3D",
    "load_model",
    "select_gpu",
    "estimate_latent_sigmas",
    "compute_dimension_perdi_map",
    "compute_all_dimension_perdi_maps",
    "compute_phenotypic_correlation_matrix",
    "compute_variant_weights",
    "estimate_perturbation_alpha",
    "compute_single_perturbation_map",
    "compute_variant_perdi_map",
    "save_perdi_nifti",
    "run_stability",
    "save_stability_results",
]
