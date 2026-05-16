# PerDI — Perturbation-based Decoder Interpretation

PerDI is a post-hoc interpretability method for autoencoder models.
It produces **voxel-level perturbation maps** that show which regions of the
output image change when the latent space is nudged in a particular direction.
PerDI works with any autoencoder whose decoder is differentiable and whose
latent space can be encoded from real data samples.

Two modes are supported:

| Mode | What is perturbed | When to use |
|------|-------------------|-------------|
| **Regular PerDI** | Each latent dimension independently (one map per dim) | Exploring what every dimension encodes; no GWAS data needed |
| **Variant-specific PerDI** | The GWAS-derived direction w = (R + λI)⁻¹z (one map per variant) | Localising the effect of a genetic variant on the decoded output |

This folder contains the self-contained implementation of both modes and scripts
demonstrating their computation and stability validation.

---

## Mathematical formulation

### Regular PerDI — per-dimension maps

For each latent dimension k, perturb by its natural cross-subject standard
deviation σ_k and average the decoder responses across N subjects:

```
diff_i_k  =  Decoder(enc2_i, latent_i + σ_k · e_k)  −  Decoder(enc2_i, latent_i)
PerDI_k   =  (1/N) Σ_i  diff_i_k
```

where e_k is the k-th standard basis vector and σ_k = SD({ latent_i[k] }).

This produces a **generative atlas** of the decoder: one signed map per latent
dimension that together describe every mode of variation the model has learned
to encode.

---

### Variant-specific PerDI

#### Step 1 — Phenotypic correlation matrix R

R is the **Pearson correlation of GCTA fastGWA-mlm LMM residuals**.
For each latent dimension k, GCTA fits a linear mixed model that accounts for
the genetic random effect (population structure, cryptic relatedness):

```
phenotype_k  =  X β  +  g  +  ε       (g ~ N(0, σ²_g · GRM),  ε ~ N(0, σ²_e · I))
```

where X contains covariates (age, sex, PCs, assessment centre, …) and GRM is
the genetic relationship matrix.  The LMM residual (ε̂) is used for dimension k
**when GCTA's variance component estimation converges** (`.fastGWA.residual`
file is produced).  For dimensions where GCTA does not converge or Vg is not
significant, plain **OLS residuals** (projecting out X only) are used as a
fallback.  This hybrid strategy is implemented in
`load_gcta_residuals_hybrid()` in `core.py`.

```
R  =  Corr( hybrid_residuals )   ∈ ℝ^(D×D)
```

#### Step 2 — Variant weights w

Given GWAS univariate z-scores z ∈ ℝ^D (one z-score per latent dimension for
the variant of interest), the optimal multivariate weight vector is:

```
w  =  (R + λI)⁻¹ z        (ridge-regularised, λ = 1e-4 by default)
```

This solves the multivariate chi-squared maximisation problem:
the direction **w** in feature space has the maximal multivariate
association statistic **z^T R⁻¹ z** with the variant.

#### Step 3 — Perturbation scale α

We perturb the latent **along the w direction** by an amount equal to
**1 standard deviation of subjects' scores along w**:

```
α   =  SD({ w · latent_i })          (1-SD spread along the w direction)
```

The perturbation vector that shifts the w-direction score by exactly α is
`α · w / ‖w‖²`.  Normalising first gives the equivalent implementation:

```
ŵ   =  w / ‖w‖
α̂   =  α / ‖w‖  =  SD({ ŵ · latent_i })
perturbation  =  α̂ · ŵ
```

Both forms produce the same displacement in latent space.

#### Step 4 — Weighted decoder perturbation map

For each subject i with encoded representation (enc2_i, latent_i):

```
diff_i  =  Decoder(enc2_i, latent_i + α̂ · ŵ)  −  Decoder(enc2_i, latent_i)
```

The variant-level PerDI map is the genotype-weighted mean:

```
PerDI  =  (1/N) Σ_i  weight_i · diff_i
```

#### Derivation of the subject weights

Model each subject's perturbation map as a sum of three components:

```
diff_i  =  β · g_i · Δ  +  B  +  ε_i
```

where  
- **β g_i Δ** is the variant-specific part (scales linearly with dosage g_i),  
- **B** is a common bias term shared by all subjects,  
- **ε_i** is independent noise with variance σ².

We want to find weights {c_i} such that the weighted average Σ_i c_i · diff_i
recovers **Δ** (up to a constant), while:

1. **The bias cancels**: Σ_i c_i · B = 0  ⟹  **Σ_i c_i = 0**  
2. **The variant signal is preserved**: Σ_i c_i · g_i = const  
3. **Noise variance is minimised**: minimise Σ_i c_i²

Applying Lagrange multipliers to minimise Σ_i c_i² subject to the two linear
constraints yields:

```
c_i  ∝  g_i  −  mean({g_j})
```

i.e. the optimal weights are exactly the **mean-centred dosages**.  The
mean-centring simultaneously cancels the common bias term and minimises the
contribution of independent noise, making the sign of the resulting map
directly reflect the direction of the genetic effect (positive dosage →
positive image change).

---

## Files

| File | Description |
|------|-------------|
| `model.py` | Standalone CNN3D autoencoder definition and `load_model` helper |
| `core.py` | Regular PerDI, variant-specific PerDI, and helpers |
| `stability.py` | Scale-linearity, cross-subject consistency, Hessian ratio, cross-fold Jacobian CCA |
| `examples/config_template.yaml` | Template configuration with placeholder paths |
| `examples/example_regular_perdi.py` | Regular PerDI: per-dimension maps without GWAS data |
| `examples/example_compute_weights.py` | Variant-specific: residuals → R → w = R⁻¹z |
| `examples/example_perdi_map.py` | Variant-specific: weights + dosages → PerDI NIfTI map |
| `examples/example_stability.py` | Stability experiments demo |

---

## Quick start

### Regular PerDI (no GWAS required)

```python
from PerDI.model import load_model, select_gpu
from PerDI.core import estimate_latent_sigmas, compute_all_dimension_perdi_maps

device    = select_gpu()
model     = load_model("/path/to/model_epoch_10.pth", device)
img_paths = [...]   # list of paths to encoded images (one per subject)

sigma      = estimate_latent_sigmas(model, img_paths, device)
perdi_maps, affine, n = compute_all_dimension_perdi_maps(
    model, img_paths, sigma, device
)
# perdi_maps: (D, H, W, Depth) — one signed float32 map per latent dimension
```

### Variant-specific PerDI

```python
import numpy as np
from PerDI.model import load_model, select_gpu
from PerDI.core import (
    compute_phenotypic_correlation_matrix,
    compute_variant_weights,
    compute_variant_perdi_map,
    save_perdi_nifti,
)

device = select_gpu()
model  = load_model("/path/to/model_epoch_10.pth", device)

# Compute R from precomputed residuals (N × D)
residuals = np.load("/path/to/residuals.npy")
R         = compute_phenotypic_correlation_matrix(residuals)

# Compute variant weights from GWAS z-scores (D,)
z_scores = np.load("/path/to/z_scores_rsXXXXXX.npy")
w        = compute_variant_weights(R, z_scores)

# Compute PerDI map
#   img_paths: list of paths to encoded images (one per sampled subject)
#   dosages:   np.ndarray (N,) — hard-call or posterior mean genotype dosages
perdi_map, affine, meta = compute_variant_perdi_map(model, img_paths, dosages, w, device)
save_perdi_nifti(perdi_map, affine, "output/rsXXXXXX.nii.gz")
```

---

## Stability validation

Four experiments are implemented in `stability.py` and demonstrated in
`examples/example_stability.py`:

| Experiment | What it measures | Expected result |
|------------|-----------------|-----------------|
| **A — Scale linearity** | Pearson r between maps at α and 2α | > 0.95 |
| **B — Cross-subject consistency** | Mean pairwise r across held-out subjects | > 0.80 |
| **D — Hessian ratio** | ‖H_k‖ / ‖J_k‖ (second-order vs first-order) | < 0.10 |
| **E — Cross-fold Jacobian CCA** | Top-k Gram-matrix CCA across two model folds | > 0.85 |

Run all four with a single call:

```python
from PerDI.stability import run_stability, save_stability_results

results = run_stability(model_fold1, model_fold2, img_paths, device)
save_stability_results(results, out_dir="output/stability", tag="2ch")
```

---