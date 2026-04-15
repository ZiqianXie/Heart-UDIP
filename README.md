# Heart-UDIP

Code accompanying the study:

**Uncovering genetic architecture of the heart via genetic association studies of unsupervised deep learning derived endophenotypes**

This repository contains the imaging preprocessing, self-supervised representation learning, latent phenotype extraction, evaluation, interpretability, and GWAS preparation workflows used in the Heart-UDIP project.

## Workflow overview

<img width="1125" height="633" alt="Heart-UDIP workflow overview" src="https://github.com/user-attachments/assets/26fa0d37-f63f-45e5-9f40-20bbdef01f37" />

## Why this repository was reorganized

The original codebase was a collection of research scripts with hard-coded paths and inconsistent directory names. It has been refactored into a clearer project structure with:

- installable Python package layout under `src/`
- parameterized command-line entry points instead of in-file path editing
- separate locations for source code, scripts, configs, docs, data placeholders, and results
- README sections aligned with common academic code-release expectations
- explicit placeholders for `Code availability`, `Data availability`, and reproducibility reporting

## Repository structure

```text
Heart-UDIP/
├── configs/                     # Example path and environment templates
├── data/
│   ├── raw/                     # Raw data placeholder only; not tracked
│   └── processed/               # Derived data placeholder only; not tracked
├── docs/
│   └── reproducibility.md       # Notes for manuscript-ready reporting
├── model_weights/
│   └── readme                   # The links for the pretrained weights of our CineMAE
├── results/                     # Output placeholder only; not tracked
├── scripts/
│   └── post_gwas_analysis.R     # Downstream exploratory R analysis
│   └── pipeline_test_with_other_data.py     # the pipeline test script for our models.
├── src/
│   └── heart_udip/
│       ├── preprocessing/       # DICOM conversion, cropping, template building, registration
│       ├── models/              # CNN3D definition and training
│       ├── evaluation/          # Metrics, latent extraction, perturbation analysis
│       └── gwas/                # fastGWA preparation, min-P aggregation, correlation, UMAP
├── LICENSE
├── pyproject.toml
└── requirements.txt
```

## Scientific workflow

### 1. Imaging preprocessing

1. Convert one DICOM folder per participant into NIfTI volumes.
2. Obtain heart masks with your segmentation workflow, for example an nnU-Net model trained on manually annotated cases.
3. Crop masked images into a standardized heart-centered tensor, typically `80 x 80 x 50`.
4. Build a registration template from a representative subset of cropped volumes.
5. Register all cropped volumes to the fixed template.

### 2. Representation learning

1. Train the 3D cine-MAE style autoencoder on the cropped or registered volumes.
2. Save model checkpoints, logs, and reconstruction previews.

### 3. Latent phenotype generation

1. Run latent extraction on the trained encoder.
2. Export one feature vector per participant.
3. Use the resulting feature table as quantitative imaging endophenotypes for GWAS.

### 4. Evaluation and interpretation

1. Quantify reconstruction quality with SSIM and PSNR.
2. Generate perturbation-based latent attribution maps.
3. Export slice-wise visualizations for manuscript figures.

### 5. GWAS and post-GWAS analysis

1. Convert the latent feature table into one phenotype file per latent dimension.
2. Run GCTA fastGWA externally.
3. Aggregate summary statistics with a min-P strategy.
4. Visualize latent spaces and downstream association patterns.

## Installation

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## Example commands

Set `PYTHONPATH=src` if you do not install the package.

### Convert DICOM to NIfTI

```bash
python -m heart_udip.preprocessing.dicom_to_nifti \
  --dicom-root /path/to/dicom_root \
  --output-dir /path/to/nii_output \
  --error-log results/dicom_conversion.log
```

### Crop heart-centered volumes using segmentation masks

```bash
python -m heart_udip.preprocessing.crop_nifti \
  --input-dir /path/to/nii_output \
  --mask-dir /path/to/segmentation_masks \
  --output-dir /path/to/cropped_80x80x50
```

### Build an ANTs template

```bash
python -m heart_udip.preprocessing.build_template \
  --input-dir /path/to/cropped_80x80x50 \
  --output-path results/template/template_150_zfixed.nii.gz \
  --max-files 150
```

### Register volumes to the template

```bash
python -m heart_udip.preprocessing.register_to_template \
  --input-dir /path/to/cropped_80x80x50 \
  --template-path results/template/template_150_zfixed.nii.gz \
  --output-dir /path/to/registered_volumes
```

### Train the cine-MAE model

```bash
python -m heart_udip.models.train_cinemae \
  --input-dir /path/to/registered_volumes \
  --output-dir results/train_run_01 \
  --epochs 100 \
  --batch-size 16 \
  --device cuda:0
```

### Evaluate reconstruction quality

```bash
python -m heart_udip.evaluation.reconstruction_metrics \
  --input-dir /path/to/registered_volumes \
  --weights results/train_run_01/model_epoch_100.pth \
  --sample-size 500
```

### Extract latent phenotypes

```bash
python -m heart_udip.evaluation.extract_latents \
  --input-dir /path/to/registered_volumes \
  --weights results/train_run_01/model_epoch_100.pth \
  --output-csv results/latents/ukb_heart_udip_features.csv
```

### Generate perturbation-based attribution maps

```bash
python -m heart_udip.evaluation.perturbation_analysis \
  --input-dir /path/to/registered_volumes \
  --weights results/train_run_01/model_epoch_100.pth \
  --output-dir results/perturbation_maps
```

### Export slice-wise figures

```bash
python -m heart_udip.evaluation.plot_perturbation \
  --input-path results/perturbation_maps \
  --output-dir results/perturbation_pngs
```

### Prepare phenotype files for fastGWA

```bash
python -m heart_udip.gwas.prepare_fastgwa_inputs \
  --feature-csv results/latents/ukb_heart_udip_features.csv \
  --pheno-dir results/gwas/phenotypes
```

### Aggregate fastGWA outputs with min-P

```bash
python -m heart_udip.gwas.minp_analysis \
  --gwas-dir results/gwas/fastgwa \
  --pattern "*.fastGWA" \
  --output-tsv results/gwas/minp/minp_results.tsv \
  --output-manhattan results/gwas/minp/minp_manhattan.png
```

### Run configurable post-GWAS R analyses

```bash
Rscript scripts/post_gwas_analysis.R \
  --task=heart-manhattan \
  --gwas-file=results/gwas/4ch_minp.tsv \
  --annotation-dir=results/fuma/4ch \
  --output-prefix=results/post_gwas/4ch
```

```bash
Rscript scripts/post_gwas_analysis.R \
  --task=overlap-summary \
  --lead-file-a=results/fuma/4ch/leadSNPs.txt \
  --lead-file-b=results/fuma/2ch/leadSNPs.txt \
  --output-prefix=results/post_gwas/4ch_vs_2ch
```

```bash
Rscript scripts/post_gwas_analysis.R \
  --task=function-enrichment \
  --input-file=results/fuma/4ch/magma.gsa.out \
  --output-file=results/post_gwas/4ch_function_enrichment.pdf
```

### Run post-GWAS analyses from config files

```bash
Rscript scripts/post_gwas_analysis.R \
  --config=configs/post_gwas_4ch.yaml
```

```bash
Rscript scripts/post_gwas_analysis.R \
  --config=configs/post_gwas_overlap.csv
```

Command-line flags override config values, so this is also valid:

```bash
Rscript scripts/post_gwas_analysis.R \
  --config=configs/post_gwas_4ch.yaml \
  --output-prefix=results/post_gwas/4ch_run2
```

## Input and output expectations

### Inputs

- Raw DICOM folders, one folder per subject or scan
- Segmentation masks aligned to the corresponding NIfTI volumes
- Optionally restricted-access genotype resources for downstream GWAS

### Outputs

- Cropped and registered cardiac NIfTI volumes
- Trained autoencoder checkpoints and training logs
- Participant-level latent phenotype tables
- Reconstruction metrics and perturbation maps
- GWAS phenotype files and min-P summary tables

## Limitations

- Segmentation is an external prerequisite and is not reproduced end-to-end in this repository.
- Some downstream GWAS commands require local institutional infrastructure and controlled-access genotype resources.
- The R workflow under `scripts/` has been standardized into task-based command-line analyses, but each task still expects project-specific file schemas such as FUMA or MAGMA outputs.

## Citation

If you use this code, please cite the associated preprint:

```text
You L, Zhao X, Xie Z, Patel KA, Chen C, Kitkungvan D, Mohammed KK,
Narula N, Arbustini E, Cassidy CK, Narula J, Zhi D.
Uncovering genetic architecture of the heart via genetic association
studies of unsupervised deep learning derived endophenotypes.
bioRxiv. 2025 Sep 20:2025.09.17.676827.
doi: 10.1101/2025.09.17.676827
```

Citation metadata is also available in [CITATION.cff](/Users/xzhao14/Documents/heart/Heart-UDIP/CITATION.cff#L1).

## License

This repository is distributed under the MIT License. See `LICENSE`.
