import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import subprocess
import nibabel as nib
import numpy as np
import pandas as pd
import torch
import umap
import matplotlib.pyplot as plt
from pathlib import Path
import shutil
import time
import sys
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

# --- Configuration & Paths ---
PROJECT_SRC = "./4Ch_models/Heart-UDIP/src"

if PROJECT_SRC not in sys.path:
    sys.path.append(PROJECT_SRC)

# Also update the environment for any subprocess calls
os.environ["PYTHONPATH"] = PROJECT_SRC + os.pathsep + os.environ.get("PYTHONPATH", "")
INPUT_FILE = "./test_nii/345_original_lax_4c_3d.nii.gz" # This is a four chamber view test data

# Use absolute paths for everything to avoid "0 cases" errors
BASE_DIR = Path("./4Ch_models/Heart-UDIP").resolve()
TEMP_DIR = BASE_DIR / "pipeline_temp"
# Create a dedicated, CLEAN folder for the input file
RAW_INPUT_DIR = TEMP_DIR / "raw_input" 
SEG_DIR = TEMP_DIR / "segmentation"

CROP_DIR = TEMP_DIR / "cropped"
REG_DIR = TEMP_DIR / "registered"

CSV_60K_PATH = "./CSV/UKB_60000_features.csv" # you need to generate your own feature files
TEMPLATE_PATH = "./ants_registration/4ch_template.nii.gz"# create the 4ch registration template with our codes
MODEL_WEIGHTS = "./model_4ch.pth" # load the 4ch view weight

# 1. Environment Setup (Absolute Paths Required)
os.environ["nnUNet_raw"] = str(BASE_DIR) # Placeholder
os.environ["nnUNet_preprocessed"] = "./nnUNet_preprocessed" # set the path to your own nnUnet 
os.environ["nnUNet_results"] = "./nnUNet_results" # set the path to your own nnUnet 



def run_command(cmd):
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    print(result.stdout)


def run_pipeline():
    
    # Reset directories for a clean run
    if TEMP_DIR.exists(): shutil.rmtree(TEMP_DIR)
    RAW_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    SEG_DIR.mkdir(parents=True, exist_ok=True)
    # --- Step 1: Preparation ---
    # [Your existing frame stacking code here]
    # --- Step 1: Frame Check and Stacking ---
    print("Step 1: Preparing image frames...")
    img_obj = nib.load(INPUT_FILE)
    data = img_obj.get_fdata(dtype=np.float32)

    # Handle 4D or 3D inputs (assuming time is last dim)
    if len(data.shape) == 3:
        h, w, t = data.shape
    elif len(data.shape) == 4:
        h, w, t, _ = data.shape
        data = data[:, :, :, 0]

    if t < 50:
        print(f"Stacking {t} frames to 50...")
        repeats = int(np.ceil(50 / t))
        data = np.tile(data, (1, 1, repeats))[:, :, :50]
    else:
        data = data[:, :, :50]

    # Save with the strict _0000 suffix for nnU-Net
    # Save ONLY to the raw_input folder
    nnunet_input_name = RAW_INPUT_DIR / "sample_0000.nii.gz"
    nib.save(nib.Nifti1Image(data, img_obj.affine, img_obj.header), str(nnunet_input_name))
    print(f"Input saved at: {nnunet_input_name}")
    
    # --- Step 2: nnU-Net Segmentation (Python API) ---
    print(f"Step 2: Running heart segmentation on GPU {os.environ.get('CUDA_VISIBLE_DEVICES')}...")
    
    model_folder = "./nnUNet_results/Dataset012_Heart4ch/nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres/"
    
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        verbose=False,
        allow_tqdm=True
    )

    predictor.initialize_from_trained_model_folder(
        model_folder,
        use_folds='all' 
    )

    try:
        # Note: Set num_processes to 0 or 1 if you continue to have RAM/Multiprocessing issues
        predictor.predict_from_files(
            [[str(nnunet_input_name)]], 
            [str(SEG_DIR / "sample")],
            save_probabilities=False,
            overwrite=True,
            num_processes_preprocessing=1, 
            num_processes_segmentation_export=1, 
            folder_with_segs_from_prev_stage=None,
            num_parts=1,
            part_id=0
        )
        print("Segmentation successful.")
    except Exception as e:
        print(f"Segmentation failed: {e}")
        return


    print("Step 2.5: Synchronizing filenames for cropping...")
    
    # 1. Fix the Mask Filename
    # If nnU-Net created 'sample.nii.gz.nii', rename it to 'sample.nii.gz'
    #weird_mask = SEG_DIR / "sample.nii.gz.nii"
    standard_mask = SEG_DIR / "sample.nii.gz"
    mask_file = nib.load(str(SEG_DIR / "sample.nii"))
    new_mask_file = nib.save(mask_file, str(SEG_DIR / "sample.nii.gz"))
    
    #if weird_mask.exists():
     #   os.rename(weird_mask, standard_mask)
    
    # 2. Fix the Input Image Filename
    # The script in Step 1 created 'sample_0000.nii.gz'. 
    # We must rename it to 'sample.nii.gz' to match the mask.
    nnunet_input = RAW_INPUT_DIR / "sample_0000.nii.gz"
    standard_input = RAW_INPUT_DIR / "sample.nii.gz"
    
    if nnunet_input.exists():
        os.rename(nnunet_input, standard_input)

    # --- Step 3: Cropping ---
    print("Step 3: Cropping...")
    # CRITICAL: Point --input-dir to RAW_INPUT_DIR, not the parent pipeline_temp
    crop_cmd = [
        "python", "-m", "heart_udip.preprocessing.crop_nifti",
        "--input-dir", str(RAW_INPUT_DIR), 
        "--mask-dir", str(SEG_DIR),
        "--output-dir", str(CROP_DIR)
    ]
    
    try:
        run_command(crop_cmd)
    except subprocess.CalledProcessError as e:
        print(f"Cropping failed! Error: {e.stderr}")
        # Print directory contents to debug if it fails again
        print(f"Inputs in {RAW_INPUT_DIR}: {os.listdir(RAW_INPUT_DIR)}")
        print(f"Masks in {SEG_DIR}: {os.listdir(SEG_DIR)}")
        exit(1)
    # --- Steps 3-7: Crop, Register, Extract, UMAP ---
    # [Your existing code for these steps here]
    
    

    # --- Step 4: Registration ---
    print("Step 4: Registering...")
    run_command([
        "python", "-m", "heart_udip.preprocessing.register_to_template",
        "--input-dir", str(CROP_DIR),
        "--template-path", TEMPLATE_PATH,
        "--output-dir", str(REG_DIR)
    ])
    
    # --- Step 5: Latent Extraction ---
    print("Step 5: Extracting Latent...")
    latent_csv = TEMP_DIR / "new_latent.csv"
    DEVICE = "cuda:3"
    run_command([
        "python", "-m", "heart_udip.evaluation.extract_latents",
        "--input-dir", str(REG_DIR),
        "--weights", MODEL_WEIGHTS,
        "--output-csv", str(latent_csv),
        "--device", "cuda:0"  # Add this flag
    ])
    
    # --- STEP 6: DATA REFORMATTING & UMAP ---
    print("Step 6: Formatting new features and running UMAP...")
    
    # Load the cohort and the new extraction
    df_60k = pd.read_csv(CSV_60K_PATH)
    df_new = pd.read_csv(latent_csv)
    
    # 1. Reformating df_new to match the 60k cohort format
    # Create the mapping for feature names (feature_1 -> Feature_1, etc.)
    feature_cols = [f"Feature_{i}" for i in range(1, 257)]
    rename_map = {f"feature_{i}": f"Feature_{i}" for i in range(1, 257)}
    df_new = df_new.rename(columns=rename_map)
    
    # Set Patient_ID as the original .nii.gz filename
    df_new['Patient_ID'] = Path(INPUT_FILE).name
    
    # Set Visit as "0"
    df_new['Visit'] = "0"
    
    # Remove 'filename' column if it exists in the new extraction
    if 'filename' in df_new.columns:
        df_new = df_new.drop(columns=['filename'])
        
    # Reorder columns to match the cohort structure: [Patient_ID, Visit, Feature_1...Feature_256]
    cols_order = ['Patient_ID', 'Visit'] + feature_cols
    df_new = df_new[cols_order]
    
    # Save the reformatted CSV for your future GWAS/records
    cleaned_csv = TEMP_DIR / "new_latent_standardized.csv"
    df_new.to_csv(cleaned_csv, index=False)
    print(f"Standardized features saved to {cleaned_csv}")

    # 2. Prepare Data for UMAP
    # Sample 5,000 from the 60k cohort to keep the plot responsive and clear
    cohort_latents = df_60k[feature_cols].sample(min(5000, len(df_60k)), random_state=42).values
    new_sample_latents = df_new[feature_cols].values
    
    # Combine for the UMAP projection
    combined_data = np.vstack([cohort_latents, new_sample_latents])
    
    print("Running UMAP dimensionality reduction...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    embedding = reducer.fit_transform(combined_data)

    # --- STEP 7: GENERATE AND SAVE PLOT ---
    print("Step 7: Saving UMAP visualization...")
    plt.figure(figsize=(10, 8))
    
    # Plot the background cohort (Grey)
    plt.scatter(embedding[:-1, 0], embedding[:-1, 1], 
                c='lightgrey', label='UKB 60k Cohort', alpha=0.4, s=3)
    
    # Plot your specific new sample (Red Star)
    plt.scatter(embedding[-1, 0], embedding[-1, 1], 
                c='red', label=f'Sample: {df_new["Patient_ID"].iloc[0]}', 
                edgecolors='black', s=150, marker='*')
    
    plt.title("Heart-UDIP Latent Projection: New Sample vs. 60k Cohort")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.legend()
    plt.grid(False)
    
    # Save to the current location as requested
    plt.savefig("pipeline_test.jpg", dpi=300, bbox_inches='tight')
    print("Success! Final plot saved as pipeline_test.jpg")

# THIS BLOCK IS MANDATORY TO FIX YOUR ERROR
if __name__ == '__main__':
    run_pipeline()