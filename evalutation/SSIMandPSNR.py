import os
import random
import torch
import nibabel as nib
import numpy as np
from glob import glob
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
from model import CNN3D   # import your model here


# -----------------------------
# Evaluation Function
# -----------------------------
def evaluate_model(folder1, weight_path, device="cuda:0"):
    # load model
    model = CNN3D(in_channels=1, latent_dim=256).to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    # collect nii files
    nii_files = glob(os.path.join(folder1, "*.nii")) + glob(os.path.join(folder1, "*.nii.gz"))
    print(f"Found {len(nii_files)} nii files.")

    # randomly sample 500
    sample_files = random.sample(nii_files, min(500, len(nii_files)))

    ssim_scores, psnr_scores = [], []

    for nii_file in sample_files:
        img = nib.load(nii_file)
        data = img.get_fdata().astype(np.float32)

        # normalize [0,1]
        data_norm = data / (np.max(data) + 1e-8)

        # prepare tensor
        tensor = torch.tensor(data_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,D,H,W)

        with torch.no_grad():
            recon, _ = model(tensor)

        recon_np = recon.squeeze().cpu().numpy()

        # compute metrics (SSIM, PSNR)
        ssim_val = ssim(data_norm, recon_np, data_range=1.0)
        psnr_val = psnr(data_norm, recon_np, data_range=1.0)

        ssim_scores.append(ssim_val)
        psnr_scores.append(psnr_val)

    # calculate mean ± std
    ssim_mean, ssim_std = np.mean(ssim_scores), np.std(ssim_scores)
    psnr_mean, psnr_std = np.mean(psnr_scores), np.std(psnr_scores)

    print(f"SSIM: {ssim_mean:.4f} ± {ssim_std:.4f}")
    print(f"PSNR: {psnr_mean:.4f} ± {psnr_std:.4f}")


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    folder1 = "" # your test folder with nii files(80*80*50)
    pretrained_weights = ""# your path to the pth file

    evaluate_model(folder1, pretrained_weights, device="cuda:3" if torch.cuda.is_available() else "cpu")
