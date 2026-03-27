import argparse
import random
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from heart_udip.models.cnn3d import CNN3D


def evaluate_model(input_dir: Path, weight_path: Path, sample_size: int, device: torch.device, seed: int) -> None:
    model = CNN3D(in_channels=1, latent_dim=256).to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    nii_files = sorted(path for path in input_dir.iterdir() if path.name.endswith((".nii", ".nii.gz")))
    print(f"Found {len(nii_files)} NIfTI files.")
    random.seed(seed)
    sample_files = random.sample(nii_files, min(sample_size, len(nii_files)))

    ssim_scores = []
    psnr_scores = []
    for nii_file in sample_files:
        data = nib.load(str(nii_file)).get_fdata(dtype=np.float32)
        data_norm = data / (np.max(data) + 1e-8)
        tensor = torch.tensor(data_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            recon, _ = model(tensor)

        recon_np = recon.squeeze().cpu().numpy()
        ssim_scores.append(ssim(data_norm, recon_np, data_range=1.0))
        psnr_scores.append(psnr(data_norm, recon_np, data_range=1.0))

    print(f"SSIM: {np.mean(ssim_scores):.4f} ± {np.std(ssim_scores):.4f}")
    print(f"PSNR: {np.mean(psnr_scores):.4f} ± {np.std(psnr_scores):.4f}")


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate reconstruction quality using SSIM and PSNR.")
    parser.add_argument("--input-dir", required=True, help="Directory containing test NIfTI files.")
    parser.add_argument("--weights", required=True, help="Path to a trained model checkpoint.")
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    evaluate_model(
        input_dir=Path(args.input_dir),
        weight_path=Path(args.weights),
        sample_size=args.sample_size,
        device=resolve_device(args.device),
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
