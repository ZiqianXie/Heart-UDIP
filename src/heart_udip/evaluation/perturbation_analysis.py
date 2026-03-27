import argparse
import random
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from scipy.stats import ttest_rel
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from heart_udip.models.cnn3d import CNN3D


class NiftiDataset(Dataset):
    def __init__(self, file_paths: list[Path]) -> None:
        self.file_paths = file_paths

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        image = nib.load(str(self.file_paths[idx])).get_fdata(dtype=np.float32)
        maximum = float(np.max(image))
        if maximum > 0:
            image = image / maximum
        return torch.tensor(image, dtype=torch.float32).unsqueeze(0), self.file_paths[idx].name


def load_nii_files(folder: Path, num_files: int, seed: int) -> list[Path]:
    all_files = sorted(path for path in folder.iterdir() if path.name.endswith((".nii", ".nii.gz")))
    random.seed(seed)
    return random.sample(all_files, min(num_files, len(all_files)))


def extract_features(model: CNN3D, dataloader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    enc2_list = []
    latent_list = []
    model.eval()
    with torch.no_grad():
        for image, _ in tqdm(dataloader, desc="Extract features", leave=False):
            image = image.to(device)
            enc1 = model.encoder_relu1(model.encoder_norm1(model.encoder_conv1(image)))
            enc2 = model.encoder_relu2(model.encoder_norm2(model.encoder_conv2(enc1)))
            enc3 = model.encoder_relu3(model.encoder_norm3(model.encoder_conv3(enc2)))
            latent = model.encoder_fc(model.flatten(enc3))
            enc2_list.append(enc2.cpu().numpy())
            latent_list.append(latent.cpu().numpy())
    return np.vstack(enc2_list), np.vstack(latent_list)


def decode(model: CNN3D, enc2_tensor: torch.Tensor, latent_tensor: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        dec_input = model.unflatten(model.decoder_fc(latent_tensor))
        dec1 = model.decoder_relu1(model.decoder_norm1(model.decoder_conv1(dec_input)))
        dec1 = 0.5 * enc2_tensor + 0.5 * dec1
        dec2 = model.decoder_relu2(model.decoder_norm2(model.decoder_conv2(dec1)))
        dec3 = model.decoder_sigmoid(model.decoder_conv3(dec2))
    return dec3


def save_nifti(attribution: np.ndarray, output_path: Path, affine: np.ndarray) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(attribution, affine), str(output_path))


def run_perturbation(
    model: CNN3D,
    enc2: np.ndarray,
    latent: np.ndarray,
    device: torch.device,
    affine: np.ndarray,
    output_dir: Path,
    batch_stride: int,
    sigma: float,
) -> None:
    latent_stds = np.std(latent, axis=0)
    for dim in tqdm(range(latent.shape[1]), desc="Perturb latent dims"):
        original_volumes = []
        perturbed_volumes = []
        for start in range(0, latent.shape[0], batch_stride):
            batch_enc2 = torch.from_numpy(enc2[start : start + batch_stride]).float().to(device)
            batch_latent = latent[start : start + batch_stride].copy()
            recon_original = decode(model, batch_enc2, torch.from_numpy(batch_latent).float().to(device))
            batch_latent[:, dim] += latent_stds[dim]
            recon_perturbed = decode(model, batch_enc2, torch.from_numpy(batch_latent).float().to(device))
            original_volumes.extend(recon_original.detach().cpu().numpy())
            perturbed_volumes.extend(recon_perturbed.detach().cpu().numpy())

        original_array = np.squeeze(np.asarray(original_volumes), axis=1)
        perturbed_array = np.squeeze(np.asarray(perturbed_volumes), axis=1)
        t_stat, _ = ttest_rel(original_array, perturbed_array, axis=0, nan_policy="omit")
        t_stat = gaussian_filter(np.abs(t_stat), sigma=sigma)
        save_nifti(t_stat, output_dir / f"{dim:03d}_paired_ttest.nii.gz", affine)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate perturbation-based latent attribution maps.")
    parser.add_argument("--input-dir", required=True, help="Directory containing test NIfTI files.")
    parser.add_argument("--weights", required=True, help="Path to a trained model checkpoint.")
    parser.add_argument("--output-dir", required=True, help="Directory for t-statistic NIfTI maps.")
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--batch-stride", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sigma", type=float, default=3.0)
    parser.add_argument("--device", default="auto")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = resolve_device(args.device)
    model = CNN3D(in_channels=1, latent_dim=256)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(device)

    nii_files = load_nii_files(Path(args.input_dir), num_files=args.sample_size, seed=args.seed)
    if not nii_files:
        raise ValueError("No NIfTI files found for perturbation analysis.")

    affine = nib.load(str(nii_files[0])).affine
    dataset = NiftiDataset(nii_files)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    enc2, latent = extract_features(model, dataloader, device)
    run_perturbation(
        model=model,
        enc2=enc2,
        latent=latent,
        device=device,
        affine=affine,
        output_dir=Path(args.output_dir),
        batch_stride=args.batch_stride,
        sigma=args.sigma,
    )


if __name__ == "__main__":
    main()
