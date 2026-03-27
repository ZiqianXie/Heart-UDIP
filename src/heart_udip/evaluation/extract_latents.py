import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from heart_udip.models.cnn3d import CNN3D


class NiftiDataset(Dataset):
    def __init__(self, folder_path: Path) -> None:
        self.file_paths = []
        for file_path in sorted(folder_path.iterdir()):
            if not file_path.is_file() or not file_path.name.endswith((".nii", ".nii.gz")):
                continue
            img = nib.load(str(file_path)).get_fdata(dtype=np.float32)
            if np.max(img) > 0:
                self.file_paths.append(file_path)
            else:
                print(f"Skipping {file_path.name}: max value is zero.")

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        file_path = self.file_paths[idx]
        image = nib.load(str(file_path)).get_fdata(dtype=np.float32)
        image = image / np.max(image)
        image = np.expand_dims(image, axis=0)
        return torch.from_numpy(image), file_path.name


def parse_metadata(filename: str) -> tuple[str, str]:
    stem = filename[:-7] if filename.endswith(".nii.gz") else filename[:-4]
    tokens = stem.split("_")
    patient_id = tokens[0] if tokens else stem
    visit = tokens[2] if len(tokens) > 2 else "NA"
    return patient_id, visit


def extract_latent_features(
    model: CNN3D,
    dataloader: DataLoader,
    device: torch.device,
    output_csv: Path,
) -> None:
    rows = []
    latent_width = None
    model.eval()
    with torch.no_grad():
        for images, filenames in dataloader:
            images = images.to(device)
            _, latents = model(images)
            latents_np = latents.cpu().numpy()
            latent_width = latents_np.shape[1]
            for index, filename in enumerate(filenames):
                patient_id, visit = parse_metadata(filename)
                rows.append([patient_id, visit, filename] + latents_np[index].tolist())

    columns = ["patient_id", "visit", "filename"] + [f"feature_{index + 1}" for index in range(latent_width or 0)]
    pd.DataFrame(rows, columns=columns).to_csv(output_csv, index=False)
    print(f"Saved latent features to {output_csv}")


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract latent features from NIfTI volumes.")
    parser.add_argument("--input-dir", required=True, help="Directory containing registered or cropped NIfTI files.")
    parser.add_argument("--weights", required=True, help="Path to a trained model checkpoint.")
    parser.add_argument("--output-csv", required=True, help="CSV file to store latent features.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = resolve_device(args.device)
    model = CNN3D(in_channels=1, latent_dim=256)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(device)

    dataset = NiftiDataset(Path(args.input_dir))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    extract_latent_features(model, dataloader, device, Path(args.output_csv))


if __name__ == "__main__":
    main()
