import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchmetrics.functional import structural_similarity_index_measure as ssim_loss
from tqdm import tqdm

from heart_udip.models.cnn3d import CNN3D


class NiftiDataset(Dataset):
    def __init__(self, file_paths: list[Path]) -> None:
        self.file_paths = file_paths

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        image = nib.load(str(self.file_paths[idx])).get_fdata(dtype=np.float32)
        maximum = float(np.max(image))
        if maximum > 0:
            image = image / maximum
        image = np.expand_dims(image, axis=0)
        return torch.from_numpy(image)


def collect_nifti_files(folder_path: Path) -> list[Path]:
    return sorted(
        path
        for path in folder_path.iterdir()
        if path.is_file() and path.suffix in {".nii", ".gz"} and path.name.endswith((".nii", ".nii.gz"))
    )


def prepare_dataloaders(
    folder_path: Path,
    test_size: float,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    file_paths = collect_nifti_files(folder_path)
    if len(file_paths) < 2:
        raise ValueError("At least two NIfTI files are required for train/test splitting.")

    train_files, test_files = train_test_split(file_paths, test_size=test_size, random_state=seed)
    train_dataset = NiftiDataset(train_files)
    test_dataset = NiftiDataset(test_files)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader


def save_slice_preview(real_nii: torch.Tensor, fake_nii: torch.Tensor, output_dir: Path, epoch: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    slice_index = min(24, real_nii.shape[-1] - 1)
    real_slice = real_nii[:, :, slice_index].detach().cpu().numpy()
    fake_slice = fake_nii[:, :, slice_index].detach().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(real_slice, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(fake_slice, cmap="gray")
    axes[1].set_title("Reconstruction")
    axes[1].axis("off")
    plt.tight_layout()
    plt.savefig(output_dir / f"epoch_{epoch:03d}_slice_preview.png", dpi=200)
    plt.close(fig)


def apply_multi_mask(
    input_data: torch.Tensor,
    mask_size: tuple[int, int, int] = (50, 8, 8),
    num_masks: int = 75,
) -> tuple[torch.Tensor, torch.Tensor]:
    permuted = False
    if input_data.shape[4] < input_data.shape[2]:
        input_data = input_data.permute(0, 1, 4, 2, 3)
        permuted = True

    batch_size, _, depth, height, width = input_data.size()
    mask_d, mask_h, mask_w = mask_size
    if mask_d > depth or mask_h > height or mask_w > width:
        raise ValueError(f"Mask size {mask_size} exceeds input shape {(depth, height, width)}.")

    mask = torch.ones((batch_size, 1, depth, height, width), device=input_data.device)
    for batch_index in range(batch_size):
        used_coords = set()
        for _ in range(num_masks):
            max_d = depth - mask_d
            max_h = height - mask_h
            max_w = width - mask_w
            while True:
                coord = (
                    random.randint(0, max_d),
                    random.randint(0, max_h),
                    random.randint(0, max_w),
                )
                if coord not in used_coords:
                    used_coords.add(coord)
                    break
            d, h, w = coord
            mask[batch_index, :, d : d + mask_d, h : h + mask_h, w : w + mask_w] = 0

    masked_input = input_data * mask
    if permuted:
        masked_input = masked_input.permute(0, 1, 3, 4, 2)
    return masked_input, mask


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def train_model(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train_log.txt"

    train_loader, test_loader = prepare_dataloaders(
        folder_path=Path(args.input_dir),
        test_size=args.test_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    model = CNN3D(in_channels=1, latent_dim=args.latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()

    with log_path.open("w", encoding="utf-8") as log_file:
        for epoch in range(1, args.epochs + 1):
            model.train()
            train_loss = 0.0
            train_batches = 0
            last_batch = None
            last_reconstruction = None

            for batch in tqdm(train_loader, desc=f"Train {epoch}/{args.epochs}", leave=False):
                batch = batch.to(device, non_blocking=True)
                masked_input, _ = apply_multi_mask(
                    batch,
                    mask_size=tuple(args.mask_size),
                    num_masks=args.num_masks,
                )
                optimizer.zero_grad()
                reconstruction, _ = model(masked_input)
                loss_mse = criterion(reconstruction, batch)
                ssim = ssim_loss(reconstruction, batch, data_range=1.0)
                loss = loss_mse - args.ssim_weight * ssim
                loss.backward()
                optimizer.step()

                train_loss += float(loss.item())
                train_batches += 1
                last_batch = batch
                last_reconstruction = reconstruction

            model.eval()
            test_loss = 0.0
            test_batches = 0
            with torch.no_grad():
                for batch in tqdm(test_loader, desc=f"Eval {epoch}/{args.epochs}", leave=False):
                    batch = batch.to(device, non_blocking=True)
                    masked_input, _ = apply_multi_mask(
                        batch,
                        mask_size=tuple(args.mask_size),
                        num_masks=args.num_masks,
                    )
                    reconstruction, _ = model(masked_input)
                    loss_mse = criterion(reconstruction, batch)
                    ssim = ssim_loss(reconstruction, batch, data_range=1.0)
                    loss = loss_mse - args.ssim_weight * ssim
                    test_loss += float(loss.item())
                    test_batches += 1

            avg_train_loss = train_loss / max(train_batches, 1)
            avg_test_loss = test_loss / max(test_batches, 1)
            message = (
                f"Epoch [{epoch}/{args.epochs}] "
                f"train_loss={avg_train_loss:.4f} test_loss={avg_test_loss:.4f}"
            )
            print(message)
            log_file.write(message + "\n")
            log_file.flush()

            if last_batch is not None and last_reconstruction is not None:
                save_slice_preview(last_batch[0, 0], last_reconstruction[0, 0], output_dir, epoch)

            if epoch % args.save_every == 0 or epoch == args.epochs:
                checkpoint_path = output_dir / f"model_epoch_{epoch:03d}.pth"
                torch.save(model.state_dict(), checkpoint_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the 3D cine-MAE model.")
    parser.add_argument("--input-dir", required=True, help="Directory containing cropped NIfTI files.")
    parser.add_argument("--output-dir", required=True, help="Directory for logs, previews, and checkpoints.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", help="Device string, e.g. auto, cpu, cuda, cuda:0.")
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--num-masks", type=int, default=75)
    parser.add_argument("--mask-size", nargs=3, type=int, default=[50, 8, 8], metavar=("D", "H", "W"))
    parser.add_argument("--ssim-weight", type=float, default=0.01)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    train_model(args)


if __name__ == "__main__":
    main()
