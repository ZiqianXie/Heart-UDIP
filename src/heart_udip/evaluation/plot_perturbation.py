import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib


def save_slices(nii_path: Path, output_dir: Path, threshold: float, step: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    image = nib.load(str(nii_path))
    data = image.get_fdata()
    if data.ndim != 3:
        raise ValueError(f"Expected a 3D NIfTI image, got shape {data.shape}.")

    for index in range(0, data.shape[2], step):
        slice_data = data[:, :, index].copy()
        slice_data[slice_data < threshold] = 0
        plt.figure(figsize=(6, 5))
        im = plt.imshow(slice_data, cmap="plasma", origin="upper")
        plt.title(f"Slice {index}", fontsize=14)
        cbar = plt.colorbar(im)
        cbar.set_label("Intensity", size=14)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_dir / f"slice_{index:03d}.png", dpi=150, bbox_inches="tight")
        plt.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export slice-wise PNGs for perturbation NIfTI maps.")
    parser.add_argument("--input-path", required=True, help="Single NIfTI file or directory of NIfTI files.")
    parser.add_argument("--output-dir", required=True, help="Directory for exported PNG slices.")
    parser.add_argument("--threshold", type=float, default=5.0)
    parser.add_argument("--step", type=int, default=1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_path = Path(args.input_path)
    output_root = Path(args.output_dir)

    if input_path.is_dir():
        for nii_file in sorted(path for path in input_path.iterdir() if path.name.endswith((".nii", ".nii.gz"))):
            save_slices(nii_file, output_root / nii_file.stem, args.threshold, args.step)
    else:
        save_slices(input_path, output_root, args.threshold, args.step)


if __name__ == "__main__":
    main()
