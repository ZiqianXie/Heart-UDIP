import argparse
from pathlib import Path

import nibabel as nib
import numpy as np


def list_nifti_files(folder: Path) -> list[Path]:
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.name.endswith((".nii", ".nii.gz"))
    )


def strip_nifti_suffix(filename: str) -> str:
    if filename.endswith(".nii.gz"):
        return filename[:-7]
    if filename.endswith(".nii"):
        return filename[:-4]
    return filename


def find_matching_mask(data_file: Path, mask_files: list[Path]) -> Path | None:
    data_key = strip_nifti_suffix(data_file.name).replace(".zip", "")
    for mask_file in mask_files:
        mask_key = strip_nifti_suffix(mask_file.name)
        if mask_key in data_key or data_key in mask_key:
            return mask_file
    return None


def crop_and_pad_volume(data: np.ndarray, mask: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    masked = data * mask
    nonzero_coords = np.argwhere(masked > 0)
    if nonzero_coords.size == 0:
        raise ValueError("Mask removed the entire volume.")

    min_coords = np.min(nonzero_coords, axis=0)
    max_coords = np.max(nonzero_coords, axis=0) + 1
    cropped = masked[min_coords[0] : max_coords[0], min_coords[1] : max_coords[1], :]

    target_x, target_y, target_z = target_shape
    cropped = cropped[:target_x, :target_y, :target_z]
    output = np.zeros(target_shape, dtype=cropped.dtype)
    start_x = (target_x - cropped.shape[0]) // 2
    start_y = (target_y - cropped.shape[1]) // 2
    start_z = (target_z - cropped.shape[2]) // 2
    output[
        start_x : start_x + cropped.shape[0],
        start_y : start_y + cropped.shape[1],
        start_z : start_z + cropped.shape[2],
    ] = cropped
    return output


def process_nifti_volumes(
    data_dir: Path,
    mask_dir: Path,
    output_dir: Path,
    target_shape: tuple[int, int, int],
    start_index: int = 0,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    data_files = list_nifti_files(data_dir)
    mask_files = list_nifti_files(mask_dir)

    processed = 0
    for index, data_file in enumerate(data_files):
        if index < start_index:
            continue

        mask_file = find_matching_mask(data_file, mask_files)
        if mask_file is None:
            print(f"Skipping {data_file.name}: no matching mask.")
            continue

        data_nii = nib.load(str(data_file))
        mask_nii = nib.load(str(mask_file))
        data = data_nii.get_fdata(dtype=np.float32)
        mask = mask_nii.get_fdata(dtype=np.float32)

        if data.shape[2] > target_shape[2]:
            print(f"Skipping {data_file.name}: depth {data.shape[2]} exceeds {target_shape[2]}.")
            continue

        try:
            cropped = crop_and_pad_volume(data, mask, target_shape)
        except ValueError as error:
            print(f"Skipping {data_file.name}: {error}")
            continue

        output_file = output_dir / f"cropped_{strip_nifti_suffix(data_file.name)}.nii.gz"
        nib.save(nib.Nifti1Image(cropped, affine=data_nii.affine), str(output_file))
        processed += 1
        print(f"[{processed}] saved {output_file.name}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Crop heart-centered NIfTI volumes using segmentation masks.")
    parser.add_argument("--input-dir", required=True, help="Directory containing original NIfTI volumes.")
    parser.add_argument("--mask-dir", required=True, help="Directory containing segmentation masks.")
    parser.add_argument("--output-dir", required=True, help="Directory for cropped heart volumes.")
    parser.add_argument("--target-shape", nargs=3, type=int, default=[80, 80, 50], metavar=("X", "Y", "Z"))
    parser.add_argument("--start-index", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    process_nifti_volumes(
        data_dir=Path(args.input_dir),
        mask_dir=Path(args.mask_dir),
        output_dir=Path(args.output_dir),
        target_shape=tuple(args.target_shape),
        start_index=args.start_index,
    )


if __name__ == "__main__":
    main()
