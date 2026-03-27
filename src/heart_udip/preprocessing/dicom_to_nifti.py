import argparse
from pathlib import Path

import cv2
import nibabel as nib
import numpy as np
import pydicom


def load_dicom_images(folder_path: Path, error_log_path: Path | None = None) -> np.ndarray:
    dicom_files = sorted(path for path in folder_path.iterdir() if path.suffix.lower() == ".dcm")
    dicom_data = []
    for file_path in dicom_files:
        try:
            dicom = pydicom.dcmread(str(file_path))
            instance_number = getattr(dicom, "InstanceNumber", float("inf"))
            dicom_data.append((file_path, dicom, instance_number))
        except Exception as error:  # pragma: no cover
            if error_log_path is not None:
                with error_log_path.open("a", encoding="utf-8") as handle:
                    handle.write(f"Error reading {file_path}: {error}\n")
            print(f"Skipping unreadable file: {file_path}")

    dicom_data.sort(key=lambda item: item[2])
    images = []
    target_size = None
    for file_path, dicom, _ in dicom_data:
        try:
            image = dicom.pixel_array
            if target_size is None:
                target_size = image.shape
            if image.shape != target_size:
                image = cv2.resize(image, (target_size[1], target_size[0]))
            images.append(image)
        except Exception as error:  # pragma: no cover
            if error_log_path is not None:
                with error_log_path.open("a", encoding="utf-8") as handle:
                    handle.write(f"Error processing {file_path}: {error}\n")
            print(f"Skipping malformed pixel data: {file_path}")
    return np.asarray(images)


def convert_to_nifti(images: np.ndarray, output_path: Path) -> None:
    if images.ndim != 3:
        raise ValueError(f"Expected a 3D array after stacking DICOM slices, got shape {images.shape}.")
    nifti_image = nib.Nifti1Image(np.transpose(images, (1, 2, 0)), np.eye(4))
    nib.save(nifti_image, str(output_path))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert one DICOM folder per subject into NIfTI volumes.")
    parser.add_argument("--dicom-root", required=True, help="Root directory containing one subdirectory per subject.")
    parser.add_argument("--output-dir", required=True, help="Directory for converted .nii.gz files.")
    parser.add_argument("--error-log", default=None, help="Optional path for logging unreadable DICOM files.")
    parser.add_argument("--start-index", type=int, default=0, help="Skip subjects before this index.")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of subjects to process.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dicom_root = Path(args.dicom_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    error_log = Path(args.error_log) if args.error_log else None

    subject_dirs = sorted(path for path in dicom_root.iterdir() if path.is_dir())
    processed = 0
    for index, subject_dir in enumerate(subject_dirs):
        if index < args.start_index:
            continue
        if args.limit is not None and processed >= args.limit:
            break

        images = load_dicom_images(subject_dir, error_log)
        if images.size == 0:
            print(f"No readable DICOM slices found in {subject_dir}")
            continue

        output_path = output_dir / f"{subject_dir.name}.nii.gz"
        convert_to_nifti(images, output_path)
        processed += 1
        print(f"[{processed}] saved {output_path}")

    print(f"Converted {processed} subject folders.")


if __name__ == "__main__":
    main()
