import argparse
from pathlib import Path

import ants
import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Register cropped NIfTI volumes to a fixed ANTs template.")
    parser.add_argument("--input-dir", required=True, help="Directory containing cropped NIfTI files.")
    parser.add_argument("--template-path", required=True, help="Path to the fixed template NIfTI file.")
    parser.add_argument("--output-dir", required=True, help="Directory for registered images.")
    parser.add_argument("--tmp-dir", default="./ant_tmp_files", help="Directory prefix for ANTs temporary files.")
    parser.add_argument("--transform", default="Affine", help="ANTs transform type, e.g. Affine or SyN.")
    parser.add_argument("--start-index", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fixed_image = ants.image_read(args.template_path)

    nii_files = sorted(
        path for path in input_dir.iterdir() if path.is_file() and path.name.endswith((".nii", ".nii.gz"))
    )
    print(f"Found {len(nii_files)} NIfTI files for registration.")

    for index, nii_file in enumerate(nii_files):
        if index < args.start_index:
            continue
        print(f"Processing {index + 1}/{len(nii_files)}: {nii_file.name}")
        moving_image = ants.image_read(str(nii_file))
        if np.all(moving_image.numpy() == 0):
            print(f"Skipping {nii_file.name}: image is all zeros.")
            continue

        try:
            registration = ants.registration(
                fixed=fixed_image,
                moving=moving_image,
                type_of_transform=args.transform,
                outprefix=str(Path(args.tmp_dir) / f"{nii_file.stem}_"),
                restrict_transformation=(1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0),
            )
            output_path = output_dir / f"registered_{nii_file.name}"
            registration["warpedmovout"].to_file(str(output_path))
            print(f"Saved {output_path.name}")
        except Exception as error:  # pragma: no cover
            print(f"Registration failed for {nii_file.name}: {error}")


if __name__ == "__main__":
    main()
