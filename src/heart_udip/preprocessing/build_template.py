import argparse
from pathlib import Path

import ants


def collect_images(directory: Path, max_files: int) -> list:
    image_paths = sorted(
        path for path in directory.iterdir() if path.is_file() and path.name.endswith((".nii", ".nii.gz"))
    )[:max_files]
    return [ants.image_read(str(path)) for path in image_paths]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build an ANTs registration template from cropped NIfTI volumes.")
    parser.add_argument("--input-dir", required=True, help="Directory containing cropped NIfTI volumes.")
    parser.add_argument("--output-path", required=True, help="Path for the generated template NIfTI file.")
    parser.add_argument("--max-files", type=int, default=150)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    images = collect_images(Path(args.input_dir), max_files=args.max_files)
    if not images:
        raise ValueError("No NIfTI images found for template construction.")

    template_image = ants.build_template(
        image_list=images,
        useNoRigid=False,
        restrict_transformation=(1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0),
    )
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    template_image.to_file(str(output_path))
    print(f"Saved template to {output_path}")


if __name__ == "__main__":
    main()
