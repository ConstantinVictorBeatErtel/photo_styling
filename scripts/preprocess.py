from __future__ import annotations

import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import rawpy
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "fivek_dataset" / "MITAboveFiveK"
TRAINING_JSON = DATASET_ROOT / "training.json"
OUTPUT_ROOT = PROJECT_ROOT / "data"
IMAGE_SIZE = (512, 512)
LIMIT = 220


def load_training_metadata() -> dict:
    with TRAINING_JSON.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def camera_dir(item: dict) -> str:
    make = item["camera"]["make"]
    model = item["camera"]["model"]
    return f"{make}_{model}".replace(" ", "_")


def raw_path_for(basename: str, item: dict) -> Path:
    return DATASET_ROOT / "raw" / camera_dir(item) / f"{basename}.dng"


def expert_path_for(basename: str, expert: str) -> Path:
    return DATASET_ROOT / "processed" / f"tiff16_{expert}" / f"{basename}.tif"


def convert_raw_dng_to_rgb(path: Path) -> Image.Image:
    with rawpy.imread(str(path)) as raw:
        rgb = raw.postprocess()
    return Image.fromarray(rgb).convert("RGB")


def convert_tiff16_to_rgb(path: Path) -> Image.Image:
    array = iio.imread(path)
    if array.dtype == np.uint16:
        array = (array / 256).astype(np.uint8)
    else:
        array = array.astype(np.uint8)
    return Image.fromarray(array).convert("RGB")


def resize_rgb(image: Image.Image) -> Image.Image:
    return image.resize(IMAGE_SIZE, Image.LANCZOS).convert("RGB")


def save_jpeg(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="JPEG", quality=95)


def main() -> None:
    metadata = load_training_metadata()
    basenames = list(metadata.keys())[:LIMIT]
    print(
        f"Using first {LIMIT} official training filenames "
        f"({basenames[0]} -> {basenames[-1]})"
    )

    matched: list[tuple[str, Path, Path, Path]] = []
    for basename in basenames:
        item = metadata[basename]
        raw_path = raw_path_for(basename, item)
        expert_c_path = expert_path_for(basename, "c")
        expert_d_path = expert_path_for(basename, "d")
        if raw_path.exists() and expert_c_path.exists() and expert_d_path.exists():
            matched.append((basename, raw_path, expert_c_path, expert_d_path))

    if len(matched) != LIMIT:
        raise RuntimeError(
            f"Expected {LIMIT} matched files across raw/expertC/expertD, found {len(matched)}"
        )

    output_dirs = {
        "expert_c_raw": OUTPUT_ROOT / "expert_c" / "raw",
        "expert_c_edited": OUTPUT_ROOT / "expert_c" / "edited",
        "expert_d_raw": OUTPUT_ROOT / "expert_d" / "raw",
        "expert_d_edited": OUTPUT_ROOT / "expert_d" / "edited",
    }
    for directory in output_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    for index, (basename, raw_path, expert_c_path, expert_d_path) in enumerate(
        matched, start=1
    ):
        out_name = f"{index:04d}.jpg"

        raw_img = resize_rgb(convert_raw_dng_to_rgb(raw_path))
        expert_c_img = resize_rgb(convert_tiff16_to_rgb(expert_c_path))
        expert_d_img = resize_rgb(convert_tiff16_to_rgb(expert_d_path))

        save_jpeg(raw_img, output_dirs["expert_c_raw"] / out_name)
        save_jpeg(raw_img, output_dirs["expert_d_raw"] / out_name)
        save_jpeg(expert_c_img, output_dirs["expert_c_edited"] / out_name)
        save_jpeg(expert_d_img, output_dirs["expert_d_edited"] / out_name)

        if index % 20 == 0 or index == LIMIT:
            print(f"Processed {index}/{LIMIT}: {basename} -> {out_name}")

    print("\nFinal counts:")
    for label, directory in output_dirs.items():
        count = len(list(directory.glob("*.jpg")))
        print(f"  {label}: {count}")

    print("\nSample dimensions:")
    for label, directory in output_dirs.items():
        sample_path = sorted(directory.glob("*.jpg"))[0]
        with Image.open(sample_path) as sample:
            print(f"  {label}: {sample_path.name} {sample.size} {sample.mode}")


if __name__ == "__main__":
    main()
