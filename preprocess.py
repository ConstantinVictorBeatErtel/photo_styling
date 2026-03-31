from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rawpy
import requests
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = PROJECT_ROOT / "fivek_dataset" / "MITAboveFiveK"
TRAINING_JSON = DATASET_ROOT / "training.json"
RAW_ROOT = DATASET_ROOT / "raw"
EXPERT_C_ROOT = DATASET_ROOT / "processed" / "tiff16_c"
EXPERT_D_ROOT = DATASET_ROOT / "processed" / "tiff16_d"
OUTPUT_ROOT = PROJECT_ROOT / "data"
IMAGE_SIZE = (512, 512)
LIMIT = 220
PROGRESS_EVERY = 20
REQUEST_TIMEOUT = (10, 120)


def output_paths() -> dict[str, Path]:
    paths = {
        "expert_c_raw": OUTPUT_ROOT / "expert_c" / "raw",
        "expert_c_edited": OUTPUT_ROOT / "expert_c" / "edited",
        "expert_d_raw": OUTPUT_ROOT / "expert_d" / "raw",
        "expert_d_edited": OUTPUT_ROOT / "expert_d" / "edited",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def raw_path_from_item(basename: str, item: dict) -> Path:
    camera_dir = f"{item['camera']['make']}_{item['camera']['model']}".replace(" ", "_")
    return RAW_ROOT / camera_dir / f"{basename}.dng"


def convert_raw_dng_to_rgb(path: Path) -> Image.Image:
    with rawpy.imread(str(path)) as raw:
        rgb = raw.postprocess(output_bps=8)
    return Image.fromarray(rgb, mode="RGB")


def redownload_file(url: str, path: Path) -> None:
    with requests.get(url, stream=True, timeout=REQUEST_TIMEOUT) as response:
        response.raise_for_status()
        with path.open("wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)


def convert_tiff_to_rgb(path: Path, url: str) -> Image.Image:
    with Image.open(path) as image:
        try:
            array = np.array(image)
        except OSError:
            # A previous interrupted download can leave a truncated TIFF on disk.
            redownload_file(url, path)
            with Image.open(path) as retry_image:
                array = np.array(retry_image)
    if array.dtype == np.uint16:
        array = (array / 256).clip(0, 255).astype(np.uint8)
    elif array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(array).convert("RGB")


def resize_rgb(image: Image.Image) -> Image.Image:
    return image.convert("RGB").resize(IMAGE_SIZE, Image.Resampling.LANCZOS)


def save_jpeg(image: Image.Image, path: Path) -> None:
    image.save(path, format="JPEG", quality=95)


def collect_first_220_matched() -> list[tuple[str, dict, Path, Path, Path]]:
    with TRAINING_JSON.open("r", encoding="utf-8") as file:
        training = json.load(file)

    matched: list[tuple[str, dict, Path, Path, Path]] = []
    for basename, item in training.items():
        raw_path = raw_path_from_item(basename, item)
        expert_c_path = EXPERT_C_ROOT / f"{basename}.tif"
        expert_d_path = EXPERT_D_ROOT / f"{basename}.tif"
        if raw_path.exists() and expert_c_path.exists() and expert_d_path.exists():
            matched.append((basename, item, raw_path, expert_c_path, expert_d_path))
        if len(matched) == LIMIT:
            break

    if len(matched) != LIMIT:
        raise RuntimeError(f"Expected {LIMIT} matched files, found {len(matched)}")

    print(f"Collected {len(matched)} matched official training items.")
    print(f"First basename: {matched[0][0]}")
    print(f"Last basename: {matched[-1][0]}")
    return matched


def main() -> None:
    paths = output_paths()
    matched = collect_first_220_matched()

    for index, (_, item, raw_path, expert_c_path, expert_d_path) in enumerate(matched, start=1):
        output_name = f"{index:04d}.jpg"

        raw_image = resize_rgb(convert_raw_dng_to_rgb(raw_path))
        expert_c_image = resize_rgb(
            convert_tiff_to_rgb(expert_c_path, item["urls"]["tiff16"]["c"])
        )
        expert_d_image = resize_rgb(
            convert_tiff_to_rgb(expert_d_path, item["urls"]["tiff16"]["d"])
        )

        save_jpeg(raw_image, paths["expert_c_raw"] / output_name)
        save_jpeg(raw_image, paths["expert_d_raw"] / output_name)
        save_jpeg(expert_c_image, paths["expert_c_edited"] / output_name)
        save_jpeg(expert_d_image, paths["expert_d_edited"] / output_name)

        if index % PROGRESS_EVERY == 0 or index == LIMIT:
            print(f"Processed {index}/{LIMIT}")

    print("\nFinal counts:")
    for label, path in paths.items():
        count = len(list(path.glob("*.jpg")))
        print(f"  {label}: {count}")

    print("\nSample dimensions:")
    for label, path in paths.items():
        sample = sorted(path.glob("*.jpg"))[0]
        with Image.open(sample) as image:
            print(f"  {label}: {sample.name} -> {image.size}, {image.mode}")


if __name__ == "__main__":
    main()
