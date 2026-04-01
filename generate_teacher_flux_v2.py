from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

from flux_openrouter import build_teacher_edit_prompt, edit_image_with_flux, load_openrouter_api_key


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data"
STYLE_FILES = {
    "c": PROJECT_ROOT / "style_c.txt",
    "d": PROJECT_ROOT / "style_d.txt",
}
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "teacher_flux_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate FLUX.2 Pro teacher outputs for v2 distillation.")
    parser.add_argument("--expert", choices=["c", "d", "both"], default="both")
    parser.add_argument("--limit", type=int, default=200, help="How many sorted raw images to process per expert.")
    parser.add_argument("--start", type=int, default=1, help="1-based start index into the sorted raw files.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--sleep-seconds", type=float, default=0.0, help="Optional delay between API calls.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--preview-count", type=int, default=5)
    return parser.parse_args()


def load_style(expert: str) -> str:
    return STYLE_FILES[expert].read_text(encoding="utf-8").strip()


def list_raw_files(expert: str, start: int, limit: int) -> list[Path]:
    raw_dir = DATA_ROOT / f"expert_{expert}" / "raw"
    files = sorted(raw_dir.glob("*.jpg"))
    start_index = max(start - 1, 0)
    return files[start_index : start_index + limit]


def save_preview_grid(raw_files: list[Path], teacher_dir: Path, out_path: Path) -> None:
    preview_files = [path for path in raw_files if (teacher_dir / path.name).exists()]
    if not preview_files:
        return

    fig, axes = plt.subplots(len(preview_files), 2, figsize=(10, 4 * len(preview_files)))
    if len(preview_files) == 1:
        axes = [axes]

    for row, raw_path in enumerate(preview_files):
        raw_image = Image.open(raw_path).convert("RGB")
        teacher_image = Image.open(teacher_dir / raw_path.name).convert("RGB")
        axes[row][0].imshow(raw_image)
        axes[row][0].axis("off")
        axes[row][0].set_title(f"Raw {raw_path.name}")
        axes[row][1].imshow(teacher_image)
        axes[row][1].axis("off")
        axes[row][1].set_title("Teacher")

    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def format_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def generate_for_expert(
    api_key: str,
    expert: str,
    *,
    output_root: Path,
    limit: int,
    start: int,
    overwrite: bool,
    sleep_seconds: float,
    preview_count: int,
) -> None:
    style = load_style(expert)
    prompt = build_teacher_edit_prompt(style)
    raw_files = list_raw_files(expert, start, limit)
    if not raw_files:
        raise FileNotFoundError(f"No raw files found for expert {expert}")

    teacher_dir = output_root / f"expert_{expert}"
    teacher_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / f"expert_{expert}_manifest.json"
    manifest_records: list[dict[str, str | int]] = []

    print(f"\n=== Generating teacher outputs for expert {expert.upper()} ===", flush=True)
    print(f"Output directory: {teacher_dir}", flush=True)
    print(f"Files to process: {len(raw_files)}", flush=True)

    for index, raw_path in enumerate(raw_files, start=1):
        output_path = teacher_dir / raw_path.name
        if output_path.exists() and not overwrite:
            print(f"  [{index}/{len(raw_files)}] {raw_path.name} already exists, skipping", flush=True)
        else:
            print(f"  [{index}/{len(raw_files)}] editing {raw_path.name}", flush=True)
            image = edit_image_with_flux(api_key, prompt, image_path=raw_path)
            raw_size = Image.open(raw_path).convert("RGB").size
            if image.size != raw_size:
                image = image.resize(raw_size, Image.LANCZOS)
            image.save(output_path, format="JPEG", quality=95)
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

        manifest_records.append(
            {
                "expert": expert,
                "filename": raw_path.name,
                "raw_path": format_path(raw_path),
                "teacher_path": format_path(output_path),
            }
        )

    manifest_path.write_text(json.dumps(manifest_records, indent=2), encoding="utf-8")
    preview_path = output_root / f"expert_{expert}_preview.png"
    save_preview_grid(raw_files[:preview_count], teacher_dir, preview_path)
    print(f"Saved manifest to {manifest_path}", flush=True)
    print(f"Saved preview grid to {preview_path}", flush=True)


def main() -> None:
    args = parse_args()
    api_key = load_openrouter_api_key()
    experts = ["c", "d"] if args.expert == "both" else [args.expert]

    for expert in experts:
        generate_for_expert(
            api_key,
            expert,
            output_root=args.output_root,
            limit=args.limit,
            start=args.start,
            overwrite=args.overwrite,
            sleep_seconds=args.sleep_seconds,
            preview_count=args.preview_count,
        )


if __name__ == "__main__":
    main()
