from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline
from PIL import Image
from skimage.metrics import structural_similarity


PROJECT_ROOT = Path("/Users/ConstiX/autohdr")
DATA_ROOT = PROJECT_ROOT / "data"
OUTPUT_ROOT = PROJECT_ROOT / "teacher_outputs"
MODEL_ID = "timbrooks/instruct-pix2pix"
NUM_INFERENCE_STEPS = 20
GUIDANCE_SCALE = 7.5
DEFAULT_IMAGE_GUIDANCE_SCALE_C = 2.5
DEFAULT_IMAGE_GUIDANCE_SCALE_D = 1.5
DEFAULT_SSIM_THRESHOLD = 0.45
DEFAULT_CONTENT_NEGATIVE_PROMPT = (
    "do not add or remove objects, do not change the scene, do not replace the background, "
    "no new buildings, no new houses, no new rooms, no new walls, no new furniture, "
    "no new grass fields, no landscape replacement, no geometry changes, no composition changes"
)


def detect_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_style(expert: str) -> str:
    style_path = PROJECT_ROOT / f"style_{expert}.txt"
    if not style_path.exists():
        raise FileNotFoundError(f"Missing style file: {style_path}")
    return style_path.read_text(encoding="utf-8").strip()


def list_training_images(
    expert: str,
    limit: int | None = None,
    filenames: list[str] | None = None,
) -> list[Path]:
    raw_dir = DATA_ROOT / f"expert_{expert}" / "raw"
    if not raw_dir.exists():
        raise FileNotFoundError(f"Missing raw image directory: {raw_dir}")

    if filenames:
        files = [raw_dir / name for name in filenames]
        missing = [str(path) for path in files if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing requested raw files: {missing}")
        return files

    files = sorted(raw_dir.glob("*.jpg"))[:200]
    if limit is not None:
        files = files[:limit]

    if not files:
        raise RuntimeError(f"No training images found in {raw_dir}")
    return files


def load_pipeline(device: str) -> StableDiffusionInstructPix2PixPipeline:
    print(f"Loading {MODEL_ID} on {device} with torch.float32...", flush=True)
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        safety_checker=None,
    )
    pipe.set_progress_bar_config(disable=True)
    pipe.enable_attention_slicing()
    if hasattr(pipe, "enable_model_cpu_offload") and device != "mps":
        # CPU offload is useful on non-MPS devices, but we keep the model fully on MPS when available.
        pass
    pipe = pipe.to(device)
    return pipe


def format_eta(seconds: float) -> str:
    remaining = max(0, int(seconds))
    hours, rem = divmod(remaining, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def default_image_guidance_scale(expert: str) -> float:
    if expert == "c":
        return DEFAULT_IMAGE_GUIDANCE_SCALE_C
    return DEFAULT_IMAGE_GUIDANCE_SCALE_D


def compute_ssim_score(source: Image.Image, generated: Image.Image) -> float:
    source_arr = np.asarray(source.convert("RGB"))
    generated_arr = np.asarray(generated.convert("RGB"))
    return float(
        structural_similarity(
            source_arr,
            generated_arr,
            channel_axis=-1,
            data_range=255,
        )
    )


def build_prompt(style: str, preserve_content: bool) -> str:
    if preserve_content:
        return (
            "Preserve the exact scene, composition, geometry, objects, and background. "
            "Apply only photographic edits such as exposure, white balance, contrast, saturation, "
            f"and mood to match this style: {style}"
        )
    return f"Edit this real estate photo: {style}"


def generate_for_expert(
    pipe: StableDiffusionInstructPix2PixPipeline,
    expert: str,
    limit: int | None = None,
    filenames: list[str] | None = None,
    overwrite: bool = False,
    image_guidance_scale: float | None = None,
    output_subdir: str | None = None,
    flagged_subdir: str | None = None,
    ssim_threshold: float | None = None,
    preserve_content: bool = False,
    negative_prompt: str | None = None,
) -> dict[str, object]:
    style = load_style(expert)
    prompt = build_prompt(style, preserve_content=preserve_content)
    raw_files = list_training_images(expert, limit=limit, filenames=filenames)
    output_dir = OUTPUT_ROOT / (output_subdir or f"expert_{expert}")
    output_dir.mkdir(parents=True, exist_ok=True)
    flagged_dir = OUTPUT_ROOT / (flagged_subdir or f"{output_dir.name}_flagged")
    flagged_dir.mkdir(parents=True, exist_ok=True)

    chosen_image_guidance_scale = (
        image_guidance_scale
        if image_guidance_scale is not None
        else default_image_guidance_scale(expert)
    )

    total = len(raw_files)
    started_at = time.time()
    completed = 0
    flagged = 0
    flagged_records: list[tuple[str, float]] = []

    print(f"\nStarting Expert {expert.upper()} teacher generation...", flush=True)
    print(f"Prompt: {prompt}", flush=True)
    print(f"Output directory: {output_dir}", flush=True)
    print(f"Flagged directory: {flagged_dir}", flush=True)
    print(f"Images to process: {total}", flush=True)
    print(f"Image guidance scale: {chosen_image_guidance_scale}", flush=True)
    print(f"Preserve content mode: {preserve_content}", flush=True)
    if negative_prompt:
        print(f"Negative prompt: {negative_prompt}", flush=True)
    if ssim_threshold is not None:
        print(f"SSIM threshold: {ssim_threshold}", flush=True)

    for index, raw_path in enumerate(raw_files, start=1):
        output_path = output_dir / raw_path.name
        flagged_path = flagged_dir / raw_path.name
        if (output_path.exists() or flagged_path.exists()) and not overwrite:
            completed += 1
            if completed % 10 == 0 or completed == total:
                elapsed = time.time() - started_at
                avg = elapsed / completed
                eta = avg * (total - completed)
                print(
                    f"  {completed}/{total} complete (skipped existing), ETA {format_eta(eta)}",
                    flush=True,
                )
            continue

        image = Image.open(raw_path).convert("RGB")
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            num_inference_steps=NUM_INFERENCE_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            image_guidance_scale=chosen_image_guidance_scale,
        ).images[0]

        destination = output_path
        if ssim_threshold is not None:
            ssim_score = compute_ssim_score(image, result)
            if ssim_score < ssim_threshold:
                flagged += 1
                flagged_records.append((raw_path.name, ssim_score))
                destination = flagged_path
                print(
                    f"  flagged {raw_path.name} with SSIM {ssim_score:.3f}",
                    flush=True,
                )

        result.save(destination, format="JPEG", quality=95)
        completed += 1

        if completed % 10 == 0 or completed == total:
            elapsed = time.time() - started_at
            avg = elapsed / completed
            eta = avg * (total - completed)
            print(f"  {completed}/{total} complete, ETA {format_eta(eta)}", flush=True)

    print(f"Finished Expert {expert.upper()} -> {output_dir}", flush=True)
    if ssim_threshold is not None:
        print(f"Flagged outputs: {flagged}", flush=True)
    return {
        "output_dir": str(output_dir),
        "flagged_dir": str(flagged_dir),
        "flagged_count": flagged,
        "flagged_records": flagged_records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate teacher outputs with InstructPix2Pix.")
    parser.add_argument(
        "--expert",
        choices=["c", "d", "all"],
        default="all",
        help="Which expert to generate teacher outputs for.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of images to process for a quick test run.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate outputs even if the target file already exists.",
    )
    parser.add_argument(
        "--filenames",
        nargs="*",
        default=None,
        help="Optional explicit filenames to generate, for smoke tests or targeted reruns.",
    )
    parser.add_argument(
        "--output-subdir",
        default=None,
        help="Optional subdirectory name under teacher_outputs/ for saved images.",
    )
    parser.add_argument(
        "--flagged-subdir",
        default=None,
        help="Optional subdirectory name under teacher_outputs/ for low-SSIM flagged images.",
    )
    parser.add_argument(
        "--image-guidance-scale",
        type=float,
        default=None,
        help="Optional override for image guidance scale. Expert C defaults to 2.5, Expert D to 1.5.",
    )
    parser.add_argument(
        "--ssim-threshold",
        type=float,
        default=None,
        help="Optional SSIM threshold. Outputs below this score go to the flagged folder instead of the main folder.",
    )
    parser.add_argument(
        "--preserve-content",
        action="store_true",
        help="Use a stricter prompt that tells the model to keep the exact scene and only apply photographic edits.",
    )
    parser.add_argument(
        "--negative-prompt",
        default=None,
        help="Optional negative prompt passed to the diffusion pipeline.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = detect_device()
    pipe = load_pipeline(device)

    experts = ["c", "d"] if args.expert == "all" else [args.expert]
    for expert in experts:
        generate_for_expert(
            pipe,
            expert=expert,
            limit=args.limit,
            filenames=args.filenames,
            overwrite=args.overwrite,
            image_guidance_scale=args.image_guidance_scale,
            output_subdir=args.output_subdir,
            flagged_subdir=args.flagged_subdir,
            ssim_threshold=args.ssim_threshold,
            preserve_content=args.preserve_content,
            negative_prompt=args.negative_prompt,
        )


if __name__ == "__main__":
    main()
