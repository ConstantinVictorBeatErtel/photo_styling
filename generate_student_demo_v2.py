from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, UNet2DConditionModel
from peft import PeftModel
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_ID = "timbrooks/instruct-pix2pix"
DEFAULT_STUDENT_ROOT = PROJECT_ROOT / "student_ip2p_v2"
DEFAULT_TEACHER_ROOT = PROJECT_ROOT / "teacher_flux_v2"


def get_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


DEVICE = get_device()


def build_student_prompt(style: str) -> str:
    return (
        "Apply this photographic style while preserving the exact scene, subject, composition, and objects: "
        f"{style}"
    )


def build_baseline_prompt() -> str:
    return (
        "Improve this photo with balanced exposure, natural color, clean white balance, and gentle contrast "
        "while preserving the original scene."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a v2 student comparison grid.")
    parser.add_argument("--student-root", type=Path, default=DEFAULT_STUDENT_ROOT)
    parser.add_argument("--teacher-root", type=Path, default=DEFAULT_TEACHER_ROOT)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--image-guidance-scale", type=float, default=2.0)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "demo_grid_student_v2.png")
    return parser.parse_args()


def load_pipe(student_dir: Path | None = None) -> StableDiffusionInstructPix2PixPipeline:
    if student_dir is None:
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float32,
            safety_checker=None,
        )
        return pipe.to(DEVICE)

    base_unet = UNet2DConditionModel.from_pretrained(MODEL_ID, subfolder="unet", torch_dtype=torch.float32)
    merged_unet = PeftModel.from_pretrained(base_unet, student_dir).merge_and_unload()
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        MODEL_ID,
        unet=merged_unet,
        torch_dtype=torch.float32,
        safety_checker=None,
    )
    return pipe.to(DEVICE)


def run_edit(
    pipe: StableDiffusionInstructPix2PixPipeline,
    raw_image: Image.Image,
    prompt: str,
    *,
    steps: int,
    guidance_scale: float,
    image_guidance_scale: float,
) -> Image.Image:
    result = pipe(
        prompt=prompt,
        image=raw_image,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        image_guidance_scale=image_guidance_scale,
    ).images[0]
    return result


def maybe_load_teacher_outputs(teacher_root: Path, filenames: list[str]) -> tuple[bool, list[Image.Image], list[Image.Image]]:
    teacher_c_paths = [teacher_root / "expert_c" / filename for filename in filenames]
    teacher_d_paths = [teacher_root / "expert_d" / filename for filename in filenames]
    available = all(path.exists() for path in teacher_c_paths + teacher_d_paths)
    if not available:
        return False, [], []
    teacher_c_images = [Image.open(path).convert("RGB") for path in teacher_c_paths]
    teacher_d_images = [Image.open(path).convert("RGB") for path in teacher_d_paths]
    return True, teacher_c_images, teacher_d_images


def main() -> None:
    args = parse_args()
    style_c = (PROJECT_ROOT / "style_c.txt").read_text(encoding="utf-8").strip()
    style_d = (PROJECT_ROOT / "style_d.txt").read_text(encoding="utf-8").strip()
    test_files = sorted((PROJECT_ROOT / "data" / "expert_c" / "raw").glob("*.jpg"))[200 : 200 + args.count]
    filenames = [path.name for path in test_files]

    print("Loading baseline pipeline...", flush=True)
    baseline_pipe = load_pipe()
    print("Loading student C pipeline...", flush=True)
    student_c_pipe = load_pipe(args.student_root / "expert_c")
    print("Loading student D pipeline...", flush=True)
    student_d_pipe = load_pipe(args.student_root / "expert_d")

    teacher_available, teacher_c_images, teacher_d_images = maybe_load_teacher_outputs(args.teacher_root, filenames)
    results = []

    for index, raw_path in enumerate(test_files, start=1):
        print(f"Processing {raw_path.name} ({index}/{len(test_files)})", flush=True)
        raw_image = Image.open(raw_path).convert("RGB")
        baseline = run_edit(
            baseline_pipe,
            raw_image,
            build_baseline_prompt(),
            steps=args.steps,
            guidance_scale=args.guidance_scale,
            image_guidance_scale=args.image_guidance_scale,
        )
        student_c = run_edit(
            student_c_pipe,
            raw_image,
            build_student_prompt(style_c),
            steps=args.steps,
            guidance_scale=args.guidance_scale,
            image_guidance_scale=args.image_guidance_scale,
        )
        student_d = run_edit(
            student_d_pipe,
            raw_image,
            build_student_prompt(style_d),
            steps=args.steps,
            guidance_scale=args.guidance_scale,
            image_guidance_scale=args.image_guidance_scale,
        )
        results.append((raw_image, baseline, student_c, student_d))

    if teacher_available:
        col_labels = ["Raw", "Baseline", "Teacher C", "Student C", "Teacher D", "Student D"]
        figure, axes = plt.subplots(len(results), len(col_labels), figsize=(28, 5 * len(results)))
    else:
        col_labels = ["Raw", "Baseline", "Student C", "Student D"]
        figure, axes = plt.subplots(len(results), len(col_labels), figsize=(20, 5 * len(results)))

    if len(results) == 1:
        axes = [axes]

    for row, (raw_image, baseline, student_c, student_d) in enumerate(results):
        row_images = [raw_image, baseline]
        if teacher_available:
            row_images.extend([teacher_c_images[row], student_c, teacher_d_images[row], student_d])
        else:
            row_images.extend([student_c, student_d])

        for col, image in enumerate(row_images):
            axes[row][col].imshow(image)
            axes[row][col].axis("off")
            if row == 0:
                axes[row][col].set_title(col_labels[col], fontweight="bold")

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
