from __future__ import annotations

import argparse
import base64
import io
import json
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import requests
from dotenv import load_dotenv
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_ID = "black-forest-labs/flux.2-pro"
TEST_FILES = [f"{i:04d}.jpg" for i in range(201, 206)]
OUTPUT_CACHE_DIR = PROJECT_ROOT / "demo_flux_outputs"
REQUEST_TIMEOUT = (30, 900)
MAX_RETRIES = 4


def load_api_key() -> str:
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key or api_key == "YOUR_KEY_HERE":
        raise RuntimeError("OPENROUTER_API_KEY is missing or still a placeholder in .env")
    return api_key


def image_to_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def decode_image_data_url(data_url: str) -> Image.Image:
    header, encoded = data_url.split(",", 1)
    image_bytes = base64.b64decode(encoded)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def make_prompt(kind: str, style: str | None = None) -> str:
    if kind == "baseline":
        return (
            "Edit this photo while preserving the exact subject, pose, scene, composition, and lighting setup. "
            "Do not add, remove, or replace objects or backgrounds. "
            "Apply only tasteful professional photo enhancement: cleaner white balance, balanced exposure, subtle contrast, "
            "gentle highlight control, and natural detail."
        )
    if style is None:
        raise ValueError("style prompt requested without style text")
    return (
        "Edit this photo while preserving the exact subject, pose, scene, composition, and background. "
        "Do not add, remove, or replace objects or scenery. "
        "Apply only photographic color grading and tonal adjustments to match this style: "
        f"{style}"
    )


def edit_with_flux(api_key: str, image_path: Path, prompt: str) -> Image.Image:
    payload = {
        "model": MODEL_ID,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_to_data_url(image_path),
                        },
                    },
                ],
            }
        ],
        "modalities": ["image"],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://photo-styling-demo.local",
        "X-Title": "Photo Styling Demo",
    }

    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(
                OPENROUTER_URL,
                headers=headers,
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            result = response.json()
            if "choices" not in result:
                raise RuntimeError(f"OpenRouter did not return choices: {json.dumps(result)[:2000]}")
            images = result["choices"][0]["message"]["images"]
            data_url = images[0]["image_url"]["url"]
            return decode_image_data_url(data_url)
        except Exception as exc:
            last_error = exc
            if attempt == MAX_RETRIES:
                break
            wait_seconds = attempt * 10
            print(
                f"    Request failed on attempt {attempt}/{MAX_RETRIES}: {exc}. Retrying in {wait_seconds}s...",
                flush=True,
            )
            time.sleep(wait_seconds)

    assert last_error is not None
    raise last_error


def load_styles() -> tuple[str, str]:
    with open(PROJECT_ROOT / "style_c.txt", encoding="utf-8") as f:
        style_c = f.read().strip()
    with open(PROJECT_ROOT / "style_d.txt", encoding="utf-8") as f:
        style_d = f.read().strip()
    return style_c, style_d


def run_probe() -> Path:
    api_key = load_api_key()
    style_c, style_d = load_styles()

    image_path = PROJECT_ROOT / "data" / "expert_c" / "raw" / TEST_FILES[0]
    raw = Image.open(image_path).convert("RGB")

    print(f"Running FLUX.2 Pro probe on {image_path.name}...", flush=True)
    baseline = edit_with_flux(api_key, image_path, make_prompt("baseline"))
    print("  Baseline complete", flush=True)
    out_c = edit_with_flux(api_key, image_path, make_prompt("style", style_c))
    print("  Expert C complete", flush=True)
    out_d = edit_with_flux(api_key, image_path, make_prompt("style", style_d))
    print("  Expert D complete", flush=True)

    out_path = PROJECT_ROOT / "demo_flux_probe.png"
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    for ax, title, img in zip(
        axes,
        ["Raw", "Baseline FLUX", "Expert C FLUX", "Expert D FLUX"],
        [raw, baseline, out_c, out_d],
    ):
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}", flush=True)
    return out_path


def run_full_demo() -> Path:
    api_key = load_api_key()
    style_c, style_d = load_styles()
    OUTPUT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    style_c_label = (style_c[:60] + "...") if len(style_c) > 60 else style_c
    style_d_label = (style_d[:60] + "...") if len(style_d) > 60 else style_d

    col_labels = [
        "Raw Input",
        "Baseline FLUX.2 Pro",
        f"Expert C FLUX.2 Pro\n\"{style_c_label}\"",
        f"Expert D FLUX.2 Pro\n\"{style_d_label}\"",
    ]

    results: list[tuple[Image.Image, Image.Image, Image.Image, Image.Image]] = []
    for index, filename in enumerate(TEST_FILES, start=1):
        image_path = PROJECT_ROOT / "data" / "expert_c" / "raw" / filename
        raw = Image.open(image_path).convert("RGB")
        print(f"\nProcessing {filename} ({index}/{len(TEST_FILES)})...", flush=True)

        baseline_path = OUTPUT_CACHE_DIR / f"{image_path.stem}_baseline.jpg"
        if baseline_path.exists():
            baseline = Image.open(baseline_path).convert("RGB")
            print("  Baseline loaded from cache", flush=True)
        else:
            baseline = edit_with_flux(api_key, image_path, make_prompt("baseline"))
            baseline.save(baseline_path, format="JPEG", quality=95)
            print("  Baseline complete", flush=True)

        out_c_path = OUTPUT_CACHE_DIR / f"{image_path.stem}_expert_c.jpg"
        if out_c_path.exists():
            out_c = Image.open(out_c_path).convert("RGB")
            print("  Expert C loaded from cache", flush=True)
        else:
            out_c = edit_with_flux(api_key, image_path, make_prompt("style", style_c))
            out_c.save(out_c_path, format="JPEG", quality=95)
            print("  Expert C complete", flush=True)

        out_d_path = OUTPUT_CACHE_DIR / f"{image_path.stem}_expert_d.jpg"
        if out_d_path.exists():
            out_d = Image.open(out_d_path).convert("RGB")
            print("  Expert D loaded from cache", flush=True)
        else:
            out_d = edit_with_flux(api_key, image_path, make_prompt("style", style_d))
            out_d.save(out_d_path, format="JPEG", quality=95)
            print("  Expert D complete", flush=True)

        results.append((raw, baseline, out_c, out_d))

    out_path = PROJECT_ROOT / "demo_grid_flux.png"
    fig, axes = plt.subplots(5, 4, figsize=(24, 30))
    fig.suptitle(
        "Photo Styling with FLUX.2 Pro\nSame raw input -> personalized photographer styles",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )

    for row, images in enumerate(results):
        for col, img in enumerate(images):
            axes[row][col].imshow(img)
            axes[row][col].axis("off")
            if row == 0:
                axes[row][col].set_title(col_labels[col], fontsize=9, fontweight="bold", pad=10)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}", flush=True)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate FLUX.2 Pro demo outputs via OpenRouter.")
    parser.add_argument(
        "--probe",
        action="store_true",
        help="Only generate the single-image probe on 0201.jpg.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.probe:
        run_probe()
    else:
        run_full_demo()
