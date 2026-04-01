from __future__ import annotations

import base64
import io
import json
import os
import time
from pathlib import Path
from typing import Final

import requests
from PIL import Image

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional for Streamlit-only installs
    load_dotenv = None


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent
OPENROUTER_URL: Final[str] = "https://openrouter.ai/api/v1/chat/completions"
FLUX_MODEL_ID: Final[str] = "black-forest-labs/flux.2-pro"
DEFAULT_TIMEOUT: Final[tuple[int, int]] = (30, 900)
DEFAULT_MAX_RETRIES: Final[int] = 4


def load_openrouter_api_key(env_path: Path | None = None) -> str:
    if load_dotenv is not None:
        load_dotenv(env_path or PROJECT_ROOT / ".env")
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key or api_key == "YOUR_KEY_HERE":
        raise RuntimeError("OPENROUTER_API_KEY is missing or still a placeholder in .env")
    return api_key


def image_bytes_to_data_url(image_bytes: bytes, mime_type: str = "image/jpeg") -> str:
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def image_path_to_data_url(image_path: Path) -> str:
    suffix = image_path.suffix.lower()
    mime_type = "image/png" if suffix == ".png" else "image/jpeg"
    return image_bytes_to_data_url(image_path.read_bytes(), mime_type=mime_type)


def decode_image_data_url(data_url: str) -> Image.Image:
    _, encoded = data_url.split(",", 1)
    image_bytes = base64.b64decode(encoded)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def build_teacher_edit_prompt(style: str) -> str:
    return (
        "Edit this image while preserving the exact subject, pose, scene, composition, geometry, and background. "
        "Do not add, remove, replace, or hallucinate objects, scenery, structures, or people. "
        "Apply only photographic color grading and tonal adjustments to match this style: "
        f"{style}"
    )


def build_baseline_edit_prompt() -> str:
    return (
        "Edit this image while preserving the exact subject, pose, scene, composition, and background. "
        "Do not add, remove, or replace objects or scenery. "
        "Apply only tasteful photo enhancement with balanced exposure, natural white balance, "
        "gentle contrast, and clean color."
    )


def edit_image_with_flux(
    api_key: str,
    prompt: str,
    *,
    image_path: Path | None = None,
    image_bytes: bytes | None = None,
    mime_type: str = "image/jpeg",
    model_id: str = FLUX_MODEL_ID,
    timeout: tuple[int, int] = DEFAULT_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
    referer: str = "https://photo-styling-demo.local",
    title: str = "Photo Styling Demo",
) -> Image.Image:
    if (image_path is None) == (image_bytes is None):
        raise ValueError("Provide exactly one of image_path or image_bytes")

    if image_path is not None:
        image_url = image_path_to_data_url(image_path)
    else:
        image_url = image_bytes_to_data_url(image_bytes or b"", mime_type=mime_type)

    payload = {
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        "modalities": ["image"],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": referer,
        "X-Title": title,
    }

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(
                OPENROUTER_URL,
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
            result = response.json()
            if "choices" not in result:
                raise RuntimeError(f"OpenRouter did not return choices: {json.dumps(result)[:2000]}")
            images = result["choices"][0]["message"]["images"]
            data_url = images[0]["image_url"]["url"]
            return decode_image_data_url(data_url)
        except Exception as exc:  # pragma: no cover - depends on network/provider state
            last_error = exc
            if attempt == max_retries:
                break
            wait_seconds = attempt * 10
            print(
                f"Request failed on attempt {attempt}/{max_retries}: {exc}. Retrying in {wait_seconds}s...",
                flush=True,
            )
            time.sleep(wait_seconds)

    assert last_error is not None
    raise last_error
