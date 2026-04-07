from __future__ import annotations

import json
import os
from pathlib import Path

import requests
import torch
from dotenv import load_dotenv
from PIL import Image
from transformers import (
    AutoTokenizer,
    Blip2ForConditionalGeneration,
    Blip2Processor,
    BlipImageProcessor,
)

try:
    from style_profile import build_style_paths, normalize_style_profile, save_style_profile
except ModuleNotFoundError:  # pragma: no cover - supports python -m scripts.generate_style
    from .style_profile import build_style_paths, normalize_style_profile, save_style_profile


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPERT_C_DIR = PROJECT_ROOT / "data" / "expert_c" / "edited"
EXPERT_D_DIR = PROJECT_ROOT / "data" / "expert_d" / "edited"
MODEL_ID = "Salesforce/blip2-opt-2.7b"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
CAPTION_COUNT = 50


def load_api_key() -> str:
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key or api_key == "YOUR_KEY_HERE":
        raise RuntimeError("OPENROUTER_API_KEY is missing or still a placeholder in .env")
    return api_key


def load_blip2() -> tuple[Blip2Processor, Blip2ForConditionalGeneration]:
    print(f"Loading BLIP-2 processor from {MODEL_ID}...", flush=True)
    image_processor = BlipImageProcessor.from_pretrained(MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
    processor = Blip2Processor(image_processor=image_processor, tokenizer=tokenizer)
    print(f"Loading BLIP-2 model from {MODEL_ID} on CPU...", flush=True)
    model = Blip2ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        device_map=None,
    )
    model = model.to("cpu")
    model.eval()
    return processor, model


def caption_images(
    edited_dir: Path,
    processor: Blip2Processor,
    model: Blip2ForConditionalGeneration,
    n: int = CAPTION_COUNT,
) -> list[str]:
    captions: list[str] = []
    files = sorted(path.name for path in edited_dir.glob("*.jpg"))[:n]
    for index, filename in enumerate(files, start=1):
        image = Image.open(edited_dir / filename).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to("cpu")
        with torch.inference_mode():
            generated_ids = model.generate(**inputs, max_new_tokens=60)
        caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        captions.append(caption)
        if index % 10 == 0:
            print(f"  {index}/{n}: {caption}", flush=True)
    return captions


def save_captions(path: Path | None, captions: list[str]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(captions, indent=2), encoding="utf-8")


def extract_json_object(response_text: str) -> dict[str, object]:
    """Parse a single JSON object from a model response."""

    content = response_text.strip()
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"Model response did not include a JSON object: {response_text}")
    return json.loads(content[start : end + 1])


def get_style_profile(captions: list[str], api_key: str, profile_name: str) -> dict[str, object]:
    """Generate a structured style profile from BLIP-2 captions."""

    caption_block = "\n".join(f"- {caption}" for caption in captions)
    response = requests.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://photo-styling-demo.local",
        },
        json={
            "model": "deepseek/deepseek-v3.2",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "These are auto-generated captions of photos edited by a professional photographer.\n"
                        "Return a JSON object only, with no markdown fences and no extra text.\n\n"
                        f'Use "profile_name": "{profile_name}".\n'
                        "Use exactly these top-level keys: profile_name, style_summary, facets.\n"
                        "Use exactly these facet keys: exposure, contrast, white_balance, saturation, "
                        "highlight_handling, shadow_handling, scene_tendency, mood.\n"
                        "Rules:\n"
                        "- style_summary must be one sentence that is specific and visual.\n"
                        "- each facet should be a short phrase grounded in the captions.\n"
                        "- if a facet is not evident, use an empty string.\n"
                        "- do not mention cameras, metadata, or editing software.\n\n"
                        "Captions:\n\n"
                        f"{caption_block}\n\n"
                        "Example structure:\n"
                        "{\n"
                        '  "profile_name": "cool_muted",\n'
                        '  "style_summary": "Cool, restrained, editorial real-estate finish.",\n'
                        '  "facets": {\n'
                        '    "exposure": "slightly lifted exposure with controlled brightness",\n'
                        '    "contrast": "moderate, soft contrast",\n'
                        '    "white_balance": "slightly cool white balance",\n'
                        '    "saturation": "muted saturation",\n'
                        '    "highlight_handling": "preserve bright window detail",\n'
                        '    "shadow_handling": "keep shadows open but not flat",\n'
                        '    "scene_tendency": "leans toward calm interiors when visible",\n'
                        '    "mood": "calm, documentary, restrained"\n'
                        "  }\n"
                        "}"
                    ),
                }
            ],
            "max_tokens": 300,
            "temperature": 0.2,
        },
        timeout=(10, 120),
    )
    response.raise_for_status()
    result = response.json()
    content = result["choices"][0]["message"]["content"].strip()
    raw_profile = extract_json_object(content)
    return normalize_style_profile(raw_profile, profile_name=profile_name)


def main() -> None:
    api_key = load_api_key()
    processor, model = load_blip2()
    caption_save_dir = PROJECT_ROOT / "artifacts" / "style_discovery"

    print("Captioning Expert C...", flush=True)
    captions_c = caption_images(EXPERT_C_DIR, processor, model)
    save_captions(caption_save_dir / "profile_c_captions.json", captions_c)

    print("Captioning Expert D...", flush=True)
    captions_d = caption_images(EXPERT_D_DIR, processor, model)
    save_captions(caption_save_dir / "profile_d_captions.json", captions_d)

    print("\nGenerating structured style profile for Expert C...", flush=True)
    style_c = get_style_profile(captions_c, api_key, "expert_c")
    print(f"Expert C summary: {style_c['style_summary']}", flush=True)
    print(f"Expert C composed prompt: {style_c['composed_prompt']}", flush=True)

    print("\nGenerating structured style profile for Expert D...", flush=True)
    style_d = get_style_profile(captions_d, api_key, "expert_d")
    print(f"Expert D summary: {style_d['style_summary']}", flush=True)
    print(f"Expert D composed prompt: {style_d['composed_prompt']}", flush=True)

    style_paths_c = build_style_paths(PROJECT_ROOT, "c")
    style_paths_d = build_style_paths(PROJECT_ROOT, "d")

    save_style_profile(style_paths_c["json"], style_c)
    save_style_profile(style_paths_d["json"], style_d)
    # Keep the old txt files around for legacy consumers.
    style_paths_c["txt"].write_text(style_c["style_summary"], encoding="utf-8")
    style_paths_d["txt"].write_text(style_d["style_summary"], encoding="utf-8")
    print("\nSaved style_c.json, style_d.json, and legacy style_c.txt/style_d.txt", flush=True)


if __name__ == "__main__":
    main()
