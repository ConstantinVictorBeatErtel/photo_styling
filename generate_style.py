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


PROJECT_ROOT = Path(__file__).resolve().parent
EXPERT_C_DIR = PROJECT_ROOT / "data" / "expert_c" / "edited"
EXPERT_D_DIR = PROJECT_ROOT / "data" / "expert_d" / "edited"
STYLE_C_PATH = PROJECT_ROOT / "style_c.txt"
STYLE_D_PATH = PROJECT_ROOT / "style_d.txt"
CAPTIONS_C_PATH = PROJECT_ROOT / "captions_c.json"
CAPTIONS_D_PATH = PROJECT_ROOT / "captions_d.json"
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


def save_captions(path: Path, captions: list[str]) -> None:
    path.write_text(json.dumps(captions, indent=2), encoding="utf-8")


def get_style_prompt(captions: list[str], api_key: str) -> str:
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
                        "These are auto-generated captions of photos edited by a professional photographer:\n\n"
                        f"{caption_block}\n\n"
                        "Summarize their editing style in ONE sentence, as if describing how they edit to an AI photo editor.\n"
                        "Focus on: tone (warm/cool/neutral), brightness, contrast, color palette, and mood.\n"
                        'Be specific and visual. Do not use vague words like "professional" or "high quality".\n'
                        'Example format: "Bright, airy interiors with lifted shadows, warm white balance, and soft contrast."\n'
                        "Reply with only the sentence, nothing else."
                    ),
                }
            ],
            "max_tokens": 100,
            "temperature": 0.3,
        },
        timeout=(10, 120),
    )
    response.raise_for_status()
    result = response.json()
    return result["choices"][0]["message"]["content"].strip()


def main() -> None:
    api_key = load_api_key()
    processor, model = load_blip2()

    print("Captioning Expert C...", flush=True)
    captions_c = caption_images(EXPERT_C_DIR, processor, model)
    save_captions(CAPTIONS_C_PATH, captions_c)

    print("Captioning Expert D...", flush=True)
    captions_d = caption_images(EXPERT_D_DIR, processor, model)
    save_captions(CAPTIONS_D_PATH, captions_d)

    print("\nGenerating style prompt for Expert C...", flush=True)
    style_c = get_style_prompt(captions_c, api_key)
    print(f"Expert C style: {style_c}", flush=True)

    print("\nGenerating style prompt for Expert D...", flush=True)
    style_d = get_style_prompt(captions_d, api_key)
    print(f"Expert D style: {style_d}", flush=True)

    STYLE_C_PATH.write_text(style_c, encoding="utf-8")
    STYLE_D_PATH.write_text(style_d, encoding="utf-8")
    print("\nSaved style_c.txt and style_d.txt", flush=True)


if __name__ == "__main__":
    main()
