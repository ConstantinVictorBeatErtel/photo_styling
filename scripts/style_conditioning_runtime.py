from __future__ import annotations

import sys
from pathlib import Path

import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from models.style_cross_attention import load_style_conditioner, style_conditioner_path


@torch.no_grad()
def encode_text_embeddings(tokenizer, text_encoder, prompt: str, device: str | torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = tokenizer(
        [prompt],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = tokens.input_ids.to(device)
    attention_mask = tokens.attention_mask.to(device)
    text_outputs = text_encoder(input_ids=input_ids, attention_mask=attention_mask)
    return text_outputs.last_hidden_state, attention_mask


def maybe_load_inference_style_conditioner(student_dir: Path, device: str | torch.device):
    checkpoint_path = style_conditioner_path(student_dir)
    if not checkpoint_path.exists():
        return None
    module = load_style_conditioner(checkpoint_path, map_location=device).to(device)
    module.eval()
    return module


@torch.no_grad()
def build_pipeline_conditioning(
    pipe,
    *,
    prompt: str,
    image: Image.Image,
    style_conditioner,
    device: str | torch.device,
) -> dict[str, torch.Tensor] | None:
    """Return precomputed prompt embeddings when the custom fusion module is available."""

    if style_conditioner is None:
        return None

    prompt_embeds, attention_mask = encode_text_embeddings(pipe.tokenizer, pipe.text_encoder, prompt, device)
    negative_prompt_embeds, _ = encode_text_embeddings(pipe.tokenizer, pipe.text_encoder, "", device)
    image_tensor = pipe.image_processor.preprocess(image).to(device=device, dtype=pipe.vae.dtype)
    source_latents = pipe.vae.encode(image_tensor).latent_dist.mode()

    augmented_prompt_embeds, _ = style_conditioner(source_latents, prompt_embeds, attention_mask)
    embed_dtype = getattr(pipe.text_encoder, "dtype", augmented_prompt_embeds.dtype)
    augmented_prompt_embeds = augmented_prompt_embeds.to(device=device, dtype=embed_dtype)
    negative_prompt_embeds = negative_prompt_embeds.to(device=device, dtype=embed_dtype)
    zero_fused_token = torch.zeros(
        negative_prompt_embeds.shape[0],
        1,
        negative_prompt_embeds.shape[-1],
        device=device,
        dtype=embed_dtype,
    )
    negative_prompt_embeds = torch.cat([negative_prompt_embeds, zero_fused_token], dim=1)
    return {
        "prompt_embeds": augmented_prompt_embeds,
        "negative_prompt_embeds": negative_prompt_embeds,
    }
