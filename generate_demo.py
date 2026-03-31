import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from diffusers import (
    StableDiffusionImg2ImgPipeline,
    AutoencoderKL,
    UNet2DConditionModel,
)
from peft import PeftModel

device = torch.device("mps")
MODEL_ID = "runwayml/stable-diffusion-v1-5"
DEMO_PROMPT = ""
DEMO_STRENGTH = 0.3

print("Loading baseline img2img pipeline...")
baseline_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,
    safety_checker=None,
    requires_safety_checker=False,
).to("mps")
baseline_pipe.enable_attention_slicing()


def load_lora_pipeline(lora_path):
    """Load SD img2img pipeline with a trained LoRA adapter merged into the UNet."""
    # Load base UNet
    base_unet = UNet2DConditionModel.from_pretrained(
        MODEL_ID, subfolder="unet", torch_dtype=torch.float32
    )
    # Load and merge LoRA weights
    lora_unet = PeftModel.from_pretrained(base_unet, lora_path)
    lora_unet = lora_unet.merge_and_unload()  # merge into base weights for inference
    lora_unet = lora_unet.to("mps")

    # Build pipeline with the merged UNet
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        MODEL_ID,
        unet=lora_unet,
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    ).to("mps")
    pipe.enable_attention_slicing()
    return pipe


print("Loading Expert C LoRA pipeline...")
pipe_c = load_lora_pipeline("lora_expert_c")

print("Loading Expert D LoRA pipeline...")
pipe_d = load_lora_pipeline("lora_expert_d")

with open("style_c.txt") as f:
    style_c = f.read().strip()
with open("style_d.txt") as f:
    style_d = f.read().strip()


def run_img2img(pipe, raw_image, prompt, strength=DEMO_STRENGTH, steps=30):
    """
    strength controls how much the output deviates from the raw input.
    0.0 = identical to input, 1.0 = fully regenerated.
    We use a lower value here because the LoRA was trained with a null prompt
    and should preserve the source scene while nudging it toward the learned style.
    """
    result = pipe(
        prompt=prompt,
        image=raw_image,
        strength=strength,
        num_inference_steps=steps,
        guidance_scale=7.5,
    ).images[0]
    return result


test_files = sorted(os.listdir("data/expert_c/raw"))[200:205]  # 0201-0205
print(f"Test files: {test_files}")
print(f"Using null prompt for inference: {repr(DEMO_PROMPT)}")
print(f"Using strength: {DEMO_STRENGTH}")

results = []
for i, fname in enumerate(test_files):
    print(f"\nProcessing {fname} ({i+1}/5)...")
    raw = Image.open(f"data/expert_c/raw/{fname}").convert("RGB")

    print("  Running baseline...")
    baseline = run_img2img(baseline_pipe, raw, DEMO_PROMPT)

    print("  Running Expert C LoRA...")
    out_c = run_img2img(pipe_c, raw, DEMO_PROMPT)

    print("  Running Expert D LoRA...")
    out_d = run_img2img(pipe_d, raw, DEMO_PROMPT)

    results.append((raw, baseline, out_c, out_d))

print("\nAll inference done. Building grid...")

style_c_label = (style_c[:60] + "...") if len(style_c) > 60 else style_c
style_d_label = (style_d[:60] + "...") if len(style_d) > 60 else style_d

col_labels = [
    "Raw Input",
    "Baseline\n(No LoRA)",
    f"Expert C LoRA\n\"{style_c_label}\"",
    f"Expert D LoRA\n\"{style_d_label}\"",
]

fig, axes = plt.subplots(5, 4, figsize=(24, 30))
fig.suptitle(
    "AutoHDR: Personalized Style Distillation\nSame raw input → two distinct photographer styles",
    fontsize=14, fontweight="bold", y=0.99
)

for row, (raw, baseline, out_c, out_d) in enumerate(results):
    for col, img in enumerate([raw, baseline, out_c, out_d]):
        axes[row][col].imshow(img)
        axes[row][col].axis("off")
        if row == 0:
            axes[row][col].set_title(
                col_labels[col], fontsize=9, fontweight="bold", pad=10
            )

plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig("demo_grid.png", dpi=150, bbox_inches="tight")
print("Saved demo_grid.png")
