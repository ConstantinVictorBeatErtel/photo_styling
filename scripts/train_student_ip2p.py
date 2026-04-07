from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import CLIPTextModel, CLIPTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from models.style_cross_attention import (
    StyleConditioningAdapter,
    load_style_conditioner,
    save_style_conditioner,
    style_conditioner_path,
)

try:
    from style_profile import build_style_paths, load_style_profile
except ModuleNotFoundError:  # pragma: no cover - supports python -m scripts.train_student_ip2p
    from .style_profile import build_style_paths, load_style_profile


ASSETS_ROOT = PROJECT_ROOT / "assets"
MODEL_ID = "timbrooks/instruct-pix2pix"
DEFAULT_TEACHER_ROOT = PROJECT_ROOT / "teacher_kontext_v2"
DEFAULT_SAVE_ROOT = PROJECT_ROOT / "student_ip2p_v2_r8"
DEFAULT_LOSS_OUTPUT = ASSETS_ROOT / "loss_curves_v2_r8.png"
STYLE_FILES = {
    "c": build_style_paths(PROJECT_ROOT, "c"),
    "d": build_style_paths(PROJECT_ROOT, "d"),
}


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()


def format_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def build_student_prompt(style_prompt: str) -> str:
    return (
        "Apply this photographic style while preserving the exact scene, subject, composition, and objects: "
        f"{style_prompt}"
    )


class TeacherPairDataset(Dataset):
    def __init__(self, raw_dir: Path, teacher_dir: Path, limit: int | None = None):
        raw_files = {path.name for path in raw_dir.glob("*.jpg")}
        teacher_files = {path.name for path in teacher_dir.glob("*.jpg")}
        self.files = sorted(raw_files & teacher_files)
        if limit is not None:
            self.files = self.files[:limit]
        if not self.files:
            raise ValueError(f"No matched training pairs found in {raw_dir} and {teacher_dir}")
        self.raw_dir = raw_dir
        self.teacher_dir = teacher_dir
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        filename = self.files[index]
        raw_image = Image.open(self.raw_dir / filename).convert("RGB")
        target_image = Image.open(self.teacher_dir / filename).convert("RGB")
        if target_image.size != raw_image.size:
            target_image = target_image.resize(raw_image.size, Image.LANCZOS)
        raw = self.transform(raw_image)
        target = self.transform(target_image)
        return {"raw": raw, "target": target, "filename": filename}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the current best edit-conditioned student LoRAs.")
    parser.add_argument("--experts", nargs="+", choices=["c", "d"], default=["c", "d"])
    parser.add_argument("--teacher-root", type=Path, default=DEFAULT_TEACHER_ROOT)
    parser.add_argument("--save-root", type=Path, default=DEFAULT_SAVE_ROOT)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--checkpoint-interval", type=int, default=25)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument(
        "--disable-style-cross-attention",
        action="store_true",
        help="Use the original plain text conditioning path without the custom fusion block.",
    )
    parser.add_argument("--style-fusion-hidden-dim", type=int, default=256)
    parser.add_argument("--style-fusion-heads", type=int, default=4)
    parser.add_argument("--style-source-grid-size", type=int, default=16)
    parser.add_argument("--style-fusion-mlp-ratio", type=float, default=2.0)
    parser.add_argument("--loss-output", type=Path, default=DEFAULT_LOSS_OUTPUT)
    parser.add_argument("--disable-plot", action="store_true")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore existing checkpoints and train from scratch.",
    )
    return parser.parse_args()


def load_frozen_components() -> tuple[CLIPTokenizer, CLIPTextModel, AutoencoderKL, DDPMScheduler]:
    tokenizer = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(MODEL_ID, subfolder="text_encoder").to(DEVICE)
    text_encoder.eval()
    for parameter in text_encoder.parameters():
        parameter.requires_grad = False

    vae = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae").to(DEVICE)
    vae.eval()
    for parameter in vae.parameters():
        parameter.requires_grad = False

    scheduler = DDPMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")
    return tokenizer, text_encoder, vae, scheduler


def build_lora_config(rank: int, lora_alpha: int, lora_dropout: float) -> LoraConfig:
    return LoraConfig(
        r=rank,
        lora_alpha=lora_alpha,
        target_modules=["to_q", "to_v"],
        lora_dropout=lora_dropout,
        bias="none",
    )


def load_trainable_unet(
    checkpoint_dir: Path | None = None,
    *,
    rank: int,
    lora_alpha: int,
    lora_dropout: float,
) -> UNet2DConditionModel:
    unet = UNet2DConditionModel.from_pretrained(MODEL_ID, subfolder="unet").to(DEVICE)
    if checkpoint_dir is not None:
        unet = PeftModel.from_pretrained(unet, checkpoint_dir, is_trainable=True)
        print(f"Loaded checkpoint from {checkpoint_dir}", flush=True)
        unet.print_trainable_parameters()
        return unet

    lora_config = build_lora_config(rank, lora_alpha, lora_dropout)
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()
    return unet


def encode_prompt(
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    prompt: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        tokens = tokenizer(
            [prompt],
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).to(DEVICE)
        return text_encoder(**tokens).last_hidden_state, tokens.attention_mask


def encode_source_image_latents(vae: AutoencoderKL, images: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return vae.encode(images).latent_dist.mode()


def encode_target_latents(vae: AutoencoderKL, images: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        latents = vae.encode(images).latent_dist.sample()
        return latents * vae.config.scaling_factor


def save_metadata(
    out_dir: Path,
    *,
    expert: str,
    style_profile: dict[str, object],
    training_prompt: str,
    style_conditioning_enabled: bool,
    style_conditioner_config: dict[str, object] | None,
    teacher_dir: Path,
    dataset_size: int,
    steps: int,
    learning_rate: float,
) -> None:
    metadata = {
        "expert": expert,
        "style": style_profile["style_summary"],
        "style_profile": style_profile,
        "composed_style_prompt": style_profile["composed_prompt"],
        "training_prompt": training_prompt,
        "style_conditioning_enabled": style_conditioning_enabled,
        "style_conditioner_config": style_conditioner_config,
        "teacher_dir": format_path(teacher_dir),
        "dataset_size": dataset_size,
        "steps": steps,
        "learning_rate": learning_rate,
        "model_id": MODEL_ID,
    }
    (out_dir / "distillation_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def save_checkpoint(
    *,
    unet: UNet2DConditionModel,
    style_conditioner: StyleConditioningAdapter | None,
    optimizer: torch.optim.Optimizer,
    out_dir: Path,
    expert: str,
    completed_steps: int,
    total_steps: int,
    losses: list[float],
    style_conditioning_enabled: bool,
) -> None:
    checkpoints_dir = out_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = checkpoints_dir / f"step_{completed_steps:04d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    unet.save_pretrained(checkpoint_dir)
    if style_conditioner is not None:
        save_style_conditioner(style_conditioner_path(checkpoint_dir), style_conditioner)

    state = {
        "expert": expert,
        "completed_steps": completed_steps,
        "total_steps": total_steps,
        "losses": losses,
        "style_conditioning_enabled": style_conditioning_enabled,
        "optimizer_state_dict": optimizer.state_dict(),
        "checkpoint_dir": str(checkpoint_dir),
        "saved_at": time.time(),
    }
    torch.save(state, out_dir / "latest_training_state.pt")
    print(f"Saved checkpoint for expert {expert.upper()} at step {completed_steps}", flush=True)


def load_resume_state(out_dir: Path) -> dict | None:
    state_path = out_dir / "latest_training_state.pt"
    if not state_path.exists():
        return None
    return torch.load(state_path, map_location="cpu")


def build_style_conditioner(
    *,
    checkpoint_dir: Path | None,
    source_dim: int,
    style_dim: int,
    hidden_dim: int,
    num_heads: int,
    source_grid_size: int,
    mlp_ratio: float,
) -> StyleConditioningAdapter:
    if checkpoint_dir is not None:
        checkpoint_path = style_conditioner_path(checkpoint_dir)
        if checkpoint_path.exists():
            return load_style_conditioner(checkpoint_path, map_location=DEVICE).to(DEVICE)

    return StyleConditioningAdapter(
        source_dim=source_dim,
        style_dim=style_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        source_grid_size=source_grid_size,
        mlp_ratio=mlp_ratio,
    ).to(DEVICE)


def train_one_expert(
    *,
    expert: str,
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    vae: AutoencoderKL,
    scheduler: DDPMScheduler,
    steps: int,
    batch_size: int,
    gradient_accumulation: int,
    learning_rate: float,
    weight_decay: float,
    teacher_root: Path,
    save_root: Path,
    limit: int | None,
    checkpoint_interval: int,
    rank: int,
    lora_alpha: int,
    lora_dropout: float,
    use_style_cross_attention: bool,
    style_fusion_hidden_dim: int,
    style_fusion_heads: int,
    style_source_grid_size: int,
    style_fusion_mlp_ratio: float,
    fresh: bool,
) -> list[float]:
    style_profile = load_style_profile(
        STYLE_FILES[expert]["json"],
        STYLE_FILES[expert]["txt"],
        profile_name=f"expert_{expert}",
    )
    prompt = build_student_prompt(style_profile["composed_prompt"])
    prompt_embeds, prompt_attention_mask = encode_prompt(tokenizer, text_encoder, prompt)

    raw_dir = PROJECT_ROOT / "data" / f"expert_{expert}" / "raw"
    teacher_dir = teacher_root / f"expert_{expert}"
    dataset = TeacherPairDataset(raw_dir, teacher_dir, limit=limit)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    out_dir = save_root / f"expert_{expert}"
    out_dir.mkdir(parents=True, exist_ok=True)
    resume_state = None if fresh else load_resume_state(out_dir)

    checkpoint_dir = None
    completed_steps = 0
    losses: list[float] = []
    if resume_state is not None:
        completed_steps = int(resume_state.get("completed_steps", 0))
        checkpoint_dir_value = resume_state.get("checkpoint_dir")
        checkpoint_dir = Path(checkpoint_dir_value) if checkpoint_dir_value else None
        losses = list(resume_state.get("losses", []))
        resume_style_conditioning = bool(resume_state.get("style_conditioning_enabled", False))
        if resume_style_conditioning != use_style_cross_attention:
            raise ValueError(
                "Resume checkpoint style-conditioning setting does not match the current run. "
                "Use the same setting or pass --fresh."
            )
        if completed_steps >= steps and (out_dir / "adapter_model.safetensors").exists():
            print(f"Expert {expert.upper()} already finished at step {completed_steps}, skipping.", flush=True)
            return losses

    unet = load_trainable_unet(
        checkpoint_dir=checkpoint_dir,
        rank=rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )
    style_conditioner = None
    trainable_parameters = [parameter for parameter in unet.parameters() if parameter.requires_grad]
    if use_style_cross_attention:
        style_conditioner = build_style_conditioner(
            checkpoint_dir=checkpoint_dir,
            source_dim=int(vae.config.latent_channels),
            style_dim=prompt_embeds.shape[-1],
            hidden_dim=style_fusion_hidden_dim,
            num_heads=style_fusion_heads,
            source_grid_size=style_source_grid_size,
            mlp_ratio=style_fusion_mlp_ratio,
        )
        style_conditioner.train()
        trainable_parameters.extend(parameter for parameter in style_conditioner.parameters() if parameter.requires_grad)

    optimizer = torch.optim.AdamW(trainable_parameters, lr=learning_rate, weight_decay=weight_decay)
    if resume_state is not None and "optimizer_state_dict" in resume_state:
        optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        print(f"Resuming expert {expert.upper()} from step {completed_steps}", flush=True)

    print(f"\n=== Training student for expert {expert.upper()} ===", flush=True)
    print(f"Matched teacher pairs: {len(dataset)}", flush=True)
    print(f"Prompt: {prompt}", flush=True)
    if style_conditioner is not None:
        print(
            "Style cross-attention enabled: source VAE latents attend over style text tokens before the UNet.",
            flush=True,
        )

    data_iter = iter(loader)
    optimizer.zero_grad(set_to_none=True)
    start_time = time.time()
    unet.train()

    for step in range(completed_steps, steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        raw = batch["raw"].to(DEVICE)
        target = batch["target"].to(DEVICE)

        source_latents = encode_source_image_latents(vae, raw)
        target_latents = encode_target_latents(vae, target)

        noise = torch.randn_like(target_latents)
        timesteps = torch.randint(
            0,
            scheduler.config.num_train_timesteps,
            (target_latents.shape[0],),
            device=DEVICE,
        ).long()
        noisy_target_latents = scheduler.add_noise(target_latents, noise, timesteps)
        model_input = torch.cat([noisy_target_latents, source_latents], dim=1)
        prompt_batch = prompt_embeds.expand(target_latents.shape[0], -1, -1)
        prompt_mask_batch = prompt_attention_mask.expand(target_latents.shape[0], -1)
        if style_conditioner is not None:
            prompt_batch, _ = style_conditioner(source_latents, prompt_batch, prompt_mask_batch)

        noise_pred = unet(
            model_input,
            timesteps,
            encoder_hidden_states=prompt_batch,
        ).sample

        if scheduler.config.prediction_type == "v_prediction":
            target_tensor = scheduler.get_velocity(target_latents, noise, timesteps)
        else:
            target_tensor = noise

        loss = torch.nn.functional.mse_loss(noise_pred, target_tensor)
        (loss / gradient_accumulation).backward()

        if (step + 1) % gradient_accumulation == 0:
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        losses.append(loss.item())

        if (step + 1) % 50 == 0:
            avg_loss = sum(losses[-50:]) / min(len(losses[-50:]), 50)
            elapsed_minutes = (time.time() - start_time) / 60
            remaining_minutes = elapsed_minutes * max(steps - step - 1, 0) / max(step + 1, 1)
            print(
                f"  [Expert {expert.upper()}] Step {step + 1}/{steps} | Loss: {avg_loss:.4f} | ~{remaining_minutes:.0f} min left",
                flush=True,
            )

        if (step + 1) % checkpoint_interval == 0 and (step + 1) < steps:
            save_checkpoint(
                unet=unet,
                style_conditioner=style_conditioner,
                optimizer=optimizer,
                out_dir=out_dir,
                expert=expert,
                completed_steps=step + 1,
                total_steps=steps,
                losses=losses,
                style_conditioning_enabled=style_conditioner is not None,
            )

    unet.save_pretrained(out_dir)
    if style_conditioner is not None:
        save_style_conditioner(style_conditioner_path(out_dir), style_conditioner)
    save_checkpoint(
        unet=unet,
        style_conditioner=style_conditioner,
        optimizer=optimizer,
        out_dir=out_dir,
        expert=expert,
        completed_steps=steps,
        total_steps=steps,
        losses=losses,
        style_conditioning_enabled=style_conditioner is not None,
    )
    save_metadata(
        out_dir,
        expert=expert,
        style_profile=style_profile,
        training_prompt=prompt,
        style_conditioning_enabled=style_conditioner is not None,
        style_conditioner_config=style_conditioner.get_config() if style_conditioner is not None else None,
        teacher_dir=teacher_dir,
        dataset_size=len(dataset),
        steps=steps,
        learning_rate=learning_rate,
    )
    print(f"Saved student LoRA to {out_dir}", flush=True)

    unet.to("cpu")
    del unet
    if style_conditioner is not None:
        style_conditioner.to("cpu")
        del style_conditioner
    if DEVICE.type == "mps":
        torch.mps.empty_cache()
    return losses


def save_loss_plot(losses_by_expert: dict[str, list[float]], out_path: Path) -> None:
    def moving_average(values: list[float], window: int) -> list[float]:
        if not values:
            return []
        if window <= 1 or len(values) < window:
            return values[:]
        averaged: list[float] = []
        running = sum(values[:window])
        averaged.extend([values[index] for index in range(window - 1)])
        averaged.append(running / window)
        for index in range(window, len(values)):
            running += values[index] - values[index - window]
            averaged.append(running / window)
        return averaged

    fig, axes = plt.subplots(1, len(losses_by_expert), figsize=(6 * len(losses_by_expert), 4))
    if len(losses_by_expert) == 1:
        axes = [axes]

    for axis, (expert, losses) in zip(axes, losses_by_expert.items()):
        smooth_losses = moving_average(losses, window=25)
        axis.plot(losses, color="#afc4cb", alpha=0.28, linewidth=1.0, label="Raw loss")
        axis.plot(smooth_losses, color="#184652", linewidth=2.6, label="25-step moving average")
        axis.set_title(f"Expert {expert.upper()} Loss")
        axis.set_xlabel("Step")
        axis.set_ylabel("MSE Loss")
        axis.grid(alpha=0.18)
        axis.legend(frameon=False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved loss plot to {out_path}", flush=True)


def main() -> None:
    args = parse_args()
    tokenizer, text_encoder, vae, scheduler = load_frozen_components()
    losses_by_expert: dict[str, list[float]] = {}

    for expert in args.experts:
        losses_by_expert[expert] = train_one_expert(
            expert=expert,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            scheduler=scheduler,
            steps=args.steps,
            batch_size=args.batch_size,
            gradient_accumulation=args.gradient_accumulation,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            teacher_root=args.teacher_root,
            save_root=args.save_root,
            limit=args.limit,
            checkpoint_interval=args.checkpoint_interval,
            rank=args.rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            use_style_cross_attention=not args.disable_style_cross_attention,
            style_fusion_hidden_dim=args.style_fusion_hidden_dim,
            style_fusion_heads=args.style_fusion_heads,
            style_source_grid_size=args.style_source_grid_size,
            style_fusion_mlp_ratio=args.style_fusion_mlp_ratio,
            fresh=args.fresh,
        )

    if not args.disable_plot:
        args.loss_output.parent.mkdir(parents=True, exist_ok=True)
        save_loss_plot(losses_by_expert, args.loss_output)


if __name__ == "__main__":
    main()
