from __future__ import annotations

import argparse
import json
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


PROJECT_ROOT = Path(__file__).resolve().parent
ASSETS_ROOT = PROJECT_ROOT / "assets"
MODEL_ID = "timbrooks/instruct-pix2pix"
DEFAULT_TEACHER_ROOT = PROJECT_ROOT / "teacher_kontext_v2"
DEFAULT_SAVE_ROOT = PROJECT_ROOT / "student_ip2p_v2"
STYLE_FILES = {
    "c": PROJECT_ROOT / "style_c.txt",
    "d": PROJECT_ROOT / "style_d.txt",
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


def build_student_prompt(style: str) -> str:
    return (
        "Apply this photographic style while preserving the exact scene, subject, composition, and objects: "
        f"{style}"
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
    parser = argparse.ArgumentParser(description="Train edit-conditioned student LoRAs with InstructPix2Pix.")
    parser.add_argument("--experts", nargs="+", choices=["c", "d"], default=["c", "d"])
    parser.add_argument("--teacher-root", type=Path, default=DEFAULT_TEACHER_ROOT)
    parser.add_argument("--save-root", type=Path, default=DEFAULT_SAVE_ROOT)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--checkpoint-interval", type=int, default=25)
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


def build_lora_config() -> LoraConfig:
    return LoraConfig(
        r=4,
        lora_alpha=32,
        target_modules=["to_q", "to_v"],
        lora_dropout=0.1,
        bias="none",
    )


def load_trainable_unet(checkpoint_dir: Path | None = None) -> UNet2DConditionModel:
    unet = UNet2DConditionModel.from_pretrained(MODEL_ID, subfolder="unet").to(DEVICE)
    if checkpoint_dir is not None:
        unet = PeftModel.from_pretrained(unet, checkpoint_dir, is_trainable=True)
        print(f"Loaded checkpoint from {checkpoint_dir}", flush=True)
        unet.print_trainable_parameters()
        return unet

    lora_config = build_lora_config()
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()
    return unet


def encode_prompt(tokenizer: CLIPTokenizer, text_encoder: CLIPTextModel, prompt: str) -> torch.Tensor:
    with torch.no_grad():
        tokens = tokenizer(
            [prompt],
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).to(DEVICE)
        return text_encoder(**tokens).last_hidden_state


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
    style: str,
    teacher_dir: Path,
    dataset_size: int,
    steps: int,
    learning_rate: float,
) -> None:
    metadata = {
        "expert": expert,
        "style": style,
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
    optimizer: torch.optim.Optimizer,
    out_dir: Path,
    expert: str,
    completed_steps: int,
    total_steps: int,
    losses: list[float],
) -> None:
    checkpoints_dir = out_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = checkpoints_dir / f"step_{completed_steps:04d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    unet.save_pretrained(checkpoint_dir)

    state = {
        "expert": expert,
        "completed_steps": completed_steps,
        "total_steps": total_steps,
        "losses": losses,
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
    fresh: bool,
) -> list[float]:
    style = STYLE_FILES[expert].read_text(encoding="utf-8").strip()
    prompt = build_student_prompt(style)
    prompt_embeds = encode_prompt(tokenizer, text_encoder, prompt)

    raw_dir = PROJECT_ROOT / "data" / f"expert_{expert}" / "raw"
    teacher_dir = teacher_root / f"expert_{expert}"
    dataset = TeacherPairDataset(raw_dir, teacher_dir, limit=limit)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    out_dir = save_root / f"expert_{expert}"
    out_dir.mkdir(parents=True, exist_ok=True)
    resume_state = None if fresh else load_resume_state(out_dir)

    print(f"\n=== Training student for expert {expert.upper()} ===", flush=True)
    print(f"Matched teacher pairs: {len(dataset)}", flush=True)
    print(f"Prompt: {prompt}", flush=True)

    checkpoint_dir = None
    completed_steps = 0
    losses: list[float] = []
    if resume_state is not None:
        completed_steps = int(resume_state.get("completed_steps", 0))
        checkpoint_dir_value = resume_state.get("checkpoint_dir")
        checkpoint_dir = Path(checkpoint_dir_value) if checkpoint_dir_value else None
        losses = list(resume_state.get("losses", []))
        if completed_steps >= steps and (out_dir / "adapter_model.safetensors").exists():
            print(f"Expert {expert.upper()} already finished at step {completed_steps}, skipping.", flush=True)
            return losses

    unet = load_trainable_unet(checkpoint_dir=checkpoint_dir)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in unet.parameters() if parameter.requires_grad],
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    if resume_state is not None and "optimizer_state_dict" in resume_state:
        optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        print(f"Resuming expert {expert.upper()} from step {completed_steps}", flush=True)

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
                optimizer=optimizer,
                out_dir=out_dir,
                expert=expert,
                completed_steps=step + 1,
                total_steps=steps,
                losses=losses,
            )

    unet.save_pretrained(out_dir)
    save_checkpoint(
        unet=unet,
        optimizer=optimizer,
        out_dir=out_dir,
        expert=expert,
        completed_steps=steps,
        total_steps=steps,
        losses=losses,
    )
    save_metadata(
        out_dir,
        expert=expert,
        style=style,
        teacher_dir=teacher_dir,
        dataset_size=len(dataset),
        steps=steps,
        learning_rate=learning_rate,
    )
    print(f"Saved student LoRA to {out_dir}", flush=True)

    unet.to("cpu")
    del unet
    if DEVICE.type == "mps":
        torch.mps.empty_cache()
    return losses


def save_loss_plot(losses_by_expert: dict[str, list[float]], out_path: Path) -> None:
    fig, axes = plt.subplots(1, len(losses_by_expert), figsize=(6 * len(losses_by_expert), 4))
    if len(losses_by_expert) == 1:
        axes = [axes]

    for axis, (expert, losses) in zip(axes, losses_by_expert.items()):
        axis.plot(losses)
        axis.set_title(f"Expert {expert.upper()} Loss")
        axis.set_xlabel("Step")
        axis.set_ylabel("MSE Loss")

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
            fresh=args.fresh,
        )

    if not args.disable_plot:
        ASSETS_ROOT.mkdir(parents=True, exist_ok=True)
        save_loss_plot(losses_by_expert, ASSETS_ROOT / "loss_curves_v2.png")


if __name__ == "__main__":
    main()
