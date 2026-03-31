import os, time
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from diffusers import UNet2DConditionModel, DDPMScheduler, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model

device = torch.device("mps")
MODEL_ID = "runwayml/stable-diffusion-v1-5"

# Load VAE — needed to encode RGB images into 4-channel latents
print("Loading VAE...")
vae = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae").to(device)
vae.eval()
for p in vae.parameters():
    p.requires_grad = False  # VAE is frozen — we only train the LoRA

# Load tokenizer and text encoder for null embedding
print("Loading text encoder...")
tokenizer = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(MODEL_ID, subfolder="text_encoder").to(device)

with torch.no_grad():
    null_tokens = tokenizer(
        [""], padding="max_length", max_length=77,
        truncation=True, return_tensors="pt"
    ).to(device)
    null_embedding = text_encoder(**null_tokens).last_hidden_state  # [1, 77, 768]

text_encoder.to("cpu")
torch.mps.empty_cache()
print("Null embedding computed, text encoder offloaded")

# Load UNet and attach LoRA
print("Loading UNet...")
unet = UNet2DConditionModel.from_pretrained(MODEL_ID, subfolder="unet").to(device)

lora_config = LoraConfig(
    r=4,
    lora_alpha=32,
    target_modules=["to_q", "to_v"],
    lora_dropout=0.1,
    bias="none",
)
unet = get_peft_model(unet, lora_config)
unet.print_trainable_parameters()


class StyleDataset(Dataset):
    def __init__(self, edited_dir):
        # Training on edited images only — LoRA learns the style distribution
        self.files = sorted(os.listdir(edited_dir))
        self.edited_dir = edited_dir
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        fname = self.files[i]
        edited = self.transform(
            Image.open(os.path.join(self.edited_dir, fname)).convert("RGB")
        )
        return edited


def encode_to_latents(images):
    """Encode a batch of RGB images [-1,1] to VAE latents."""
    with torch.no_grad():
        latents = vae.encode(images).latent_dist.sample()
        latents = latents * vae.config.scaling_factor
    return latents


def train_lora(edited_dir, save_path, expert_name, steps=500):
    dataset = StyleDataset(edited_dir)
    print(f"Dataset: {len(dataset)} edited images")
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    scheduler = DDPMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")
    optimizer = torch.optim.AdamW(
        [p for p in unet.parameters() if p.requires_grad],
        lr=1e-4, weight_decay=1e-2
    )

    encoder_hidden = null_embedding.expand(1, -1, -1)  # [1, 77, 768]

    unet.train()
    losses = []
    data_iter = iter(loader)
    start = time.time()

    for step in range(steps):
        try:
            edited = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            edited = next(data_iter)

        edited = edited.to(device)

        # Encode edited image to latent space via VAE — gives [1, 4, 64, 64]
        latents = encode_to_latents(edited)

        # Sample noise and timestep
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0, scheduler.config.num_train_timesteps, (1,), device=device
        ).long()

        # Add noise to latents — this is what the UNet learns to reverse
        noisy_latents = scheduler.add_noise(latents, noise, timesteps)

        # Predict noise — UNet now receives correct 4-channel latents
        noise_pred = unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden,
        ).sample

        loss = torch.nn.functional.mse_loss(noise_pred, noise)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss.item())

        if (step + 1) % 50 == 0:
            avg = sum(losses[-50:]) / 50
            elapsed = time.time() - start
            remaining = (steps - step - 1) / (step + 1) * elapsed / 60
            print(f"  [{expert_name}] Step {step+1}/{steps} | "
                  f"Loss: {avg:.4f} | ~{remaining:.0f} min left")

    os.makedirs(save_path, exist_ok=True)
    unet.save_pretrained(save_path)
    print(f"LoRA saved to {save_path}")
    return losses


if __name__ == "__main__":
    print("=== Training Expert C LoRA ===")
    losses_c = train_lora(
        "data/expert_c/edited",
        "lora_expert_c",
        "Expert C"
    )

    # Reload fresh UNet for Expert D — mandatory
    print("\nReloading UNet for Expert D...")
    unet = UNet2DConditionModel.from_pretrained(MODEL_ID, subfolder="unet").to(device)
    unet = get_peft_model(unet, lora_config)
    torch.mps.empty_cache()

    print("=== Training Expert D LoRA ===")
    losses_d = train_lora(
        "data/expert_d/edited",
        "lora_expert_d",
        "Expert D"
    )

    # Save loss curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(losses_c)
    ax1.set_title("Expert C Loss")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("MSE Loss")
    ax2.plot(losses_d)
    ax2.set_title("Expert D Loss")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("MSE Loss")
    plt.tight_layout()
    plt.savefig("loss_curves.png", dpi=120)
    print("Loss curves saved to loss_curves.png")
