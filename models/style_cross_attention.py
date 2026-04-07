from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


STYLE_CONDITIONER_FILENAME = "style_conditioner.pt"


class StyleCrossAttention(nn.Module):
    """A compact cross-attention block with explicit Q/K/V math."""

    def __init__(
        self,
        *,
        source_dim: int,
        style_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})")

        self.source_dim = source_dim
        self.style_dim = style_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)

        self.source_proj = nn.Linear(source_dim, hidden_dim)
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(style_dim, hidden_dim)
        self.value_proj = nn.Linear(style_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.attn_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, token_count, _ = tensor.shape
        tensor = tensor.view(batch_size, token_count, self.num_heads, self.head_dim)
        return tensor.transpose(1, 2)

    def _merge_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, _, token_count, _ = tensor.shape
        tensor = tensor.transpose(1, 2).contiguous()
        return tensor.view(batch_size, token_count, self.hidden_dim)

    def forward(
        self,
        source_feats: torch.Tensor,
        style_feats: torch.Tensor,
        style_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fuse source queries with style keys/values.

        Args:
            source_feats: `[B, N_src, D_src]`
            style_feats: `[B, N_style, D_style]`
            style_mask: optional `[B, N_style]` tensor with `1` for valid style tokens.
        """

        source_hidden = self.source_proj(source_feats)
        queries = self._split_heads(self.query_proj(source_hidden))
        keys = self._split_heads(self.key_proj(style_feats))
        values = self._split_heads(self.value_proj(style_feats))

        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) * self.scale
        if style_mask is not None:
            mask = style_mask[:, None, None, :].to(dtype=torch.bool, device=attention_scores.device)
            attention_scores = attention_scores.masked_fill(~mask, torch.finfo(attention_scores.dtype).min)

        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = self.attn_dropout(attention_weights)

        fused = torch.matmul(attention_weights, values)
        fused = self._merge_heads(fused)
        fused = self.output_dropout(self.out_proj(fused))

        hidden = self.norm1(source_hidden + fused)
        hidden = self.norm2(hidden + self.mlp(hidden))
        return hidden, attention_weights


class StyleConditioningAdapter(nn.Module):
    """Create one extra style-aware conditioning token from source latents and style tokens."""

    def __init__(
        self,
        *,
        source_dim: int = 4,
        style_dim: int = 768,
        hidden_dim: int = 256,
        num_heads: int = 4,
        source_grid_size: int = 16,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.source_dim = source_dim
        self.style_dim = style_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.source_grid_size = source_grid_size
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout

        self.cross_attention = StyleCrossAttention(
            source_dim=source_dim,
            style_dim=style_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
        self.output_proj = nn.Linear(hidden_dim, style_dim)
        self.output_norm = nn.LayerNorm(style_dim)

    def get_config(self) -> dict[str, Any]:
        return {
            "source_dim": self.source_dim,
            "style_dim": self.style_dim,
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "source_grid_size": self.source_grid_size,
            "mlp_ratio": self.mlp_ratio,
            "dropout": self.dropout,
        }

    def source_latents_to_tokens(self, source_latents: torch.Tensor) -> torch.Tensor:
        """Pool the VAE latent map down before attention to keep the module lightweight."""

        pooled = F.adaptive_avg_pool2d(source_latents, (self.source_grid_size, self.source_grid_size))
        return pooled.permute(0, 2, 3, 1).reshape(source_latents.shape[0], -1, source_latents.shape[1])

    def forward(
        self,
        source_latents: torch.Tensor,
        style_tokens: torch.Tensor,
        style_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        source_tokens = self.source_latents_to_tokens(source_latents)
        fused_source_tokens, attention_weights = self.cross_attention(source_tokens, style_tokens, style_mask)

        pooled_source = fused_source_tokens.mean(dim=1, keepdim=True)
        pooled_style = style_tokens.mean(dim=1, keepdim=True)
        fused_token = self.output_norm(pooled_style + self.output_proj(pooled_source))
        augmented_style_tokens = torch.cat([style_tokens, fused_token], dim=1)
        return augmented_style_tokens, {"attention_weights": attention_weights, "fused_token": fused_token}


def style_conditioner_path(directory: Path) -> Path:
    return directory / STYLE_CONDITIONER_FILENAME


def save_style_conditioner(path: Path, module: StyleConditioningAdapter) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": module.get_config(), "state_dict": module.state_dict()}, path)


def load_style_conditioner(path: Path, *, map_location: str | torch.device = "cpu") -> StyleConditioningAdapter:
    payload = torch.load(path, map_location=map_location)
    module = StyleConditioningAdapter(**payload["config"])
    module.load_state_dict(payload["state_dict"])
    return module
