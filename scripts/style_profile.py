from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


STYLE_PROFILE_VERSION = 1
STYLE_FACETS = (
    "exposure",
    "contrast",
    "white_balance",
    "saturation",
    "highlight_handling",
    "shadow_handling",
    "scene_tendency",
    "mood",
)
FACET_LABELS = {
    "exposure": "Exposure",
    "contrast": "Contrast",
    "white_balance": "White balance",
    "saturation": "Saturation",
    "highlight_handling": "Highlight handling",
    "shadow_handling": "Shadow handling",
    "scene_tendency": "Scene tendency",
    "mood": "Mood",
}
DEFAULT_STYLE_SUMMARY = "Balanced, natural real-estate editing finish."


def build_style_paths(project_root: Path, expert: str) -> dict[str, Path]:
    """Return the JSON and legacy txt paths for one photographer profile."""

    return {
        "json": project_root / f"style_{expert}.json",
        "txt": project_root / f"style_{expert}.txt",
    }


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split())


def _ensure_sentence(text: str) -> str:
    text = _clean_text(text)
    if not text:
        return ""
    if text.endswith((".", "!", "?")):
        return text
    return f"{text}."


def empty_style_facets() -> dict[str, str]:
    """Create an empty facet dictionary using the tracked schema."""

    return {facet: "" for facet in STYLE_FACETS}


def compose_style_prompt(profile: Mapping[str, Any]) -> str:
    """Compose a richer text prompt from the summary plus any populated facets."""

    summary = _clean_text(profile.get("style_summary", ""))
    facets = profile.get("facets", {})

    prompt_parts: list[str] = []
    if summary:
        prompt_parts.append(_ensure_sentence(summary))

    for facet in STYLE_FACETS:
        value = _clean_text(facets.get(facet, "")) if isinstance(facets, Mapping) else ""
        if value:
            prompt_parts.append(_ensure_sentence(f"{FACET_LABELS[facet]}: {value}"))

    return " ".join(prompt_parts) or _ensure_sentence(DEFAULT_STYLE_SUMMARY)


def normalize_style_profile(
    profile: Mapping[str, Any] | None,
    *,
    profile_name: str | None = None,
    fallback_style: str = "",
) -> dict[str, Any]:
    """Normalize either JSON or legacy text input into the shared profile schema."""

    raw_profile = dict(profile or {})
    raw_facets = raw_profile.get("facets", {})
    facets = empty_style_facets()
    if isinstance(raw_facets, Mapping):
        for facet in STYLE_FACETS:
            facets[facet] = _clean_text(raw_facets.get(facet, ""))

    resolved_name = _clean_text(raw_profile.get("profile_name", "")) or _clean_text(profile_name) or "unknown_profile"
    style_summary = (
        _clean_text(raw_profile.get("style_summary", ""))
        or _clean_text(fallback_style)
        or DEFAULT_STYLE_SUMMARY
    )

    normalized = {
        "schema_version": STYLE_PROFILE_VERSION,
        "profile_name": resolved_name,
        "style_summary": style_summary,
        "facets": facets,
    }
    normalized["composed_prompt"] = _clean_text(raw_profile.get("composed_prompt", "")) or compose_style_prompt(normalized)
    return normalized


def save_style_profile(path: Path, profile: Mapping[str, Any]) -> dict[str, Any]:
    """Save a normalized style profile JSON to disk."""

    normalized = normalize_style_profile(profile, profile_name=profile.get("profile_name"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(normalized, indent=2), encoding="utf-8")
    return normalized


def load_style_profile(
    json_path: Path,
    txt_path: Path | None = None,
    *,
    profile_name: str | None = None,
) -> dict[str, Any]:
    """Load the new JSON style profile, falling back to the legacy txt sentence."""

    if json_path.exists():
        raw_profile = json.loads(json_path.read_text(encoding="utf-8"))
        return normalize_style_profile(raw_profile, profile_name=profile_name)

    fallback_style = ""
    if txt_path is not None and txt_path.exists():
        fallback_style = txt_path.read_text(encoding="utf-8").strip()

    return normalize_style_profile(
        {"style_summary": fallback_style},
        profile_name=profile_name,
        fallback_style=fallback_style,
    )
