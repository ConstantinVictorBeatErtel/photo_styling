from __future__ import annotations

import base64
import io
import os
from pathlib import Path

import requests
import streamlit as st
from PIL import Image


ROOT = Path(__file__).resolve().parent
DEMO_GRID = ROOT / "demo_grid_flux.png"
PROBE = ROOT / "demo_flux_probe.png"
STYLE_C = (ROOT / "style_c.txt").read_text(encoding="utf-8").strip()
STYLE_D = (ROOT / "style_d.txt").read_text(encoding="utf-8").strip()
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_ID = "black-forest-labs/flux.2-pro"


def get_api_key() -> str | None:
    secret_value = None
    if "OPENROUTER_API_KEY" in st.secrets:
        secret_value = st.secrets["OPENROUTER_API_KEY"]
    return secret_value or os.getenv("OPENROUTER_API_KEY")


def image_bytes_to_data_url(image_bytes: bytes) -> str:
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def decode_image_data_url(data_url: str) -> Image.Image:
    _, encoded = data_url.split(",", 1)
    image_bytes = base64.b64decode(encoded)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def make_prompt(style: str) -> str:
    return (
        "Edit this image while preserving the exact subject, pose, scene, composition, and background. "
        "Do not add, remove, or replace objects or scenery. "
        "Apply only photographic color grading and tonal adjustments to match this style: "
        f"{style}"
    )


def render_style_preview(api_key: str, image_bytes: bytes, style: str) -> Image.Image:
    payload = {
        "model": MODEL_ID,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": make_prompt(style)},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_bytes_to_data_url(image_bytes),
                        },
                    },
                ],
            }
        ],
        "modalities": ["image"],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://photo-styling-demo.local",
        "X-Title": "Photo Styling Demo",
    }
    response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=(30, 240))
    response.raise_for_status()
    result = response.json()
    images = result["choices"][0]["message"]["images"]
    return decode_image_data_url(images[0]["image_url"]["url"])


st.set_page_config(page_title="Photo Styling", layout="wide")

st.title("Photo Styling")
st.caption("A small demo of personalized photo aesthetics learned from historical edits.")

metric_col1, metric_col2, metric_col3 = st.columns(3)
metric_col1.metric("Styles", "2")
metric_col2.metric("Training Images per Style", "200")
metric_col3.metric("Demo Model", "FLUX.2 Pro")

st.subheader("Style Signatures")
style_col1, style_col2 = st.columns(2)
with style_col1:
    st.markdown("**Candidate C**")
    st.info(STYLE_C)
with style_col2:
    st.markdown("**Candidate D**")
    st.info(STYLE_D)

st.subheader("Comparison Grid")
st.image(str(DEMO_GRID), use_container_width=True)

with st.expander("Show Single-Image Comparison"):
    st.image(str(PROBE), use_container_width=True)

st.subheader("How It Works")
st.markdown(
    """
    1. Past edited photos are captioned and summarized into concise visual style descriptions.
    2. Candidate-specific editing behavior is compared on the same held-out inputs.
    3. The result is a side-by-side look at how the same image changes under different aesthetic directions.
    """
)

st.subheader("Try Your Own Image")
st.write(
    "Upload a photo to preview how it might look in Candidate C and Candidate D style. "
    "This uses FLUX.2 Pro through OpenRouter when an API key is configured."
)

uploaded_file = st.file_uploader(
    "Upload a JPG or PNG image",
    type=["jpg", "jpeg", "png"],
)

api_key = get_api_key()
if uploaded_file and not api_key:
    st.warning("Add `OPENROUTER_API_KEY` to Streamlit secrets to enable uploaded-image previews.")

if uploaded_file and api_key:
    image_bytes = uploaded_file.getvalue()
    input_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    preview_col1, preview_col2, preview_col3 = st.columns(3)
    with preview_col1:
        st.markdown("**Original**")
        st.image(input_image, use_container_width=True)

    if st.button("Generate Candidate Previews", type="primary"):
        with st.spinner("Generating styled previews..."):
            try:
                candidate_c_image = render_style_preview(api_key, image_bytes, STYLE_C)
                candidate_d_image = render_style_preview(api_key, image_bytes, STYLE_D)
            except Exception as exc:
                st.error(f"Preview generation failed: {exc}")
            else:
                with preview_col2:
                    st.markdown("**Candidate C**")
                    st.image(candidate_c_image, use_container_width=True)
                with preview_col3:
                    st.markdown("**Candidate D**")
                    st.image(candidate_d_image, use_container_width=True)
