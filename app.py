from __future__ import annotations

from pathlib import Path

import streamlit as st


ROOT = Path(__file__).resolve().parent
DEMO_GRID = ROOT / "demo_grid_flux.png"
LOSS_CURVES = ROOT / "loss_curves.png"
PROBE = ROOT / "demo_flux_probe.png"
STYLE_C = (ROOT / "style_c.txt").read_text(encoding="utf-8").strip()
STYLE_D = (ROOT / "style_d.txt").read_text(encoding="utf-8").strip()
TALKING_POINTS = (ROOT / "talking_points.txt").read_text(encoding="utf-8").strip().splitlines()


st.set_page_config(
    page_title="Photo Styling Demo",
    page_icon="",
    layout="wide",
)

st.title("Photo Styling Personalized Style Distillation")
st.caption("Photographer-specific editing styles discovered from FiveK expert edits and showcased with FLUX.2 Pro.")

col1, col2, col3 = st.columns(3)
col1.metric("Photographers Modeled", "2")
col2.metric("Training Images per Expert", "200")
col3.metric("Approved Demo Path", "FLUX.2 Pro")

st.subheader("What This Project Does")
st.write(
    "Photo Styling learns editing style from past expert-retouched images, summarizes each photographer's look into a reusable style description, "
    "and compares how the same raw image changes under different photographer-specific aesthetics."
)

st.subheader("Learned Photographer Styles")
style_col1, style_col2 = st.columns(2)
with style_col1:
    st.markdown("**Expert C**")
    st.info(STYLE_C)
with style_col2:
    st.markdown("**Expert D**")
    st.info(STYLE_D)

st.subheader("Approved Demo Grid")
st.image(str(DEMO_GRID), use_container_width=True)

with st.expander("Show One-Image FLUX Probe"):
    st.image(str(PROBE), use_container_width=True)

with st.expander("Show Training Curves"):
    st.image(str(LOSS_CURVES), use_container_width=True)

st.subheader("Pipeline Summary")
st.markdown(
    """
    1. **Style Discovery**: BLIP-2 captions real expert edits and DeepSeek summarizes each expert into one style sentence.
    2. **Student Training**: Photographer-specific LoRA adapters are trained on real FiveK expert edits in latent space.
    3. **Demo Generation**: FLUX.2 Pro applies baseline, Expert C, and Expert D editing behavior to the same held-out raw images.
    """
)

st.subheader("CTO Talking Points")
for line in TALKING_POINTS:
    if line.strip():
        st.write(f"- {line.strip()}")

st.subheader("Run Locally")
st.code("streamlit run app.py", language="bash")
