from __future__ import annotations

import gc
import time
from io import BytesIO
from pathlib import Path

import streamlit as st
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
DATA_ROOT = ROOT / "data"
STYLE_C = (ROOT / "style_c.txt").read_text(encoding="utf-8").strip()
STYLE_D = (ROOT / "style_d.txt").read_text(encoding="utf-8").strip()

PIPELINE_DIAGRAM = ASSETS / "pipeline_overview_v2.svg"
TRAIN_GRID = ASSETS / "demo_grid_student_v2_final_train.png"
TEACHER_C_PREVIEW = ASSETS / "teacher_preview_c_v2.png"
TEACHER_D_PREVIEW = ASSETS / "teacher_preview_d_v2.png"
STUDENT_ROOT = ROOT / "student_ip2p_v2"


def image_exists(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def show_image(path: Path, caption: str | None = None) -> None:
    if image_exists(path):
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.warning(f"Missing asset: {path.name}")


def count_files(path: Path, suffix: str) -> int:
    if not path.exists():
        return 0
    return len(list(path.glob(f"*.{suffix}")))


def sample_images() -> list[Path]:
    sample_dir = DATA_ROOT / "expert_c" / "raw"
    if not sample_dir.exists():
        return []
    files = sorted(sample_dir.glob("*.jpg"))
    return files[200:220] if len(files) >= 220 else files[: min(8, len(files))]


def build_student_prompt(style: str) -> str:
    return (
        "Apply this photographic style while preserving the exact scene, subject, composition, and objects: "
        f"{style}"
    )


def build_baseline_prompt() -> str:
    return (
        "Improve this photo with balanced exposure, natural color, clean white balance, and gentle contrast "
        "while preserving the original scene."
    )


def prepare_input_image(image: Image.Image) -> Image.Image:
    return ImageOps.fit(image.convert("RGB"), (512, 512), Image.LANCZOS)


def runtime_error() -> str | None:
    if not (STUDENT_ROOT / "expert_c" / "adapter_model.safetensors").exists():
        return "Student adapter for Candidate C is missing."
    if not (STUDENT_ROOT / "expert_d" / "adapter_model.safetensors").exists():
        return "Student adapter for Candidate D is missing."
    try:
        import torch  # noqa: F401
        import diffusers  # noqa: F401
        import peft  # noqa: F401
    except Exception as exc:  # pragma: no cover - streamlit runtime feedback
        return (
            "Local inference is unavailable because the ML stack is missing or incomplete. "
            f"Details: {exc}"
        )
    return None


@st.cache_data(show_spinner=False)
def load_sample_preview(path: str) -> bytes:
    image = Image.open(path).convert("RGB")
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=92)
    return buffer.getvalue()


def load_pipe(kind: str, *, expert: str | None = None):
    import torch
    from diffusers import StableDiffusionInstructPix2PixPipeline, UNet2DConditionModel
    from peft import PeftModel

    model_id = "timbrooks/instruct-pix2pix"
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    if kind == "baseline":
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            safety_checker=None,
            requires_safety_checker=False,
        )
    else:
        assert expert is not None
        base_unet = UNet2DConditionModel.from_pretrained(
            model_id,
            subfolder="unet",
            torch_dtype=torch.float32,
        )
        merged_unet = PeftModel.from_pretrained(
            base_unet,
            STUDENT_ROOT / f"expert_{expert}",
        ).merge_and_unload()
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            model_id,
            unet=merged_unet,
            torch_dtype=torch.float32,
            safety_checker=None,
            requires_safety_checker=False,
        )

    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    pipe.set_progress_bar_config(disable=True)
    return pipe, device


def unload_pipe(pipe, device: str) -> None:
    del pipe
    gc.collect()
    if device == "mps":
        import torch

        torch.mps.empty_cache()


def run_edit(kind: str, image: Image.Image, *, steps: int, guidance_scale: float, image_guidance_scale: float):
    if kind == "baseline":
        prompt = build_baseline_prompt()
        pipe, device = load_pipe("baseline")
        label = "Baseline"
    elif kind == "c":
        prompt = build_student_prompt(STYLE_C)
        pipe, device = load_pipe("student", expert="c")
        label = "Candidate C"
    else:
        prompt = build_student_prompt(STYLE_D)
        pipe, device = load_pipe("student", expert="d")
        label = "Candidate D"

    start = time.perf_counter()
    result = pipe(
        prompt=prompt,
        image=image,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        image_guidance_scale=image_guidance_scale,
    ).images[0]
    elapsed = time.perf_counter() - start
    unload_pipe(pipe, device)
    return label, result, elapsed


st.set_page_config(page_title="Personalized Photo Editing", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at 0% 0%, rgba(193, 240, 227, 0.55) 0%, transparent 28%),
            radial-gradient(circle at 100% 0%, rgba(183, 215, 255, 0.48) 0%, transparent 30%),
            linear-gradient(180deg, #fbfaf6 0%, #eef3f7 100%);
        color: #111111 !important;
    }
    .stApp, .stApp p, .stApp li, .stApp label, .stApp span, .stApp div, .stApp h1, .stApp h2, .stApp h3, .stApp h4 {
        color: #111111;
    }
    .hero {
        padding: 1.6rem 1.8rem;
        border-radius: 28px;
        background: rgba(255, 255, 255, 0.8);
        border: 1px solid rgba(14, 52, 62, 0.10);
        box-shadow: 0 18px 45px rgba(22, 38, 55, 0.10);
        margin-bottom: 1rem;
    }
    .eyebrow {
        text-transform: uppercase;
        letter-spacing: 0.18em;
        font-size: 0.76rem;
        color: #57737d;
        margin-bottom: 0.55rem;
        font-weight: 800;
    }
    .hero h1 {
        margin: 0;
        font-size: 3.15rem;
        line-height: 1.0;
        color: #111111;
    }
    .hero p {
        margin-top: 0.9rem;
        color: #111111;
        font-size: 1.07rem;
        max-width: 54rem;
    }
    .glass-card {
        padding: 1rem 1.05rem;
        border-radius: 20px;
        background: rgba(255, 255, 255, 0.78);
        border: 1px solid rgba(18, 51, 61, 0.08);
        box-shadow: 0 12px 30px rgba(25, 43, 57, 0.06);
        height: 100%;
    }
    .glass-card h3 {
        margin-top: 0;
        margin-bottom: 0.5rem;
        color: #111111;
    }
    .glass-card p {
        color: #111111;
        margin-bottom: 0;
        line-height: 1.55;
    }
    .pill {
        display: inline-block;
        padding: 0.33rem 0.62rem;
        border-radius: 999px;
        background: #e3eff2;
        color: #20515d;
        font-size: 0.8rem;
        margin-right: 0.35rem;
        margin-bottom: 0.4rem;
        font-weight: 700;
    }
    .section-copy {
        color: #111111;
        font-size: 1rem;
        line-height: 1.65;
    }
    .style-note {
        padding: 1rem 1.05rem;
        border-radius: 22px;
        background: linear-gradient(180deg, rgba(255,255,255,0.86), rgba(244,249,250,0.92));
        border: 1px solid rgba(20, 62, 74, 0.10);
        box-shadow: 0 18px 34px rgba(25, 43, 57, 0.08);
    }
    .style-note h4 {
        margin: 0 0 0.5rem 0;
        color: #111111;
        font-size: 1.12rem;
    }
    .style-note p {
        margin: 0;
        color: #111111;
        line-height: 1.6;
    }
    .mini-label {
        text-transform: uppercase;
        letter-spacing: 0.14em;
        font-size: 0.72rem;
        color: #64808a;
        font-weight: 800;
        margin-bottom: 0.55rem;
    }
    .stRadio > label,
    .stSelectbox > label,
    .stMultiSelect > label,
    .stFileUploader > label,
    .stSlider > label {
        color: #111111 !important;
        font-weight: 700 !important;
    }
    [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] span,
    .stRadio p,
    .stMarkdown p,
    .stCaption {
        color: #111111 !important;
    }
    div[data-baseweb="select"] > div,
    div[data-baseweb="base-input"] > div,
    .stTextInput input,
    .stNumberInput input {
        background: rgba(255, 255, 255, 0.96) !important;
        color: #111111 !important;
        border-color: rgba(20, 62, 74, 0.12) !important;
    }
    div[data-baseweb="tag"] {
        background: #dfecef !important;
        color: #111111 !important;
    }
    [data-baseweb="select"] input {
        color: #111111 !important;
    }
    [data-testid="stFileUploaderDropzone"] {
        background: rgba(255, 255, 255, 0.9) !important;
        border: 1px dashed rgba(20, 62, 74, 0.25) !important;
    }
    [data-testid="stFileUploaderDropzone"] * {
        color: #111111 !important;
    }
    [data-testid="stStatusWidget"] {
        background: rgba(255, 255, 255, 0.88) !important;
    }
    .stButton button {
        color: #111111 !important;
    }
    .stMetric label, .stMetric div {
        color: #111111 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Personalized Photo Editing Demo</div>
        <h1>One model per photographer</h1>
        <p>
            This system learns a photographer's look from their edit history, distills that behavior into a compact LoRA,
            and then applies that personalized style to new raw photos at much lower inference cost than the original teacher.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption("A personalized photo-editing system that learns a photographer's signature look and deploys it as a compact student model.")

teacher_total = count_files(ROOT / "teacher_kontext_v2" / "expert_c", "jpg") + count_files(
    ROOT / "teacher_kontext_v2" / "expert_d", "jpg"
)
metric_cols = st.columns(4)
metric_cols[0].metric("Portfolio Pairs Used", "220")
metric_cols[1].metric("Teacher Outputs", str(teacher_total))
metric_cols[2].metric("Student Adapters", "2")
metric_cols[3].metric("Inference Target", "Photographer-specific")

st.markdown("### Product")
product_cols = st.columns(3)
product_cards = [
    (
        "Why it matters",
        "Real estate photographers do not want generic enhancement. They want an AI assistant that keeps composition intact while matching their own finish.",
        ["Natural language", "Personalized"],
    ),
    (
        "What the model does",
        "A stronger teacher model creates style-consistent edits, then a smaller InstructPix2Pix LoRA learns that mapping so deployment is cheaper and easier to scale.",
        ["Teacher-student", "LoRA"],
    ),
    (
        "What gets deployed",
        "Instead of serving a heavyweight teacher per photographer, the deployed system uses a compact style adapter that reproduces the learned look on new photos.",
        ["Lower cost", "Fast swap"],
    ),
]
for col, (title, text, pills) in zip(product_cols, product_cards):
    with col:
        pill_html = "".join(f'<span class="pill">{pill}</span>' for pill in pills)
        st.markdown(
            f"""
            <div class="glass-card">
                <h3>{title}</h3>
                <p>{pill_html}</p>
                <p>{text}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("### How personalization works")
explain_cols = st.columns([1.3, 1.0])
with explain_cols[0]:
    show_image(PIPELINE_DIAGRAM, "Pipeline overview: portfolio -> teacher/student distillation -> personalized deployment")
with explain_cols[1]:
    st.markdown(
        """
        <div class="section-copy">
        The product starts with a photographer's own portfolio. BLIP-2 captions past edits, DeepSeek compresses those
        captions into a style sentence, and FLUX Kontext creates teacher edits that preserve scene content while shifting
        tone and mood. A much smaller InstructPix2Pix LoRA is then distilled on those raw-to-teacher pairs.
        <br><br>
        The result is the deployment-friendly piece at the bottom of the diagram: one personalized LoRA per
        photographer that can style a new raw photo without re-running the expensive teacher for every request.
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("### Example comparison")
st.write(
    "This single comparison view shows all six versions head to head: raw input, generic baseline, both teacher targets, "
    "and both distilled student outputs."
)
show_image(TRAIN_GRID, "Raw | Baseline | Teacher C | Student C | Teacher D | Student D")

style_cols = st.columns(2)
with style_cols[0]:
    st.markdown(
        f"""
        <div class="style-note">
            <div class="mini-label">Candidate C</div>
            <h4>Bright, clean, premium real-estate finish</h4>
            <p>{STYLE_C}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with style_cols[1]:
    st.markdown(
        f"""
        <div class="style-note">
            <div class="mini-label">Candidate D</div>
            <h4>Moody, contrast-rich, cinematic finish</h4>
            <p>{STYLE_D}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("### Try the student model")
runtime_issue = runtime_error()
studio_left, studio_right = st.columns([1.05, 0.95])

sample_paths = sample_images()
sample_map = {path.name: path for path in sample_paths}

with studio_left:
    source_mode = st.radio("Image source", ["Use sample", "Upload image"], horizontal=True)
    uploaded_file = None
    selected_sample = None

    if source_mode == "Upload image":
        uploaded_file = st.file_uploader("Upload a raw-style input photo", type=["jpg", "jpeg", "png"])
    else:
        if sample_map:
            selected_sample = st.selectbox("Pick a prepared sample", list(sample_map.keys()))
        else:
            st.info("No local sample images were found in data/expert_c/raw.")

    selected_outputs = st.multiselect(
        "Outputs to render",
        ["Baseline", "Candidate C", "Candidate D"],
        default=["Candidate C", "Candidate D"],
    )
    steps = st.slider("Inference steps", min_value=12, max_value=30, value=18, step=2)
    guidance_scale = st.slider("Text guidance", min_value=5.0, max_value=9.0, value=7.5, step=0.5)
    image_guidance_scale = st.slider("Image guidance", min_value=1.0, max_value=2.5, value=2.0, step=0.25)

    if runtime_issue:
        st.warning(runtime_issue)

    run_clicked = st.button(
        "Generate personalized edit",
        type="primary",
        disabled=bool(runtime_issue or not selected_outputs),
    )

with studio_right:
    source_image = None
    source_caption = "Prepared sample"
    if uploaded_file is not None:
        source_image = Image.open(uploaded_file).convert("RGB")
        source_caption = "Uploaded image"
    elif selected_sample:
        source_image = Image.open(sample_map[selected_sample]).convert("RGB")

    if source_image is not None:
        preview_image = prepare_input_image(source_image)
        st.image(preview_image, caption=f"{source_caption} resized to 512x512 for student inference", use_container_width=True)
    else:
        st.info("Choose a sample or upload an image to preview the student model.")

if run_clicked and source_image is not None:
    prepared = prepare_input_image(source_image)
    results = [("Raw input", prepared, 0.0)]
    order = []
    if "Baseline" in selected_outputs:
        order.append("baseline")
    if "Candidate C" in selected_outputs:
        order.append("c")
    if "Candidate D" in selected_outputs:
        order.append("d")

    status = st.status("Running student inference...", expanded=True)
    for item in order:
        with status:
            label = {"baseline": "Baseline", "c": "Candidate C", "d": "Candidate D"}[item]
            st.write(f"Loading and generating {label}...")
        label, image, elapsed = run_edit(
            item,
            prepared,
            steps=steps,
            guidance_scale=guidance_scale,
            image_guidance_scale=image_guidance_scale,
        )
        results.append((label, image, elapsed))
    status.update(label="Inference complete", state="complete")
    st.session_state["latest_results"] = results

if "latest_results" in st.session_state:
    st.markdown("#### Latest inference")
    result_cols = st.columns(len(st.session_state["latest_results"]))
    for col, (label, image, elapsed) in zip(result_cols, st.session_state["latest_results"]):
        with col:
            st.image(image, caption=label, use_container_width=True)
            if label == "Candidate C":
                st.caption(STYLE_C)
            elif label == "Candidate D":
                st.caption(STYLE_D)
            elif label != "Raw input":
                st.caption(f"{elapsed:.1f}s")

st.markdown("### Model summary")
st.write(
    "The teacher is FLUX.1 Kontext [pro]. The deployed student is an InstructPix2Pix LoRA specialized to one "
    "photographer's style. That is the core product claim: each photographer gets a compact adapter that keeps "
    "their look consistent without needing a giant personalized model in production."
)
