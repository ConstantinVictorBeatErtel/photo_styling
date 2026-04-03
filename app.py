from __future__ import annotations

import gc
import time
from pathlib import Path

import streamlit as st
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
SAMPLE_ASSETS = ASSETS / "samples"
DATA_ROOT = ROOT / "data"
STYLE_C = (ROOT / "style_c.txt").read_text(encoding="utf-8").strip()
STYLE_D = (ROOT / "style_d.txt").read_text(encoding="utf-8").strip()

PIPELINE_DIAGRAM = ASSETS / "pipeline_overview_v2.svg"
TRAIN_GRID = ASSETS / "demo_grid_student_v2_r8.png"
STUDENT_ROOT = ROOT / "student_ip2p_v2_r8"
PRECOMPUTED_ROOT = ASSETS / "precomputed"


def image_exists(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def show_image(path: Path, caption: str | None = None) -> None:
    if image_exists(path):
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.warning(f"Missing asset: {path.name}")


def sample_images() -> list[Path]:
    asset_files = [
        path
        for path in sorted(SAMPLE_ASSETS.glob("*.jpg"))
        if (PRECOMPUTED_ROOT / f"{path.stem}_baseline.jpg").exists()
        and (PRECOMPUTED_ROOT / f"{path.stem}_c.jpg").exists()
        and (PRECOMPUTED_ROOT / f"{path.stem}_d.jpg").exists()
    ]
    if asset_files:
        return asset_files
    sample_dir = DATA_ROOT / "expert_c" / "raw"
    if not sample_dir.exists():
        return []
    files = sorted(sample_dir.glob("*.jpg"))
    return files[200:220] if len(files) >= 220 else files[: min(8, len(files))]


def adapter_size_mb(expert: str) -> float | None:
    adapter_path = STUDENT_ROOT / f"expert_{expert}" / "adapter_model.safetensors"
    if not adapter_path.exists():
        return None
    return adapter_path.stat().st_size / (1024 * 1024)


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
        return "Local student adapters are not available in this deployment."
    if not (STUDENT_ROOT / "expert_d" / "adapter_model.safetensors").exists():
        return "Local student adapters are not available in this deployment."
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


def precomputed_result_path(sample_name: str, kind: str) -> Path:
    suffix_map = {"baseline": "baseline", "c": "c", "d": "d"}
    return PRECOMPUTED_ROOT / f"{Path(sample_name).stem}_{suffix_map[kind]}.jpg"


def precomputed_available(sample_name: str, selected_outputs: list[str]) -> bool:
    kind_map = {"Baseline": "baseline", "Bright neutral finish": "c", "Cool muted finish": "d"}
    return all(precomputed_result_path(sample_name, kind_map[label]).exists() for label in selected_outputs)


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
        label = "Generic edit"
    elif kind == "c":
        prompt = build_student_prompt(STYLE_C)
        pipe, device = load_pipe("student", expert="c")
        label = "Bright neutral adapter"
    else:
        prompt = build_student_prompt(STYLE_D)
        pipe, device = load_pipe("student", expert="d")
        label = "Cool muted adapter"

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
    [data-testid="stFileUploaderDropzone"] button,
    [data-testid="stBaseButton-secondary"] {
        background: #d9ecef !important;
        color: #111111 !important;
        border: 1px solid rgba(20, 62, 74, 0.18) !important;
    }
    [data-testid="stFileUploaderDropzone"] button:hover,
    [data-testid="stBaseButton-secondary"]:hover {
        background: #c6e2e7 !important;
        color: #111111 !important;
    }
    [data-testid="stStatusWidget"] {
        background: rgba(255, 255, 255, 0.88) !important;
    }
    .stButton button {
        background: #d9ecef !important;
        color: #111111 !important;
        border: 1px solid rgba(20, 62, 74, 0.18) !important;
    }
    .stButton button:hover {
        background: #c6e2e7 !important;
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
        <div class="eyebrow">Listing Photo Editing Demo</div>
        <h1>One compact adapter per photographer</h1>
        <p>
            This prototype targets a real product constraint in listing photography: different photographers want different
            finishes, but serving a heavyweight model per photographer is too expensive. The system learns each
            photographer's finish from past edits, distills it into a compact LoRA, and applies that finish to new images
            while preserving scene content.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

profile_c_size = adapter_size_mb("c")
adapter_size_text = f"{profile_c_size:.2f} MB" if profile_c_size is not None else "a few megabytes"

st.markdown("### What this demo proves")
product_cols = st.columns(4)
product_cards = [
    (
        "Editing target",
        "Preserve room geometry, framing, and objects while matching the photographer's finish.",
        ["Scene preservation", "Photographer-specific"],
    ),
    (
        "Teacher",
        "FLUX.1 Kontext [pro] supplies high-quality, content-preserving target edits for each profile.",
        ["Teacher model", "High-quality targets"],
    ),
    (
        "Student",
        "InstructPix2Pix LoRA is the deployable unit. The current best run uses rank 8 and 2000 training steps.",
        ["Source-conditioned", "Compact adapter"],
    ),
    (
        "Deployment fit",
        f"Each tracked student adapter is about {adapter_size_text}. That is much easier to swap per photographer than a hosted teacher.",
        ["~3 MB per profile", "Cheaper to serve"],
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

st.markdown("### Strongest evidence")
st.write(
    "This is the strongest tracked qualitative result in the repository. Success here means two things at once: "
    "the scene stays intact, and the two profile-specific students separate into meaningfully different finishes."
)
show_image(
    TRAIN_GRID,
    "Input | Generic edit | Bright-neutral teacher | Bright-neutral student | Cool-muted teacher | Cool-muted student",
)

style_cols = st.columns(2)
with style_cols[0]:
    st.markdown(
        f"""
        <div class="style-note">
            <div class="mini-label">Bright neutral finish</div>
            <h4>Photographer-specific bright neutral adapter</h4>
            <p>{STYLE_C}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with style_cols[1]:
    st.markdown(
        f"""
        <div class="style-note">
            <div class="mini-label">Cool muted finish</div>
            <h4>Photographer-specific cool muted adapter</h4>
            <p>{STYLE_D}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("### Why this matters for listing workflows")
listing_cols = st.columns(3)
listing_cards = [
    (
        "Consistency across listings",
        "A generic enhancement model smooths everything toward the same look. A per-photographer adapter keeps one shoot consistent with that photographer's existing portfolio.",
    ),
    (
        "Scene preservation",
        "Listing edits should change tone and finish, not rearrange layout cues. The teacher-student setup is explicitly designed around preserving geometry and content.",
    ),
    (
        "Deployment efficiency",
        "Serving a compact student per photographer is more operationally realistic than routing every request through a heavyweight teacher model.",
    ),
]
for col, (title, text) in zip(listing_cols, listing_cards):
    with col:
        st.markdown(
            f"""
            <div class="glass-card">
                <h3>{title}</h3>
                <p>{text}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("### How the system is built")
explain_cols = st.columns([1.3, 1.0])
with explain_cols[0]:
    show_image(PIPELINE_DIAGRAM, "Portfolio history -> style discovery -> teacher edits -> per-photographer student adapter")
with explain_cols[1]:
    st.markdown(
        """
        <div class="section-copy">
        BLIP-2 captions each photographer's historical edits. DeepSeek compresses those captions into a one-sentence
        finish description. FLUX Kontext then generates teacher edits that preserve scene content while matching that
        finish. Finally, a smaller InstructPix2Pix LoRA is distilled on those raw-to-teacher pairs.
        <br><br>
        This is the important v2 correction over a style-only LoRA: the student is trained as a source-conditioned edit
        model, not just a stylistic image generator.
        </div>
        """,
        unsafe_allow_html=True,
    )
st.markdown("### Try the student model")
st.write(
    "Use the prepared sample or upload your own image. The generic edit is the non-personalized baseline from the base model. "
    "The bright-neutral and cool-muted adapters are separate photographer-specific students applied directly to the same source image, so the comparison is about finish control, not a multi-stage edit stack."
)
runtime_issue = runtime_error()
studio_left, studio_right = st.columns([1.05, 0.95])

sample_paths = sample_images()
sample_map = {path.name: path for path in sample_paths}
selected_sample = None
precomputed_ready = False

with studio_left:
    source_mode = st.radio("Image source", ["Use sample", "Upload image"], horizontal=True)
    uploaded_file = None

    if source_mode == "Upload image":
        uploaded_file = st.file_uploader("Upload a source photo", type=["jpg", "jpeg", "png"])
    else:
        if sample_map:
            if len(sample_map) == 1:
                selected_sample = next(iter(sample_map))
                st.caption("Prepared demo sample selected.")
            else:
                selected_sample = st.selectbox("Prepared demo sample", list(sample_map.keys()))
        else:
            st.info("No bundled sample images are available.")

    selected_outputs = st.multiselect(
        "Outputs to render",
        ["Baseline", "Bright neutral finish", "Cool muted finish"],
        default=["Baseline", "Bright neutral finish", "Cool muted finish"],
    )
    steps = st.slider("Inference steps", min_value=20, max_value=60, value=50, step=5)
    guidance_scale = st.slider("Text guidance", min_value=1.0, max_value=6.0, value=3.0, step=0.5)
    image_guidance_scale = st.slider("Image guidance", min_value=0.5, max_value=2.0, value=1.0, step=0.25)
    precomputed_ready = bool(selected_sample and precomputed_available(selected_sample, selected_outputs))

    run_clicked = st.button(
        "Generate personalized edit",
        type="primary",
        disabled=bool((runtime_issue and not precomputed_ready) or not selected_outputs),
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
        st.image(preview_image, caption=f"{source_caption} resized to 512x512 for inference", use_container_width=True)
    else:
        st.info("Choose the prepared sample or upload an image to run the profile adapters.")

    if runtime_issue:
        if precomputed_ready:
            st.caption("This deployment uses bundled outputs for the prepared sample. Full local inference is available in the complete ML setup.")
        else:
            st.caption("Live generation is only available in the full local setup.")

if run_clicked and source_image is not None and runtime_issue and source_mode == "Use sample" and selected_sample and precomputed_ready:
    prepared = prepare_input_image(source_image)
    results = [("Input image", prepared, 0.0)]
    if "Baseline" in selected_outputs:
        results.append(("Generic edit", Image.open(precomputed_result_path(selected_sample, "baseline")).convert("RGB"), 0.0))
    if "Bright neutral finish" in selected_outputs:
        results.append(("Bright neutral adapter", Image.open(precomputed_result_path(selected_sample, "c")).convert("RGB"), 0.0))
    if "Cool muted finish" in selected_outputs:
        results.append(("Cool muted adapter", Image.open(precomputed_result_path(selected_sample, "d")).convert("RGB"), 0.0))
    st.session_state["latest_results"] = results

if run_clicked and source_image is not None and not runtime_issue:
    prepared = prepare_input_image(source_image)
    results = [("Input image", prepared, 0.0)]
    order = []
    if "Baseline" in selected_outputs:
        order.append("baseline")
    if "Bright neutral finish" in selected_outputs:
        order.append("c")
    if "Cool muted finish" in selected_outputs:
        order.append("d")

    status = st.status("Running student inference...", expanded=True)
    for item in order:
        with status:
            label = {"baseline": "Generic edit", "c": "Bright neutral adapter", "d": "Cool muted adapter"}[item]
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
    st.caption("The newest generated comparison appears here so the result is easy to review after each run.")
    result_cols = st.columns(len(st.session_state["latest_results"]))
    for col, (label, image, elapsed) in zip(result_cols, st.session_state["latest_results"]):
        with col:
            st.image(image, caption=label, use_container_width=True)
            if label == "Bright neutral adapter":
                st.caption(STYLE_C)
            elif label == "Cool muted adapter":
                st.caption(STYLE_D)
            elif label != "Input image":
                st.caption(f"{elapsed:.1f}s")
