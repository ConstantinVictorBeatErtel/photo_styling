# Photo Styling Personalized Style Distillation

## Problem

Real estate photographers have distinct editing styles, but today's AI tools apply the same edit to everyone. There's no way to capture a photographer's unique aesthetic without manual preset-building.

## Approach

Three-stage pipeline:

- *Style Discovery:* BLIP-2 auto-captions a photographer's past edits. DeepSeek V3.2 summarizes the captions into one style sentence — no human input needed.
- *Student Training:* A lightweight LoRA adapter is fine-tuned on real FiveK expert edits so each adapter learns the visual distribution of one photographer's style from professional ground-truth examples.
- *Demo Generation:* For the final showcase grid, FLUX.2 Pro applies the discovered photographer styles to the same raw images, producing a clean side-by-side demonstration of personalized editing behavior.

## Results

![Photo Styling Demo Grid](demo_grid_flux.png)

The approved demo grid shows the same raw input edited three ways: a generic baseline, an Expert C style, and an Expert D style. Expert C produces brighter, crisper, more natural-looking outputs, while Expert D pushes the same images toward cooler, flatter, more muted tones. The columns remain visually tied to the same source photo, but the editing intent clearly changes by photographer.

## Scale Path

With real user edit history, each photographer's style profile can be inferred automatically from prior work, then paired with lightweight photographer-specific adapters or teacher-guided personalization workflows. No manual presets, no user configuration. And because compact adapters are tiny compared with full frontier image models, the long-term production path is economically viable even if a stronger teacher model is used during training or demo generation.

## Run Locally

Install dependencies:

```bash
cd ~/autohdr
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-ml.txt
```

Launch the Streamlit app:

```bash
streamlit run app.py
```

Important note: the approved showcase grid in this repo was generated with FLUX.2 Pro via OpenRouter. The local SD 1.5 LoRA path remains in the repo as a prototype and research artifact, but the final demo image used for packaging is `demo_grid_flux.png`.

For Streamlit Community Cloud, the repo uses a lightweight `requirements.txt` with only the app dependency. The full local ML stack lives in `requirements-ml.txt`.
