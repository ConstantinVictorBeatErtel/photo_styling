# Photo Styling Personalized Style Distillation

## Problem

Different photographers and editors often prefer different color, contrast, and tonal treatments, but many AI tools still apply the same generic enhancement to every image.

## Approach

Three-stage pipeline:

- *Style Discovery:* BLIP-2 auto-captions past edits. DeepSeek V3.2 summarizes those captions into one concise style sentence.
- *Student Training:* Lightweight LoRA adapters are trained on real FiveK expert edits to capture style-specific visual tendencies.
- *Demo Generation:* FLUX.2 Pro applies the discovered style directions to the same raw images, producing a clean side-by-side comparison.

## Results

![Photo Styling Demo Grid](demo_grid_flux.png)

The demo grid shows the same raw input edited three ways: a generic baseline, a Candidate C style, and a Candidate D style. Candidate C produces brighter, crisper, more natural-looking outputs, while Candidate D pushes the same images toward cooler, flatter, more muted tones. The columns remain visually tied to the same source photo, but the editing intent clearly changes by style.

## Scale Path

With a larger edit history, style profiles can be inferred automatically from prior work, then paired with lightweight adapters or teacher-guided personalization workflows. No manual presets and no manual style authoring are required. Because compact adapters are tiny compared with full frontier image models, the longer-term production path can stay efficient even when a stronger teacher model is used for training or evaluation.

## Notes

The showcase grid in this repo was generated with FLUX.2 Pro via OpenRouter. The local SD 1.5 LoRA path remains in the repo as a research artifact, while the public demo focuses on the strongest visual comparison.

For Streamlit Community Cloud, the app uses the lightweight `requirements.txt`. The full local ML stack lives in `requirements-ml.txt`.
