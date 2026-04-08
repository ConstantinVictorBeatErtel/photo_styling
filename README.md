# Personalized Listing Photo Editing

## One-Paragraph Pitch

This repository is a product-oriented ML prototype for personalized real-estate photo editing. The core idea is simple: a listing platform should not force every photographer through the same generic enhancement model. Instead, it should learn each photographer's preferred finish from historical edits, distill that behavior into a compact adapter, and apply that finish to new scenes while preserving room layout, framing, and content.

The deployable unit in this repo is one LoRA adapter per photographer profile. That is the main product point: personalized output without serving a heavyweight teacher model on every request.

## Why One Model Per Photographer Matters

Real-estate editing has a specific product constraint set:

- photographers want consistency across listings, not one generic enhancement pass
- edits must preserve scene geometry and content
- deployment needs to be cheap enough to support many photographers, each with their own finish

This prototype is built around those constraints. It is not a general "artistic styling" demo. It is a source-conditioned editing system designed to keep the scene intact while changing exposure, color, white balance, contrast, and finish.

## Why This Matters For Real-Estate Photo Editing

- **Consistency across listings**: one photographer can maintain a recognizable finish across multiple homes without hand-building presets every time.
- **Scene preservation**: the system is designed around source-conditioned editing, which is more compatible with interiors and exteriors than unconstrained image generation.
- **Photographer-specific finish**: two photographers can produce different edits from the same source image without maintaining separate heavyweight models.
- **Cheaper deployment**: the repo ships compact student adapters instead of requiring a large teacher model to serve every request.

## System Overview

The current repository centers on the v2 teacher-student pipeline:

1. **Style discovery**
   BLIP-2 captions historical edits for each photographer profile, and DeepSeek now turns those captions into a small structured style profile with a summary plus editable style facets.
2. **Teacher generation**
   FLUX.1 Kontext [pro] generates content-preserving target edits from the source image plus a composed prompt built from that style profile.
3. **Student distillation**
   InstructPix2Pix with LoRA is trained on `(source -> teacher)` pairs using the same richer composed prompt, so the deployed model is a compact, photographer-specific adapter.
4. **Demo app**
   A Streamlit app presents the pipeline, the qualitative evidence, and local profile-specific inference.

## Why The V2 Teacher-Student Approach Is Stronger Than The Earlier Approach

The earlier style-only LoRA direction could imitate a visual distribution, but it was not a true source-conditioned editing model. That matters for listing workflows.

The v2 pipeline is stronger because the student learns from paired source and teacher outputs. The training objective matches the product requirement more closely: preserve the original scene while reproducing the teacher's finish. That is a better fit for property photography, where layout cues, furniture placement, windows, and exterior structure must remain stable.

## Repository Guide / Key Files

- `app.py`: Streamlit demo for the product story and local inference
- `assets/demo_grid_student_v2_r8.png`: strongest six-panel qualitative comparison
- `assets/demo_grid_cross_attention_story.png`: prompt-only vs cross-attention comparison on the same source image
- `assets/pipeline_overview_v2.svg`: system diagram
- `models/style_cross_attention.py`: small educational multimodal fusion block
- `scripts/style_profile.py`: shared loader/composer for structured style profiles
- `scripts/train_student_ip2p.py`: source-conditioned student training script
- `scripts/generate_teacher_kontext_v2.py`: teacher generation script
- `scripts/generate_style.py`: style discovery script
- `scripts/preprocess.py`: dataset preprocessing
- `student_ip2p_v2_r8/`: current best tracked student adapters
- `TEACHER_STUDENT_V2.md`: compact technical note for the v2 pipeline

## Results

![Six-panel comparison](assets/demo_grid_student_v2_r8.png)

The strongest tracked result in this repo is the `r=8`, `2000`-step student run shown above. The comparison is intentionally structured around the product question:

- `Input`: the source image fed into every branch
- `Generic edit`: the non-personalized baseline from the base model
- `Teacher outputs`: the high-quality target edits from the teacher
- `Student outputs`: the distilled photographer-specific adapters

What success looks like here:

- the scene stays intact across all edits
- the two profile-specific students produce clearly different finishes
- the student outputs track the teacher direction more closely than the generic baseline

| Item | Current tracked value |
| --- | --- |
| Teacher model | `FLUX.1 Kontext [pro]` |
| Student model | `timbrooks/instruct-pix2pix` + LoRA |
| Tracked finish adapters | `2` (`bright-neutral`, `cool-muted`) |
| Style discovery input | `50` edited images per profile |
| Preprocessed portfolio size | `220` source/edit pairs per profile |
| Requested student train split | first `200` images per profile |
| Matched teacher pairs in best run | `199` for bright-neutral, `198` for cool-muted |
| LoRA rank | `8` |
| Training steps | `2000` |
| Student artifact type | LoRA adapter (`.safetensors`) |
| Adapter size | about `3.06 MB` per profile |
| Demo app inference | local inference supported when ML stack and weights are present |

## Cross-Attention Example

![Prompt-only vs cross-attention comparison](assets/demo_grid_cross_attention_story.png)

The original tracked grid stays at the top because it remains the cleanest overall repository summary. The new comparison above is there to show what the custom cross-attention block is actually doing.

How to read it:

- the top row is the current tracked prompt-only student
- the bottom row is the cross-attention student on the same source image
- both rows use the same six-panel layout so the difference is about conditioning, not presentation

What looks better in the cross-attention row:

- the `cool-muted` student separates more clearly from `bright-neutral`, which is the main product goal for profile-specific editing
- the cool treatment is pushed more deliberately into the sky, shadows, and statue highlights instead of reading like a lighter version of the neutral finish
- scene structure still stays intact, so the stronger style separation is not coming from a content rewrite

> Why cross-attention helps
>
> The student is no longer relying on style text alone. Source image features can attend over style tokens, which makes the conditioning more source-aware. On this example, that mainly shows up as stronger cool-muted separation without losing the scene.

> How we implemented it
>
> We kept the change intentionally small: source tokens come from the frozen VAE latent map, style tokens come from the existing CLIP text encoder, and a custom cross-attention block produces one extra fused conditioning token that is appended before the UNet call.

The tradeoff is that the `bright-neutral` student also drifts a little cooler than the prompt-only baseline on this example. So the honest takeaway is not "cross-attention wins everywhere." It is:

> the fusion block is doing something real and explainable, and on this image it improves style separation most clearly for the cool-muted profile

## Evaluation Protocol

This repository does **not** claim benchmark-level quantitative evaluation. That would require a cleaner domain-specific dataset and an agreed metric suite.

What is included and defensible:

- held-out qualitative comparison assets
- a six-panel teacher/student comparison
- tracked style prompts for two photographer profiles
- tracked deployment artifacts for the strongest student run
- a local demo path that can run profile-specific inference

What is **not** included:

- PSNR / LPIPS / CLIP benchmark claims
- serving latency claims
- production throughput claims
- user study results

That is intentional. The repo is honest about being a product-minded prototype, not a finished production system.

## How To Run The App

For the lightweight app experience:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

For local ML inference and full pipeline work:

```bash
python -m pip install -r requirements-ml.txt
python scripts/preprocess.py
streamlit run app.py
```

The app can always display the tracked evidence assets. Live profile-specific generation requires the local ML stack plus the tracked student adapters.

## Structured Style Profiles

The main Level 1 upgrade in this repository is that a photographer style is no longer collapsed immediately into one sentence.

Each profile can now be stored as:

- `style_c.json`
- `style_d.json`

The JSON schema is intentionally small and interview-friendly:

```json
{
  "schema_version": 1,
  "profile_name": "expert_c",
  "style_summary": "Crisp, true-to-life images with balanced exposure and a clear neutral tone.",
  "facets": {
    "exposure": "balanced exposure with bright but controlled luminance",
    "contrast": "clean moderate contrast",
    "white_balance": "neutral white balance",
    "saturation": "natural, believable saturation",
    "highlight_handling": "preserve bright highlights without blowing windows",
    "shadow_handling": "keep shadows open enough for room detail",
    "scene_tendency": "works cleanly across interiors and exteriors",
    "mood": "clear, polished, restrained"
  },
  "composed_prompt": "Crisp, true-to-life images with balanced exposure and a clear neutral tone. Exposure: balanced exposure with bright but controlled luminance. Contrast: clean moderate contrast. ..."
}
```

Why this is better than a single sentence:

- it preserves a few product-relevant editing dimensions instead of forcing brightness, contrast, white balance, saturation, and finish into one compressed description
- it stays lightweight because the model architecture does not change; we only improve the text conditioning string that the existing pipeline already consumes
- it is easy to explain: we keep a compact structured style profile, then compose a richer prompt from it for both teacher generation and student training

## New Style Generation Flow

Generate structured profiles from historical edits:

```bash
python scripts/generate_style.py
```

This now does four things:

- captions edited images with BLIP-2, as before
- saves caption artifacts under `artifacts/style_discovery/`
- writes `style_c.json` and `style_d.json`
- keeps `style_c.txt` and `style_d.txt` as legacy fallback files

Train the student with the richer conditioning prompt:

```bash
caffeinate -i python scripts/train_student_ip2p.py --experts c d --limit 200
```

Teacher generation also picks up the richer composed prompt automatically:

```bash
caffeinate -i python scripts/generate_teacher_kontext_v2.py --expert both --limit 200
```

## Backward Compatibility

- `scripts/train_student_ip2p.py` first looks for `style_<expert>.json`
- if the JSON file is missing, it falls back to the old `style_<expert>.txt`
- `scripts/generate_style.py` still writes the legacy txt files so older flows do not break
- the training loop and LoRA architecture are unchanged; only the style-conditioning text is richer

## Educational Cross-Attention Module

This repository now also includes a small custom cross-attention path for practicing multimodal fusion without turning the project into a large architecture rewrite.

What cross-attention means here:

- **queries** come from source image features
- **keys** come from style text token features
- **values** also come from style text token features

In this project, the implementation is intentionally simple:

- source image features come from the frozen VAE latent map that already exists in the student pipeline
- style features come from the existing CLIP text encoder token embeddings for the composed style prompt
- a small module in `models/style_cross_attention.py` lets source tokens attend over style tokens
- the fused result is pooled into one extra conditioning token and appended to the normal prompt embeddings before the UNet call

How this differs from the original pipeline:

- before, the student only saw the standard prompt embeddings from the text encoder
- now, the student can optionally see one extra style-aware token built from both the source image and the style text
- the UNet itself is unchanged; the custom block lives just before the existing conditioning path

Why this is still a simplified educational version:

- there is no BLIP-2 rewrite
- there is no custom diffusion backbone
- there is no deep UNet surgery
- older checkpoints still work because inference falls back to the original prompt-only path when no saved fusion weights are present

Typical training command:

```bash
caffeinate -i python scripts/train_student_ip2p.py --experts c d --limit 200
```

Optional knobs for the fusion block:

```bash
python scripts/train_student_ip2p.py \
  --experts c d \
  --style-fusion-hidden-dim 256 \
  --style-fusion-heads 4 \
  --style-source-grid-size 16
```

To disable the custom module and recover the original conditioning path:

```bash
python scripts/train_student_ip2p.py --experts c d --disable-style-cross-attention
```

## Prototype Boundaries / What Is Not Production-Ready Yet

### Production-relevant ideas already demonstrated

- one adapter per photographer as the deployable unit
- source-conditioned teacher-to-student distillation
- clear separation between expensive teacher generation and cheaper student inference
- local packaging of profile-specific weights

### Prototype-only parts

- the dataset is MIT-Adobe FiveK, not a dedicated real-estate benchmark
- evaluation is primarily qualitative
- teacher generation depends on an external paid API
- there is no orchestration layer for photographer onboarding, adapter lifecycle, or serving

## Limitations

- The strongest run is still based on a generic photo dataset, not listing-only imagery.
- A few teacher outputs are missing, so the strongest students were trained on `199` and `198` matched pairs rather than a perfectly complete `200/200`.
- The repo does not include quantitative benchmark claims beyond tracked training configuration and artifacts.
- Hosted Streamlit behavior uses bundled sample outputs when the full ML stack is unavailable.

## Next Steps

- swap the training and evaluation data to real listing-photo edit histories
- add explicit scene-preservation and edit-consistency evaluation
- add photographer onboarding and adapter versioning
- separate offline teacher generation from online student serving
- define deployment policies for fallback behavior, quality review, and retraining cadence

## Supporting Notes

- `TEACHER_STUDENT_V2.md` explains why the source-conditioned v2 path is the right one.
- `student_ip2p_v2_r8/expert_c/` and `student_ip2p_v2_r8/expert_d/` contain the current best tracked adapters.
