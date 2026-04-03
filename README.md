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
   BLIP-2 captions historical edits for each photographer profile, and DeepSeek compresses those captions into one style sentence.
2. **Teacher generation**
   FLUX.1 Kontext [pro] generates content-preserving target edits from the source image plus the profile-specific style sentence.
3. **Student distillation**
   InstructPix2Pix with LoRA is trained on `(source -> teacher)` pairs so the deployed model is a compact, photographer-specific adapter.
4. **Demo app**
   A Streamlit app presents the pipeline, the qualitative evidence, and local profile-specific inference.

## Why The V2 Teacher-Student Approach Is Stronger Than The Earlier Approach

The earlier style-only LoRA direction could imitate a visual distribution, but it was not a true source-conditioned editing model. That matters for listing workflows.

The v2 pipeline is stronger because the student learns from paired source and teacher outputs. The training objective matches the product requirement more closely: preserve the original scene while reproducing the teacher's finish. That is a better fit for property photography, where layout cues, furniture placement, windows, and exterior structure must remain stable.

## Repository Guide / Key Files

- `app.py`: Streamlit demo for the product story and local inference
- `assets/demo_grid_student_v2_r8.png`: strongest six-panel qualitative comparison
- `assets/pipeline_overview_v2.svg`: system diagram
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
