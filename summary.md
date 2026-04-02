# AutoHDR Summary

AutoHDR is a personalized AI photo-editing proof of concept for real estate photographers. The core idea is that each photographer should get their own compact editing model instead of sharing one generic enhancement system.

## Current Pipeline

1. A photographer's historical edited images are captioned with BLIP-2.
2. DeepSeek V3.2 summarizes those captions into a one-sentence style description.
3. FLUX.1 Kontext [pro] generates teacher edits that preserve the scene while shifting the image into that photographer's style.
4. An InstructPix2Pix LoRA student is distilled on `(raw -> teacher)` pairs.
5. The deployed result is one lightweight LoRA adapter per photographer.

## Current Demo Story

- Candidate C represents a brighter, cleaner, more neutral finish.
- Candidate D represents a cooler, darker, more muted finish.
- The Streamlit app presents the product narrative, the training evidence, and local student inference using the surviving v2 adapters.

## Canonical Visual Assets

- `assets/demo_grid_student_v2_final_train.png`
- `assets/teacher_preview_c_v2.png`
- `assets/teacher_preview_d_v2.png`
- `assets/loss_curves_v2.png`
- `assets/pipeline_overview_v2.svg`
