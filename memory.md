# Project Memory

## Important Decisions

- The project moved from an older style-only path to a stronger v2 teacher-student setup.
- FLUX.1 Kontext [pro] is the teacher for high-quality scene-preserving edits.
- InstructPix2Pix + LoRA is the student because it can be trained locally and deployed cheaply.
- The personalized deployment story is the main product pitch: one small adapter per photographer.

## Dataset Decisions

- The working subset uses 220 paired images per expert.
- Candidate C and Candidate D are the two photographer styles.
- The first 200 images act as training inputs and the last 20 are the reserved test-style slice used in the earlier plan.

## Surviving Outputs

- Student adapters survived the laptop crash.
- Style prompt files survived.
- Teacher outputs and preview grids survived.
- Demo grid and loss plot survived.

## Style Sentences

- Candidate C: Crisp, true-to-life images with balanced exposure, natural color saturation, and a clear, neutral tone that emphasizes architectural and environmental details.
- Candidate D: Cool, desaturated tones with lifted shadows, subtle contrast, and a muted color palette that evokes a calm, documentary mood.

## App Notes

- `app.py` is now a branded AutoHDR-style demo rather than a minimal lab page.
- The app includes a short explanation of the product and model architecture.
- The app includes a local student inference section and evidence sections for teacher previews and distillation results.

## Repo Notes

- Canonical presentation assets now live in `assets/`.
- `bfl_kontext.py` is required by `generate_teacher_kontext_v2.py`.
- `requirements.txt` is the lightweight app dependency file.
- `requirements-ml.txt` is the full local ML stack for training and generation.
