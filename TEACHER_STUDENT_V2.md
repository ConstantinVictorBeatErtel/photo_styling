# Teacher-Student Pipeline Notes

This repository is organized around the second-generation pipeline, which is the version worth discussing with an ML or product reviewer:

1. `FLUX.1 Kontext [pro]` is the teacher and produces high-quality, content-preserving target edits.
2. `InstructPix2Pix + LoRA` is the student and learns the raw-to-teacher edit mapping.
3. The deployable artifact is one compact LoRA adapter per photographer profile instead of a heavyweight teacher service per request.

## Why V2 Is The Right Framing

The earlier style-only LoRA direction could imitate a look, but it was not a true source-conditioned edit model. The v2 student is trained on paired `(source image -> teacher edit)` examples, so the model is explicitly optimized to keep scene layout, geometry, and objects intact while matching the teacher's finish.

That source-conditioned setup is much closer to a listing-photo workflow, where the requirement is not "make a pretty image" but "keep the room or exterior exactly the same while matching the photographer's preferred finish."

## Core Files

- `scripts/bfl_kontext.py`: BFL API helper used for FLUX.1 Kontext teacher generation.
- `scripts/generate_teacher_kontext_v2.py`: Builds teacher outputs for the two tracked photographer finishes.
- `scripts/train_student_ip2p.py`: Trains the current best student adapters (`r=8`, `2000` steps by default).
- `scripts/generate_student_demo_v2.py`: Generates the six-panel comparison grid used in the repo and Streamlit app.

## Typical Local Workflow

From the repository root:

```bash
python -m pip install -r requirements-ml.txt
```

Generate teacher outputs:

```bash
caffeinate -i python scripts/generate_teacher_kontext_v2.py --expert both --limit 200
```

Train the student adapters:

```bash
caffeinate -i python scripts/train_student_ip2p.py --experts c d --limit 200
```

Render the comparison asset:

```bash
python scripts/generate_student_demo_v2.py --filenames 0198.jpg
```

## Current Best Tracked Outputs

- `student_ip2p_v2_r8/expert_c/`
- `student_ip2p_v2_r8/expert_d/`
- `assets/demo_grid_student_v2_r8.png`
- `assets/loss_curves_v2_r8.png`
- `assets/teacher_preview_c_v2.png`
- `assets/teacher_preview_d_v2.png`

## Practical Notes

- Teacher generation is the expensive step because it uses a paid external model.
- Student adapters are small enough to ship with the repository and demonstrate the deployment story directly.
- A few teacher outputs are missing from the strongest local run, so the final `r8` students were trained on `199` bright-neutral pairs and `198` cool-muted pairs instead of a perfect `200/200`.
