# Teacher-Student V2

This branch adds a cleaner distillation path:

1. `FLUX.1 Kontext [pro]` acts as the **teacher** and generates personalized edits from raw inputs.
2. `InstructPix2Pix + LoRA` acts as the **student** and learns the raw-to-teacher edit mapping.
3. The student can then be run locally at lower cost than the teacher model.

## Why This Version Is Stronger

The earlier SD 1.5 style-only LoRA learned a visual distribution, but it did not learn a true source-conditioned edit. This v2 path fixes that by training an edit-conditioned student whose UNet receives both:

- noisy target latents
- source image latents

That makes the training problem match the actual goal: preserve the source scene while learning the teacher's editing behavior.

## Files

- `bfl_kontext.py`
  Shared FLUX.1 Kontext helper functions for direct BFL image editing.
- `generate_teacher_kontext_v2.py`
  Generates teacher outputs from raw inputs into `teacher_kontext_v2/expert_c` and `teacher_kontext_v2/expert_d`.
- `train_student_ip2p.py`
  Trains edit-conditioned student LoRAs on `(raw -> teacher)` pairs using `timbrooks/instruct-pix2pix`.
- `generate_student_demo_v2.py`
  Runs the trained students on held-out images and optionally includes cached teacher outputs in the final grid.

## Recommended Workflow

Generate the teacher dataset:

```bash
cd ~/autohdr
source venv/bin/activate
caffeinate -i python generate_teacher_kontext_v2.py --expert both --limit 200
```

Train the student adapters:

```bash
caffeinate -i python train_student_ip2p.py --experts c d --steps 500 --limit 200
```

Generate a comparison grid:

```bash
python generate_student_demo_v2.py
```

## Outputs

- `teacher_kontext_v2/expert_c/*.jpg`
- `teacher_kontext_v2/expert_d/*.jpg`
- `student_ip2p_v2/expert_c/`
- `student_ip2p_v2/expert_d/`
- `assets/loss_curves_v2.png`
- `assets/demo_grid_student_v2_final_train.png`
- `assets/teacher_preview_c_v2.png`
- `assets/teacher_preview_d_v2.png`

## Practical Notes

- Teacher generation is the expensive step because it calls FLUX.1 Kontext [pro] through the direct BFL API.
- Student training is the cheaper path that preserves the distillation story.
- The student trainer now checkpoints automatically so a crash or sleep interruption does not wipe out a long local run.
- If you want teacher outputs for the held-out test images too, raise the teacher generation limit above `200`.
