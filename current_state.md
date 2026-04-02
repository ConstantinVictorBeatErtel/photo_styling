# Current State

## Status

The v2 teacher-student demo is in a usable state.

- The Streamlit app has been updated to match the teacher-student pipeline.
- The app now points at the stronger `student_ip2p_v2_r8` adapters and the new `r8` comparison grid.
- The app can run local student inference when the ML environment and student adapters are available.
- The key visual assets now live under `assets/`.
- The current checked-in state is on `main`.

## Confirmed Artifacts

- `student_ip2p_v2/expert_c/adapter_model.safetensors`
- `student_ip2p_v2/expert_d/adapter_model.safetensors`
- `student_ip2p_v2_r8/expert_c/adapter_model.safetensors`
- `student_ip2p_v2_r8/expert_d/adapter_model.safetensors`
- `style_c.txt`
- `style_d.txt`
- `assets/demo_grid_student_v2_r8.png`
- `assets/loss_curves_v2_r8.png`
- `assets/teacher_preview_c_v2.png`
- `assets/teacher_preview_d_v2.png`

## What Was Cleaned Up

- The README now matches the actual v2 assets.
- Final demo assets are standardized under `assets/`.
- Duplicate root-level demo images and generated cache files were removed.
- Project tracking docs were restored.

## Recommended Next Work

1. Review the Streamlit app visually and make any last presentation tweaks.
2. Regenerate teacher outputs if you want to fill the last few missing teacher images.
3. If desired, open a PR from the current `main` state or cut a fresh follow-up branch.
