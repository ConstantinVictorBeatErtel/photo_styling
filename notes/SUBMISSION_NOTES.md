# Submission Notes

## What Changed

- removed internal progress notes and stale intermediate artifacts from the tracked repo
- fixed hardcoded local paths in preprocessing and verification scripts
- aligned training and demo defaults with the strongest current run (`r=8`, `2000` steps)
- rewrote the README around real-estate workflow constraints, deployment logic, and honest limitations
- tightened the Streamlit copy so it reads like a product demo instead of a class project
- reframed the app around scene preservation, photographer-specific finish, and compact deployment artifacts

## What Remains Imperfect

- the dataset is not a dedicated listing-photo benchmark
- evaluation is still qualitative-first rather than metric-heavy
- teacher generation depends on a paid external model and a local setup
- the strongest current run is missing a few teacher targets, so the student training set is slightly incomplete

## Three Talking Points To Emphasize

1. The deployable artifact is one compact LoRA per photographer, which is much more realistic than serving a heavyweight teacher model per photographer.
2. The v2 system is source-conditioned, so it is built to preserve scene content while matching the photographer's finish.
3. The repo is intentionally honest about what is prototype-level versus what would still need to be built for production.

## Three Limitations To Be Ready To Discuss

1. The evaluation is qualitative and deployment-oriented, not a full benchmark suite.
2. MIT-Adobe FiveK is a reasonable proof-of-concept dataset, but it is not a real-estate-specific validation set.
3. The current demo proves the architecture and packaging story more strongly than it proves production-quality editing across a large listing-photo distribution.

## Suggested GitHub Metadata

- Repo description: `Personalized real-estate photo editing prototype with FLUX teacher generation and compact per-photographer LoRA students`
- Website field: `Streamlit demo URL for the current main branch`
- Topics: `machine-learning`, `computer-vision`, `diffusion-models`, `lora`, `streamlit`, `real-estate`, `photo-editing`, `pytorch`
