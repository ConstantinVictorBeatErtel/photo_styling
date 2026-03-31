from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


PROJECT_ROOT = Path("/Users/ConstiX/autohdr")
TOOL_ROOT = PROJECT_ROOT / "vendor" / "mit-adobe-fivek-dataset"
DATASET_ROOT = Path(os.path.expanduser("~/autohdr/fivek_dataset"))
DATASET_DIR = DATASET_ROOT / "MITAboveFiveK"
EXPERTS = ("c", "d")
LIMIT = 220
MAX_WORKERS = 8

if str(TOOL_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOL_ROOT))

from dataset.fivek_builder import MITAboveFiveKBuilder, download  # noqa: E402


def print_first_five(directory: Path, label: str) -> None:
    files = sorted(path.name for path in directory.glob("*"))
    print(f"{label}: {directory}")
    for name in files[:5]:
        print(f"  - {name}")
    if not files:
        print("  - <no files found>")


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def download_if_missing(url: str, destination: Path) -> str:
    ensure_parent(destination)
    if destination.exists() and destination.stat().st_size > 0:
        return destination.name
    download(url, str(destination))
    return destination.name


def run_stage(label: str, jobs: list[tuple[str, str, Path]]) -> None:
    if not jobs:
        print(f"{label}: nothing to download")
        return

    print(f"{label}: downloading {len(jobs)} files with {MAX_WORKERS} workers")
    completed = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(download_if_missing, url, destination)
            for _, url, destination in jobs
        ]
        for future in as_completed(futures):
            future.result()
            completed += 1
            if completed % 10 == 0 or completed == len(jobs):
                print(f"  {label}: {completed}/{len(jobs)}")


def main() -> None:
    print(f"Using dataset root: {DATASET_ROOT}")
    builder = MITAboveFiveKBuilder(
        dataset_dir=str(DATASET_DIR),
        config_name="per_camera_model",
        experts=list(EXPERTS),
    )

    metadata = builder.metadata("train")
    builder._metadata = metadata
    basenames = list(metadata.keys())[:LIMIT]
    print(
        f"Selected first {LIMIT} official training items "
        f"({basenames[0]} -> {basenames[-1]})"
    )

    raw_jobs: list[tuple[str, str, Path]] = []
    expert_jobs: list[tuple[str, str, Path]] = []

    for basename in basenames:
        item = metadata[basename]
        raw_path = Path(builder.raw_file_path(basename))
        raw_jobs.append((basename, item["urls"]["dng"], raw_path))
        for expert in EXPERTS:
            expert_path = Path(builder.expert_file_path(basename, expert))
            expert_jobs.append(
                (basename, item["urls"]["tiff16"][expert], expert_path)
            )

    run_stage("Raw DNG", raw_jobs)
    for expert in EXPERTS:
        stage_jobs = [job for job in expert_jobs if f"tiff16_{expert}" in str(job[2])]
        run_stage(f"Expert {expert.upper()} TIFF", stage_jobs)

    print(f"Ready basenames: {len(basenames)}")
    sample = metadata[basenames[0]]["files"]
    print("Sample entry:", sample)
    print_first_five(DATASET_DIR / "processed" / "tiff16_c", "Expert C TIFFs")
    print_first_five(DATASET_DIR / "processed" / "tiff16_d", "Expert D TIFFs")


if __name__ == "__main__":
    main()
