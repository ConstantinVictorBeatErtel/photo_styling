from __future__ import annotations

import random
from pathlib import Path

from PIL import Image


DATA_ROOT = Path(__file__).resolve().parent / "data"
EXPECTED_COUNT = 220
EXPECTED_SIZE = (512, 512)
EXPECTED_MODE = "RGB"
SAMPLE_COUNT = 3


def check_folder(path: Path) -> tuple[int, list[str]]:
    files = sorted(path.glob("*.jpg"))
    issues: list[str] = []

    if len(files) != EXPECTED_COUNT:
        issues.append(f"count={len(files)} expected={EXPECTED_COUNT}")

    sample_files = random.Random(42).sample(files, k=min(SAMPLE_COUNT, len(files)))
    for sample in sample_files:
        with Image.open(sample) as image:
            if image.size != EXPECTED_SIZE or image.mode != EXPECTED_MODE:
                issues.append(
                    f"{sample.name}: size={image.size} mode={image.mode}"
                )

    return len(files), issues


def main() -> None:
    folders = {
        "expert_c/raw": DATA_ROOT / "expert_c" / "raw",
        "expert_c/edited": DATA_ROOT / "expert_c" / "edited",
        "expert_d/raw": DATA_ROOT / "expert_d" / "raw",
        "expert_d/edited": DATA_ROOT / "expert_d" / "edited",
    }

    print("Folder              Count  Status")
    print("-----------------------------------")
    for label, path in folders.items():
        count, issues = check_folder(path)
        status = "OK" if not issues else "FAIL"
        print(f"{label:<18} {count:<5}  {status}")
        for issue in issues:
            print(f"  - {issue}")


if __name__ == "__main__":
    main()
