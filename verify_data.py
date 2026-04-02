from __future__ import annotations

import random
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path("/Users/ConstiX/autohdr")
DATA_ROOT = PROJECT_ROOT / "data"
EXPECTED_COUNT = 220
RANDOM_CHECKS = 3


def check_folder(path: Path) -> tuple[int, list[str]]:
    files = sorted(path.glob("*.jpg"))
    problems: list[str] = []

    if len(files) != EXPECTED_COUNT:
        problems.append(f"expected {EXPECTED_COUNT}, found {len(files)}")

    samples = random.sample(files, k=min(RANDOM_CHECKS, len(files)))
    for sample in samples:
        with Image.open(sample) as image:
            if image.size != (512, 512):
                problems.append(f"{sample.name} size={image.size}")
            if image.mode != "RGB":
                problems.append(f"{sample.name} mode={image.mode}")

    return len(files), problems


def main() -> None:
    random.seed(42)
    folders = {
        "expert_c/raw": DATA_ROOT / "expert_c" / "raw",
        "expert_c/edited": DATA_ROOT / "expert_c" / "edited",
        "expert_d/raw": DATA_ROOT / "expert_d" / "raw",
        "expert_d/edited": DATA_ROOT / "expert_d" / "edited",
    }

    print("Verification summary:")
    failed = False
    for label, path in folders.items():
        count, problems = check_folder(path)
        status = "OK" if not problems else "FAIL"
        print(f"  {label:<16} count={count:<3} status={status}")
        for problem in problems:
            print(f"    - {problem}")
        failed = failed or bool(problems)

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
