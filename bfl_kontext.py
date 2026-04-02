from __future__ import annotations

import base64
import os
import time
from pathlib import Path
from typing import Final

import requests
from PIL import Image

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional outside the full local env
    load_dotenv = None


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent
BFL_BASE_URL: Final[str] = "https://api.bfl.ai/v1"
KONTEXT_PRO_ENDPOINT: Final[str] = f"{BFL_BASE_URL}/flux-kontext-pro"
DEFAULT_CREATE_TIMEOUT: Final[tuple[int, int]] = (30, 120)
DEFAULT_FETCH_TIMEOUT: Final[tuple[int, int]] = (30, 240)
DEFAULT_POLL_INTERVAL_SECONDS: Final[float] = 0.5
DEFAULT_POLL_TIMEOUT_SECONDS: Final[float] = 600.0
DEFAULT_REQUEST_RETRIES: Final[int] = 5


class KontextRequestError(RuntimeError):
    pass


class KontextModeratedError(KontextRequestError):
    pass


def load_bfl_api_key(env_path: Path | None = None) -> str:
    if load_dotenv is not None:
        load_dotenv(env_path or PROJECT_ROOT / ".env")
    api_key = os.environ.get("BFL_API_KEY", "").strip()
    if not api_key or api_key == "YOUR_KEY_HERE":
        raise RuntimeError("BFL_API_KEY is missing or still a placeholder in .env")
    return api_key


def image_path_to_base64(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode("utf-8")


def request_with_retry(
    method: str,
    url: str,
    *,
    headers: dict[str, str],
    json: dict[str, object] | None = None,
    timeout: tuple[int, int] = DEFAULT_FETCH_TIMEOUT,
    retries: int = DEFAULT_REQUEST_RETRIES,
) -> requests.Response:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.request(method, url, headers=headers, json=json, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            last_error = exc
            if attempt == retries:
                break
            wait_seconds = min(5 * attempt, 20)
            print(
                f"{method} {url} failed on attempt {attempt}/{retries}: {exc}. Retrying in {wait_seconds}s...",
                flush=True,
            )
            time.sleep(wait_seconds)

    assert last_error is not None
    raise last_error


def create_kontext_edit_request(
    api_key: str,
    *,
    prompt: str,
    image_path: Path,
    seed: int | None = None,
    output_format: str = "jpeg",
    prompt_upsampling: bool = False,
    safety_tolerance: int = 2,
) -> tuple[str, str]:
    payload: dict[str, object] = {
        "prompt": prompt,
        "input_image": image_path_to_base64(image_path),
        "output_format": output_format,
        "prompt_upsampling": prompt_upsampling,
        "safety_tolerance": safety_tolerance,
    }
    if seed is not None:
        payload["seed"] = seed

    response = request_with_retry(
        "POST",
        KONTEXT_PRO_ENDPOINT,
        headers={
            "accept": "application/json",
            "x-key": api_key,
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=DEFAULT_CREATE_TIMEOUT,
    )
    response.raise_for_status()
    data = response.json()
    return data["id"], data["polling_url"]


def poll_kontext_result(
    api_key: str,
    polling_url: str,
    *,
    poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
    poll_timeout_seconds: float = DEFAULT_POLL_TIMEOUT_SECONDS,
) -> str:
    deadline = time.time() + poll_timeout_seconds
    last_status = "Pending"

    while time.time() < deadline:
        response = request_with_retry(
            "GET",
            polling_url,
            headers={
                "accept": "application/json",
                "x-key": api_key,
            },
            timeout=DEFAULT_FETCH_TIMEOUT,
        )
        data = response.json()
        status = data.get("status", "Unknown")
        last_status = status

        if status == "Ready":
            return data["result"]["sample"]
        if status == "Content Moderated":
            raise KontextModeratedError(f"Kontext request was content moderated: {data}")
        if status in {"Error", "Failed"}:
            raise KontextRequestError(f"Kontext request failed: {data}")

        time.sleep(poll_interval_seconds)

    raise TimeoutError(f"Timed out waiting for Kontext result. Last status: {last_status}")


def edit_image_with_kontext(
    api_key: str,
    *,
    prompt: str,
    image_path: Path,
    seed: int | None = None,
    output_format: str = "jpeg",
) -> Image.Image:
    _, polling_url = create_kontext_edit_request(
        api_key,
        prompt=prompt,
        image_path=image_path,
        seed=seed,
        output_format=output_format,
    )
    sample_url = poll_kontext_result(api_key, polling_url)
    response = request_with_retry(
        "GET",
        sample_url,
        headers={},
        timeout=DEFAULT_FETCH_TIMEOUT,
    )
    from io import BytesIO

    return Image.open(BytesIO(response.content)).convert("RGB")
