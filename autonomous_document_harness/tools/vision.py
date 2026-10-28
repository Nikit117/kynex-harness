"""Page rendering (pypdfium2) + region crops (pillow). No model calls here."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any, Dict, Tuple


def image_fingerprint(image_path: str) -> str:
    """sha256 of the exact rendered bytes (traceability across retries)."""
    digest = hashlib.sha256()
    with open(image_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def render_page_to_png(
    pdf_path: str,
    page_no: int,
    dpi: int,
    output_path: str,
    timeout_seconds: float = 180.0,
) -> Dict[str, Any]:
    """Render one PDF page to PNG in an ISOLATED subprocess.

    PDFium can crash natively on pathological pages; running the render out
    of process converts such crashes into a catchable error so the agent can
    try another page instead of dying. Raises RuntimeError (missing backend),
    TimeoutError (hung render), or ValueError/RenderCrashed for failures.
    """
    import json
    import subprocess

    try:
        import pypdfium2  # noqa: F401 (presence probe only; render runs isolated)
    except ImportError as exc:
        raise RuntimeError("pypdfium2 is not installed") from exc
    worker = str(Path(__file__).resolve().parent / "render_worker.py")
    try:
        completed = subprocess.run(
            [
                sys.executable,
                worker,
                str(pdf_path),
                str(page_no),
                str(dpi),
                str(output_path),
            ],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(
            f"render timed out after {timeout_seconds}s (page {page_no})"
        ) from exc
    if completed.returncode != 0:
        detail = (completed.stdout or completed.stderr or "")[:200]
        raise RenderCrashed(f"renderer failed on page {page_no}: {detail}")
    try:
        info = json.loads(completed.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError) as exc:
        raise RenderCrashed(f"renderer returned bad output: {exc}")
    if "error" in info:
        raise RenderCrashed(f"renderer error on page {page_no}: {info['error']}")
    return info


class RenderCrashed(RuntimeError):
    """The isolated renderer died on this page (native crash or bad output)."""


def crop_region(
    image_path: str, bbox_px: Tuple[float, float, float, float], output_path: str
) -> Dict[str, Any]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("pillow is not installed") from exc
    with Image.open(image_path) as im:
        x0, y0, x1, y1 = (int(v) for v in bbox_px)
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, im.width), min(y1, im.height)
        if x1 <= x0 or y1 <= y0:
            raise ValueError(
                f"crop {bbox_px} is empty for image size {(im.width, im.height)}"
            )
        crop = im.crop((x0, y0, x1, y1))
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        crop.save(output_path)
        return {
            "image_path": output_path,
            "width": crop.width,
            "height": crop.height,
            "bbox_px": [x0, y0, x1, y1],
        }
