"""GPT-5.4 Vision perception: semantic reading of rendered pages and crops.

Consumes rendered artifacts only (never raw document bytes beyond the image).
Secrets stay in the gateway; nothing credential-like enters returned data.
"""

from __future__ import annotations

import base64
import io
import json
import re
from typing import Any, Dict, List

INSPECT_PROMPT = (
    "You are a precise document-vision reader. Study the rendered document image and "
    "answer the question using ONLY what is visibly present. "
    "When you state a concrete fact (entity, attribute, value), ALSO emit it as its "
    "own line in exactly this form:\n"
    "CLAIM: <subject> | <attribute> | <value>\n"
    "If text is ambiguous, say what is ambiguous instead of guessing.\n"
    "Question: {question}"
)

VERIFY_PROMPT = (
    "You are verifying one extracted fact against a rendered document image. "
    "Reply with a single JSON object: "
    '\'{{"supported": true|false, "rationale": "<short>", '
    '"confidence": 0.0-1.0}}\'. '
    "supported=true ONLY if the image visibly states the value. "
    "Fact: {subject} / {attribute} = {value!r}"
)


def vision_configured(config) -> bool:
    """True only when the openai-compatible backend is fully configured."""
    model = getattr(config, "model", config)
    if getattr(model, "backend", "") != "openai-compatible":
        return False
    return bool(
        getattr(model, "api_key", "")
        and getattr(model, "base_url", "")
        and getattr(model, "chat_deployment", "")
    )


def image_to_data_url(image_path: str, max_side: int = 1568) -> str:
    """Downscale (aspect-preserving) and encode as a JPEG data URL."""
    from PIL import Image

    with Image.open(image_path) as im:
        im = im.convert("RGB")
        im.thumbnail((max_side, max_side))
        buffer = io.BytesIO()
        im.save(buffer, format="JPEG", quality=82)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def inspect_images(
    gateway,
    image_paths: List[str],
    question: str,
    max_side: int = 1568,
    max_tokens: int = 1200,
) -> Dict[str, Any]:
    """Ask the vision model about rendered images. Raises ModelUnavailable/Error."""
    from ..models.schemas import ModelRequest

    urls = [image_to_data_url(path, max_side) for path in image_paths]
    response = gateway.complete(
        ModelRequest(
            kind="vision",
            prompt=INSPECT_PROMPT.format(question=question),
            images=urls,
            max_tokens=max_tokens,
        )
    )
    return {
        "answer": response.text,
        "backend": response.backend,
        "tokens_in": response.tokens_in,
        "tokens_out": response.tokens_out,
    }


def verify_claim_image(
    gateway,
    image_path: str,
    subject: str,
    attribute: str,
    value: Any,
    max_side: int = 1568,
) -> Dict[str, Any]:
    """Visual verification of one claim. Returns supported True/False/None."""
    from ..models.schemas import ModelRequest

    url = image_to_data_url(image_path, max_side)
    response = gateway.complete(
        ModelRequest(
            kind="vision",
            prompt=VERIFY_PROMPT.format(
                subject=subject, attribute=attribute, value=value
            ),
            images=[url],
            max_tokens=400,
            response_format="json_object",
        )
    )
    supported, rationale, confidence = _parse_verdict(response.text)
    return {
        "supported": supported,
        "rationale": rationale,
        "confidence": confidence,
        "backend": response.backend,
        "raw": response.text[:500],
    }


def _parse_verdict(text: str):
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned)
    try:
        payload = json.loads(cleaned)
        supported = payload.get("supported")
        if isinstance(supported, bool):
            return (
                supported,
                str(payload.get("rationale", ""))[:400],
                payload.get("confidence"),
            )
    except (ValueError, AttributeError):
        pass
    match = re.search(r'"supported"\s*:\s*(true|false)', text, re.IGNORECASE)
    if match:
        return (match.group(1).lower() == "true", text[:400], None)
    return (None, text[:400], None)
