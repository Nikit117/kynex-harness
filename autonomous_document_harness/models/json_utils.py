"""Robust JSON parsing for model responses (fences, prose wrappers, repair)."""

from __future__ import annotations

import json
import re
from typing import Any, Tuple


def parse_json_response(text: str) -> Tuple[Any, bool]:
    """Return (payload, strict). strict=False means repaired/fallback parse."""
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned).strip()
    try:
        return json.loads(cleaned), True
    except ValueError:
        pass
    # Last-resort: largest {...} or [...] span.
    spans = [m for m in re.finditer(r"\{.*\}|\[.*\]", cleaned, re.DOTALL)]
    for match in sorted(spans, key=lambda m: len(m.group(0)), reverse=True):
        try:
            return json.loads(match.group(0)), False
        except ValueError:
            continue
    raise ValueError(f"no JSON object found in model response: {cleaned[:200]!r}")
