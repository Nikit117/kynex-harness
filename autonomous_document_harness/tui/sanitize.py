"""Terminal-output sanitization + secret redaction.

Documents are untrusted: strip ANSI escapes / control characters before any
document-derived text reaches the terminal. Secrets never reach the UI.
"""

from __future__ import annotations

import os
import re

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|\x1b\][^\x07]*\x07|\x1b[()][0-9A-Z]")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

_SECRET_NAMES = (
    "ADH_MODEL_API_KEY",
    "ADH_DI_API_KEY",
    "AZURE_OPENAI_API_KEY",
)


def sanitize_text(text: object, max_length: int = 0) -> str:
    """Strip terminal-injection sequences and controls; optionally cap length."""
    cleaned = _ANSI_RE.sub("", str(text or ""))
    cleaned = _CONTROL_RE.sub("", cleaned)
    if max_length and len(cleaned) > max_length:
        cleaned = cleaned[:max_length] + "…"
    return cleaned


def redact_mapping(mapping: dict) -> dict:
    """Replace secret values with presence markers (never the value)."""
    redacted = dict(mapping)
    for key in list(redacted):
        upper = str(key).upper()
        if "KEY" in upper or "TOKEN" in upper or "SECRET" in upper:
            redacted[key] = "…set…" if redacted[key] else "…unset…"
    return redacted


def secret_values() -> set:
    """Secret values currently in process env (for scrubbing echoes)."""
    values = set()
    for name in _SECRET_NAMES:
        value = os.environ.get(name, "")
        if value:
            values.add(value)
    # Also bridge-mapped variants share the same values; length-guard.
    return {v for v in values if len(v) >= 8}


def scrub_secrets(text: str) -> str:
    """Replace any known secret value in text with a marker."""
    out = str(text or "")
    for value in secret_values():
        if value in out:
            out = out.replace(value, "[redacted-secret]")
    return out
