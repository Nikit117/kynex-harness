"""Routing: when is a model call justified? Heuristic-first, model as escalation."""

from __future__ import annotations

from ..core.config import RuntimeConfig


def should_use_model(config: RuntimeConfig, kind: str) -> bool:
    """True only when a real provider is configured AND the kind needs it."""
    if config.model.backend != "openai-compatible":
        return False
    model = config.model
    configured = bool(model.api_key and model.base_url and model.chat_deployment)
    return configured and kind in ("vision", "entail", "chat", "extract")
