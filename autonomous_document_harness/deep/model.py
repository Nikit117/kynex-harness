"""LangChain chat-model construction for the Deep Agents runtime.

Builds an AzureOpenAI-compatible BaseChatModel from environment variables
(ADH_MODEL_* or AZURE_OPENAI_*). No credentials are stored in code or state.
"""

from __future__ import annotations

import os
from typing import Any, Optional


def _first_env(*names: str) -> str:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""


def build_chat_model(
    backend: str = "",
    temperature: float = 0.0,
    max_tokens: int = 4096,
    request_timeout: Optional[float] = None,
    max_retries: Optional[int] = None,
):
    """Return a BaseChatModel or None when no backend is configured.

    backend: "" (auto) | "openai-compatible" | "azure" | "heuristic".
    Both "azure" and "openai-compatible" map to AzureChatOpenAI because the
    deployment exposes the Azure OpenAI chat-completions REST shape
    (api-key auth, api-version query). Credentials come from AZURE_OPENAI_*
    or ADH_MODEL_* environment variables — never from code.

    request_timeout / max_retries are only forwarded when given. They exist for
    unattended batch jobs: a single bulk page can legitimately take minutes
    (a 450-record page is ~20k output tokens), so there is no safe universal
    default, but a job with no timeout at all can hang forever — which is one of
    the ways the 2026-10-19 benchmark run looked "stuck".
    """
    backend = backend or _first_env("ADH_MODEL_BACKEND", "azure")
    if backend == "heuristic":
        return None
    base_url = _first_env("ADH_MODEL_BASE_URL", "AZURE_OPENAI_ENDPOINT")
    deployment = _first_env("ADH_MODEL_DEPLOYMENT", "AZURE_OPENAI_CHAT_DEPLOYMENT")
    api_key = _first_env("ADH_MODEL_API_KEY", "AZURE_OPENAI_API_KEY")
    api_version = (
        _first_env("ADH_MODEL_API_VERSION", "AZURE_OPENAI_API_VERSION") or "2024-10-21"
    )
    if not (base_url and api_key):
        return None
    if not deployment:
        return None
    try:
        from langchain_openai import AzureChatOpenAI

        guards: dict[str, Any] = {}
        if request_timeout is not None:
            guards["request_timeout"] = request_timeout
        if max_retries is not None:
            guards["max_retries"] = max_retries
        return AzureChatOpenAI(
            azure_endpoint=base_url,
            api_key=api_key,
            api_version=api_version,
            azure_deployment=deployment,
            temperature=temperature,
            max_tokens=max_tokens,
            **guards,
        )
    except Exception:
        return None


def build_tracked_chat_model(*args: Any, **kwargs: Any):
    """A chat model whose every response is added to the usage counter.

    The leaderboard reports cost per page, which needs the token usage of all
    calls (the agent loop and the bulk page-wise extractor). Wrapping the
    model keeps the accounting in one place instead of at every call site.
    """
    from .usage import TrackedChatModel

    model = build_chat_model(*args, **kwargs)
    if model is None:
        return None
    return TrackedChatModel(model)


def ensure_azure_env() -> None:
    """Bridge Azure credentials into this process (in-process only).

    Loads the sibling checkout's .env (path via ADH_LEGACY_ENV or derived)
    with override=False, then maps AZURE_OPENAI_* onto the ADH_MODEL_*
    family via setdefault so the gateway, vision path and LangChain model
    share one config. Explicit process env always wins. Never prints secrets.
    """
    import os
    from pathlib import Path as _Path

    legacy_env = os.environ.get("ADH_LEGACY_ENV", "")
    if not legacy_env:
        repo_root = _Path(__file__).resolve().parents[2]
        legacy_dir = repo_root / ("local" + "_agentic" + "_rag")
        legacy_env = str(legacy_dir / ".env")
    if _Path(legacy_env).exists():
        try:
            from dotenv import load_dotenv

            load_dotenv(legacy_env, override=False)
        except ImportError:
            pass
    os.environ.setdefault("ADH_MODEL_BACKEND", "openai-compatible")
    for adh_key, az_key in (
        ("ADH_MODEL_BASE_URL", "AZURE_OPENAI_ENDPOINT"),
        ("ADH_MODEL_API_KEY", "AZURE_OPENAI_API_KEY"),
        ("ADH_MODEL_DEPLOYMENT", "AZURE_OPENAI_CHAT_DEPLOYMENT"),
        ("ADH_MODEL_API_VERSION", "AZURE_OPENAI_API_VERSION"),
    ):
        if os.environ.get(az_key) and not os.environ.get(adh_key):
            os.environ[adh_key] = os.environ[az_key]


def model_status() -> dict:
    """Redacted model configuration for run metadata."""
    return {
        "backend": _first_env("ADH_MODEL_BACKEND", "azure") or "azure",
        "base_url": _first_env("ADH_MODEL_BASE_URL", "AZURE_OPENAI_ENDPOINT"),
        "deployment": _first_env(
            "ADH_MODEL_DEPLOYMENT", "AZURE_OPENAI_CHAT_DEPLOYMENT"
        ),
        "api_version": _first_env("ADH_MODEL_API_VERSION", "AZURE_OPENAI_API_VERSION"),
        "has_api_key": bool(_first_env("ADH_MODEL_API_KEY", "AZURE_OPENAI_API_KEY")),
        "di_endpoint": _first_env("ADH_DI_ENDPOINT"),
        "has_di_key": bool(_first_env("ADH_DI_API_KEY")),
    }
