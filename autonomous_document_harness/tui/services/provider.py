"""Persistent provider configuration for KYNEX.

Credentials are stored in ``<data_dir>/provider.env`` and loaded at startup
with ``override=False`` so explicit process environment variables always win.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict


def _data_dir() -> Path:
    """Return the KYNEX data directory (same logic as sessions.py)."""
    try:
        from platformdirs import user_data_dir
        root = Path(user_data_dir("kynex", appauthor=False))
    except ImportError:
        root = Path.home() / ".kynex"
    root.mkdir(parents=True, exist_ok=True)
    return root


def provider_env_path() -> Path:
    return _data_dir() / "provider.env"


def load_provider_env() -> bool:
    """Load saved provider credentials into the process environment.

    Returns True if a provider.env file was found and loaded.
    Uses override=False so explicit env vars always take precedence.
    """
    path = provider_env_path()
    if not path.exists():
        return False
    try:
        from dotenv import load_dotenv
        load_dotenv(path, override=False)
        return True
    except ImportError:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip("'\"")
                os.environ.setdefault(key, value)
        return True


def save_provider_env(config: Dict[str, str]) -> Path:
    """Write provider credentials to the persistent env file.

    Also sets them in the current process environment immediately.
    Returns the path to the saved file.
    """
    path = provider_env_path()
    lines = ["# KYNEX provider configuration", "# Saved by /provider command", ""]
    for key, value in config.items():
        if value:
            lines.append(f"{key}={value}")
            os.environ[key] = value
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def read_provider_config() -> Dict[str, str]:
    """Read saved provider config with secrets redacted."""
    path = provider_env_path()
    if not path.exists():
        return {}
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("'\"")
            if "KEY" in key.upper() or "SECRET" in key.upper() or "TOKEN" in key.upper():
                result[key] = value[:4] + "..." + value[-4:] if len(value) > 8 else "****"
            else:
                result[key] = value
    return result


def clear_provider_env() -> bool:
    """Delete the saved provider configuration. Returns True if file existed."""
    path = provider_env_path()
    if path.exists():
        path.unlink()
        return True
    return False


def provider_status() -> Dict[str, str]:
    """Return a summary of the current provider configuration for display."""
    saved = read_provider_config()

    backend = os.environ.get("ADH_MODEL_BACKEND", "")
    base_url = os.environ.get("ADH_MODEL_BASE_URL", "") or os.environ.get("AZURE_OPENAI_ENDPOINT", "")
    has_key = bool(os.environ.get("ADH_MODEL_API_KEY") or os.environ.get("AZURE_OPENAI_API_KEY"))
    deployment = os.environ.get("ADH_MODEL_DEPLOYMENT", "") or os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT", "")
    di_endpoint = os.environ.get("ADH_DI_ENDPOINT", "")
    has_di_key = bool(os.environ.get("ADH_DI_API_KEY"))

    return {
        "backend": backend or "not configured",
        "base_url": base_url or "not set",
        "has_api_key": "yes" if has_key else "no",
        "deployment": deployment or "not set",
        "di_endpoint": di_endpoint or "not set",
        "has_di_key": "yes" if has_di_key else "no",
        "saved_config": bool(saved),
        "config_path": str(provider_env_path()),
    }
