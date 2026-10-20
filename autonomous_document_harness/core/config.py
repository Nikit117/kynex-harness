"""Runtime configuration: YAML file + ADH_-prefixed env overrides.

Secrets (API keys) come ONLY from the environment — never from YAML, never
hardcoded. See ARCHITECTURE_DECISIONS.md §3.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

from .schemas import BudgetLimits


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


@dataclass
class ModelConfig:
    backend: str = "heuristic"  # heuristic | openai-compatible
    base_url: str = ""
    chat_deployment: str = ""
    embedding_deployment: str = ""
    api_version: str = ""
    di_endpoint: str = ""

    @property
    def api_key(self) -> str:
        return os.environ.get("ADH_MODEL_API_KEY", "") or os.environ.get(
            "AZURE_OPENAI_API_KEY", ""
        )

    @property
    def di_key(self) -> str:
        return os.environ.get("ADH_DI_API_KEY", "")

    def redacted(self) -> Dict[str, str]:
        """Safe for traces/logs: never contains secret values."""
        return {
            "backend": self.backend,
            "base_url": self.base_url,
            "chat_deployment": self.chat_deployment,
            "api_version": self.api_version,
            "di_endpoint": self.di_endpoint,
            "has_api_key": str(bool(self.api_key)),
            "has_di_key": str(bool(self.di_key)),
        }


@dataclass
class RuntimeConfig:
    budgets: BudgetLimits = field(default_factory=BudgetLimits)
    max_parallel_tools: int = 4
    no_progress_rounds_stop: int = 3
    model: ModelConfig = field(default_factory=ModelConfig)
    redundancy_penalty: float = 0.35
    failure_penalty: float = 0.5
    visual_cost_weight: float = 3.0
    model_cost_weight: float = 2.0
    require_evidence: bool = True
    entailment_min_overlap: float = 0.34
    sufficiency_min_verified_ratio: float = 0.8
    output_formats: Any = field(default_factory=lambda: ["json", "md", "csv"])
    domain_pack: str = ""
    di_confidence_floor: float = 0.4
    vision_image_max_side: int = 1568

    @classmethod
    def from_env(cls, yaml_path: str = "") -> "RuntimeConfig":
        data: Dict[str, Any] = {}
        if yaml_path and Path(yaml_path).exists():
            try:
                import yaml  # type: ignore

                data = yaml.safe_load(Path(yaml_path).read_text()) or {}
            except ImportError:
                data = _minimal_yaml(Path(yaml_path).read_text())
        inv = data.get("investigation", {})
        pol = data.get("policy", {})
        mod = data.get("models", {})
        ver = data.get("verification", {})
        out = data.get("output", {})
        budgets = BudgetLimits(
            max_iterations=_env_int(
                "ADH_MAX_ITERATIONS", int(inv.get("max_iterations", 24))
            ),
            max_tool_calls=_env_int(
                "ADH_MAX_TOOL_CALLS", int(inv.get("max_tool_calls", 200))
            ),
            max_model_calls=_env_int(
                "ADH_MAX_MODEL_CALLS", int(inv.get("max_model_calls", 60))
            ),
            max_visual_calls=_env_int(
                "ADH_MAX_VISUAL_CALLS", int(inv.get("max_visual_calls", 40))
            ),
            token_budget=_env_int(
                "ADH_TOKEN_BUDGET", int(inv.get("token_budget", 200000))
            ),
            timeout_seconds=_env_float(
                "ADH_TIMEOUT_SECONDS", float(inv.get("timeout_seconds", 1800.0))
            ),
        )
        return cls(
            budgets=budgets,
            max_parallel_tools=_env_int(
                "ADH_MAX_PARALLEL", int(inv.get("max_parallel_tools", 4))
            ),
            model=ModelConfig(
                backend=os.environ.get(
                    "ADH_MODEL_BACKEND", str(mod.get("backend", "heuristic"))
                ),
                base_url=os.environ.get("ADH_MODEL_BASE_URL", "")
                or os.environ.get("AZURE_OPENAI_ENDPOINT", "")
                or str(mod.get("base_url", "")),
                chat_deployment=os.environ.get("ADH_MODEL_DEPLOYMENT", "")
                or os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT", "")
                or str(mod.get("chat_deployment", "")),
                embedding_deployment=str(mod.get("embedding_deployment", "")),
                api_version=os.environ.get("ADH_MODEL_API_VERSION", "")
                or os.environ.get("AZURE_OPENAI_API_VERSION", "")
                or str(mod.get("api_version", "") or "2024-10-21"),
                di_endpoint=os.environ.get("ADH_DI_ENDPOINT", "")
                or str(mod.get("di_endpoint", "")),
            ),
            redundancy_penalty=float(pol.get("redundancy_penalty", 0.35)),
            failure_penalty=float(pol.get("failure_penalty", 0.5)),
            visual_cost_weight=float(pol.get("visual_cost_weight", 3.0)),
            model_cost_weight=float(pol.get("model_cost_weight", 2.0)),
            require_evidence=bool(ver.get("require_evidence", True)),
            entailment_min_overlap=float(ver.get("entailment_min_overlap", 0.34)),
            sufficiency_min_verified_ratio=float(
                ver.get("sufficiency_min_verified_ratio", 0.8)
            ),
            output_formats=list(out.get("formats", ["json", "md", "csv"])),
            domain_pack=os.environ.get("ADH_DOMAIN_PACK", ""),
            di_confidence_floor=float(mod.get("di_confidence_floor", 0.4)),
            vision_image_max_side=int(mod.get("vision_image_max_side", 1568)),
        )


def _minimal_yaml(text: str) -> Dict[str, Any]:
    """Tiny subset-YAML fallback (nested mapping + scalar/list values)."""
    root: Dict[str, Any] = {}
    stack: list[tuple[int, Dict[str, Any]]] = [(-1, root)]
    for raw in text.splitlines():
        if not raw.strip() or raw.strip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        key, _, val = raw.strip().partition(":")
        key, val = key.strip(), val.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if val == "":
            child: Dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        elif val.startswith("[") and val.endswith("]"):
            parent[key] = [
                v.strip().strip("'\"") for v in val[1:-1].split(",") if v.strip()
            ]
        elif val.lower() in ("true", "false"):
            parent[key] = val.lower() == "true"
        else:
            try:
                parent[key] = int(val)
            except ValueError:
                try:
                    parent[key] = float(val)
                except ValueError:
                    parent[key] = val.strip("'\"")
    return root
