"""Bounded parallel executor + doc-affinity scheduler. Safe fan-out only."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple

from ..tools.computer import DocumentComputer
from ..tools.registry import Observation
from .planner import Action


class ToolExecutor:
    def __init__(self, computer: DocumentComputer, max_workers: int = 4) -> None:
        self.computer = computer
        self.max_workers = max(1, max_workers)

    def execute(
        self, actions: List[Action], timeout: float = 60.0
    ) -> List[Tuple[Action, Observation]]:
        if not actions:
            return []
        results: List[Tuple[Action, Observation]] = []
        with ThreadPoolExecutor(
            max_workers=min(self.max_workers, len(actions))
        ) as pool:
            futures = {
                pool.submit(
                    self.computer.registry.call, action.tool, **action.args
                ): action
                for action in actions
            }
            for future, action in futures.items():
                try:
                    observation = future.result(timeout=timeout)
                except Exception as exc:
                    observation = Observation(
                        tool=action.tool,
                        args=dict(action.args),
                        status="error",
                        message=f"executor: {type(exc).__name__}: {exc}",
                    )
                results.append((action, observation))
        return results


def waves(actions: List[Action], width: int = 4) -> List[List[Action]]:
    """Partition into waves with distinct doc affinity (no two same-doc rivals)."""
    pending = list(actions)
    out: List[List[Action]] = []
    while pending:
        wave, used_docs, rest = [], set(), []
        for action in pending:
            doc = str(action.args.get("doc_id", action.action_id))
            if doc in used_docs and len(wave) >= 1:
                rest.append(action)
                continue
            if len(wave) >= max(width, 1):
                rest.append(action)
                continue
            wave.append(action)
            used_docs.add(doc)
        out.append(wave)
        pending = rest or pending[len(wave) :]
        if not wave:
            break
    return out


def doc_affinity(action: Action) -> str:
    return str(action.args.get("doc_id", ""))


__all__ = ["ToolExecutor", "waves", "doc_affinity"]
