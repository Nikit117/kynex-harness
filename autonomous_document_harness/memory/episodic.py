"""Episodic strategy memory: HOW past investigations went — never WHAT docs say.

Historical strategies may influence tool choice. They are type-separated from
evidence and can never support a claim (enforced: records carry no snippets).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class StrategyRecord:
    strategy: str
    doc_signature: str  # e.g. "pdf:selectable:tables=3" — characteristics only
    tools_used: List[str] = field(default_factory=list)
    success: bool = False
    cost: float = 0.0
    quality: float = 0.0
    note: str = ""


@dataclass
class EpisodicMemory:
    records: List[StrategyRecord] = field(default_factory=list)

    def record(
        self,
        strategy: str,
        doc_signature: str,
        tools_used: List[str],
        success: bool,
        cost: float = 0.0,
        quality: float = 0.0,
        note: str = "",
    ) -> StrategyRecord:
        rec = StrategyRecord(
            strategy, doc_signature, list(tools_used), success, cost, quality, note
        )
        self.records.append(rec)
        return rec

    def suggest(self, doc_signature: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Return strategy hints (tool sequences), never factual content."""
        scored = []
        for rec in self.records:
            score = (1.0 if rec.success else -0.5) + rec.quality
            if rec.doc_signature.split(":")[0] == doc_signature.split(":")[0]:
                score += 0.5
            scored.append((score, rec))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [
            {
                "strategy": r.strategy,
                "tools": list(r.tools_used),
                "success": r.success,
                "note": r.note,
            }
            for _, r in scored[:top_k]
            if _ > 0
        ]

    def save(self, path: str) -> str:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = [
            {
                "strategy": r.strategy,
                "doc_signature": r.doc_signature,
                "tools_used": r.tools_used,
                "success": r.success,
                "cost": r.cost,
                "quality": r.quality,
                "note": r.note,
            }
            for r in self.records
        ]
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str) -> "EpisodicMemory":
        mem = cls()
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        for item in raw:
            mem.records.append(StrategyRecord(**item))
        return mem
