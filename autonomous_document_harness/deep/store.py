"""Thread-safe run store for the Deep Agents runtime.

Holds extraction contracts, admitted answers, coverage state, budgets and
the observation ledger. LangGraph checkpoints persist this state so a run can
resume; the store itself never holds credentials.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .contracts import ExtractionContract


@dataclass
class AnswerRecord:
    answer_id: str
    contract_id: str
    subject: str
    attribute: str
    value: Any
    units: Optional[str]
    method: str
    document_id: str
    page: Optional[int]
    cell: Optional[str]
    snippet: str
    admission: str
    verification: str = "unverified"
    confidence: float = 0.0

    def to_claim(self, filename: str = "") -> Dict[str, Any]:
        snippet_hash = hashlib.sha256((self.snippet or "").encode("utf-8")).hexdigest()[
            :16
        ]
        return {
            "claim_id": self.answer_id,
            "subject": self.subject,
            "attribute": self.attribute,
            "value": self.value,
            "units": self.units,
            "verification": self.verification,
            "field_ref": {"sheet": self.contract_id, "columns": [self.attribute]},
            "evidence": [
                {
                    "evidence_id": f"ev_{self.answer_id}",
                    "filename": filename or self.document_id,
                    "document_id": self.document_id,
                    "page": self.page,
                    "cell": self.cell,
                    "method": self.method,
                    "confidence": self.confidence,
                    "snippet": self.snippet[:500],
                    "snippet_hash": snippet_hash,
                }
            ],
        }


@dataclass
class CoverageState:
    total_documents: int = 0
    total_pages: int = 0
    indexed_pages: int = 0
    relevant_candidates: int = 0
    inspected_candidates: int = 0
    relevant_tables: int = 0
    scanned_candidates: int = 0
    di_processed_pages: int = 0
    vision_inspected_regions: int = 0
    unexplored_high_value_regions: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_documents": self.total_documents,
            "total_pages": self.total_pages,
            "indexed_pages": self.indexed_pages,
            "relevant_candidates": self.relevant_candidates,
            "inspected_candidates": self.inspected_candidates,
            "relevant_tables": self.relevant_tables,
            "scanned_candidates": self.scanned_candidates,
            "di_processed_pages": self.di_processed_pages,
            "vision_inspected_regions": self.vision_inspected_regions,
            "unexplored_high_value_regions": self.unexplored_high_value_regions,
        }


@dataclass
class BudgetState:
    """Run budgets, compatible with the DocumentComputer BudgetUsage protocol."""

    max_model_calls: int = 200
    max_tool_calls: int = 400
    max_visual_calls: int = 60
    max_di_calls: int = 100
    max_iterations: int = 61  # investigation rounds + resume rounds
    token_budget: int = 400_000
    timeout_seconds: float = 3600.0
    started_at: float = field(default_factory=time.time)

    model_calls: int = 0
    tool_calls: int = 0
    visual_calls: int = 0
    di_calls: int = 0
    tokens_used: int = 0

    def __post_init__(self) -> None:
        self._used: Dict[str, int] = {
            "iteration": 0,
            "tool": 0,
            "model": 0,
            "visual": 0,
            "token": 0,
        }

    # -- BudgetUsage protocol (DocumentComputer / ToolRegistry) -------------
    @property
    def limits(self):
        from ..core.schemas import BudgetLimits

        return BudgetLimits(
            max_iterations=self.max_iterations,
            max_tool_calls=self.max_tool_calls,
            max_model_calls=self.max_model_calls,
            max_visual_calls=self.max_visual_calls,
            token_budget=self.token_budget,
            timeout_seconds=self.timeout_seconds,
        )

    def _cap(self, kind: str) -> int:
        return {
            "iteration": self.max_iterations,
            "tool": self.max_tool_calls,
            "model": self.max_model_calls,
            "visual": self.max_visual_calls,
            "token": self.token_budget,
        }[kind]

    def consume(self, kind: str, amount: int = 1) -> bool:
        """Record usage; return False (never raise) when the cap is hit."""
        if kind not in self._used:
            raise ValueError(f"unknown budget kind {kind!r}")
        if self._used[kind] + amount > self._cap(kind):
            return False
        self._used[kind] += amount
        if kind == "tool":
            self.tool_calls += amount
        elif kind == "model":
            self.model_calls += amount
        elif kind == "visual":
            self.visual_calls += amount
        elif kind == "token":
            self.tokens_used += amount
        return True

    def exhausted(self) -> bool:
        return (
            self.model_calls >= self.max_model_calls
            or self.tool_calls >= self.max_tool_calls
            or self.visual_calls >= self.max_visual_calls
            or self.di_calls >= self.max_di_calls
            or self.tokens_used >= self.token_budget
            or (time.time() - self.started_at) >= self.timeout_seconds
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_model_calls": self.max_model_calls,
            "max_tool_calls": self.max_tool_calls,
            "max_visual_calls": self.max_visual_calls,
            "max_di_calls": self.max_di_calls,
            "max_iterations": self.max_iterations,
            "token_budget": self.token_budget,
            "timeout_seconds": self.timeout_seconds,
            "model_calls": self.model_calls,
            "tool_calls": self.tool_calls,
            "visual_calls": self.visual_calls,
            "di_calls": self.di_calls,
            "tokens_used": self.tokens_used,
            "used": dict(self._used),
            "exhausted": [k for k in self._used if self._used[k] >= self._cap(k)],
            "elapsed_seconds": round(time.time() - self.started_at, 1),
        }


class RunStore:
    """Shared mutable state for one investigation run (thread-safe)."""

    def __init__(self, contracts: List[ExtractionContract]) -> None:
        self._lock = threading.RLock()
        self.contracts: Dict[str, ExtractionContract] = {
            c.contract_id: c for c in contracts
        }
        self.contract_state: Dict[str, Dict[str, Any]] = {
            c.contract_id: {
                "state": "UNSEEN",
                "candidate_regions": [],
                "inspected_regions": [],
                "candidate_answers": [],
                "rejected_observations": 0,
                "supporting_context": 0,
                "strategies_attempted": [],
                "queries": [],
                "answers": 0,
                "quiet_dives": 0,
            }
            for c in contracts
        }
        self.answers: List[AnswerRecord] = []
        self.coverage = CoverageState()
        self.budgets = BudgetState()
        self.contradictions: List[Dict[str, Any]] = []
        self.recovery_log: List[Dict[str, Any]] = []
        self.observations: List[Dict[str, Any]] = []

    # -- contracts ----------------------------------------------------------
    def get_contract(self, contract_id: str) -> Optional[ExtractionContract]:
        return self.contracts.get(contract_id)

    def unresolved_contracts(self) -> List[ExtractionContract]:
        terminal = {"COMPLETE", "NOT_FOUND", "BLOCKED"}
        return [
            c
            for c in self.contracts.values()
            if self.contract_state[c.contract_id]["state"] not in terminal
        ]

    def contract_summary(self) -> List[Dict[str, Any]]:
        out = []
        for c in self.contracts.values():
            st = self.contract_state[c.contract_id]
            out.append(
                {
                    "contract_id": c.contract_id,
                    "sheet": c.sheet,
                    "field": c.field,
                    "target_entity": c.target_entity,
                    "target_attribute": c.target_attribute,
                    "answer_type": c.answer_type,
                    "cardinality": c.cardinality,
                    "state": st["state"],
                    "answers": st["answers"],
                    "candidate_regions": len(st["candidate_regions"]),
                    "rejected": st["rejected_observations"],
                }
            )
        return out

    def set_contract_state(self, contract_id: str, state: str) -> None:
        with self._lock:
            if contract_id in self.contract_state:
                self.contract_state[contract_id]["state"] = state

    def record_strategy(self, contract_id: str, strategy: str) -> None:
        with self._lock:
            st = self.contract_state[contract_id]
            if strategy not in st["strategies_attempted"]:
                st["strategies_attempted"].append(strategy)

    def record_query(self, contract_id: str, query: str) -> None:
        with self._lock:
            self.contract_state[contract_id]["queries"].append(query)

    def record_candidate_region(self, contract_id: str, region: str) -> None:
        with self._lock:
            st = self.contract_state[contract_id]
            if region not in st["candidate_regions"]:
                st["candidate_regions"].append(region)

    def record_rejection(self, contract_id: str) -> None:
        with self._lock:
            self.contract_state[contract_id]["rejected_observations"] += 1

    # -- answers ------------------------------------------------------------
    def add_answer(self, answer: AnswerRecord) -> None:
        with self._lock:
            self.answers.append(answer)
            st = self.contract_state[answer.contract_id]
            st["answers"] += 1
            if answer.admission == "ANSWER" and answer.verification in (
                "verified",
                "uncertain",
            ):
                st["candidate_answers"].append(answer.answer_id)
                if answer.verification == "verified":
                    # Scalar contracts complete on one verified answer.
                    # Collection contracts NEVER auto-complete: one record is
                    # not a collection. The supervisor closes them via
                    # saturation (quiet dives) or terminalizes PARTIAL.
                    contract = self.contracts.get(answer.contract_id)
                    if contract is None or contract.cardinality == "scalar":
                        st["state"] = "COMPLETE"
                    elif st["state"] == "UNSEEN":
                        st["state"] = "DISCOVERING"
            self._detect_conflict(answer)

    def _detect_conflict(self, answer: AnswerRecord) -> None:
        """Flag incompatible values for the same contract + subject + attribute.

        Conflicts require matching scope (normalized subject/attribute); generic
        statements mentioning the same acronym never collide because admission
        already rejected them. Values are compared after normalization.
        """
        from ..eval.normalize import normalize_text, normalize_value

        def _norm(value: object) -> str:
            try:
                return str(normalize_value(value))
            except Exception:
                return normalize_text(value)

        for other in self.answers:
            if other.answer_id == answer.answer_id:
                continue
            if other.contract_id != answer.contract_id:
                continue
            if other.admission != "ANSWER" or answer.admission != "ANSWER":
                continue
            if normalize_text(other.subject) != normalize_text(answer.subject):
                continue
            if normalize_text(other.attribute) != normalize_text(answer.attribute):
                continue
            if _norm(other.value) == _norm(answer.value):
                continue
            conflict_id = f"conf_{other.answer_id}_{answer.answer_id}"
            if any(c.get("conflict_id") == conflict_id for c in self.contradictions):
                continue
            self.contradictions.append(
                {
                    "conflict_id": conflict_id,
                    "contract_id": answer.contract_id,
                    "subject": answer.subject,
                    "attribute": answer.attribute,
                    "claim_ids": [other.answer_id, answer.answer_id],
                    "values": [other.value, answer.value],
                    "status": "open",
                    "resolution": "",
                }
            )

    def answers_for(self, contract_id: str) -> List[AnswerRecord]:
        return [a for a in self.answers if a.contract_id == contract_id]

    # -- ledger -------------------------------------------------------------
    def record_observation(self, tool: str, args: Dict[str, Any], summary: str) -> None:
        with self._lock:
            self.observations.append(
                {
                    "tool": tool,
                    "args": args,
                    "summary": summary[:300],
                    "ts": time.time(),
                }
            )

    def record_recovery(self, contract_id: str, failure: str, action: str) -> None:
        with self._lock:
            self.recovery_log.append(
                {
                    "contract_id": contract_id,
                    "failure": failure,
                    "action": action,
                    "ts": time.time(),
                }
            )

    def snapshot(self) -> Dict[str, Any]:
        return {
            "contracts": self.contract_summary(),
            "answers": len(self.answers),
            "coverage": self.coverage.to_dict(),
            "budgets": self.budgets.to_dict(),
        }

    # -- persistence (checkpoints; never contains credentials) ---------------
    def to_ledger(self) -> Dict[str, Any]:
        import dataclasses

        return {
            "version": 1,
            "contracts": [c.to_dict() for c in self.contracts.values()],
            "contract_state": self.contract_state,
            "answers": [dataclasses.asdict(a) for a in self.answers],
            "coverage": self.coverage.to_dict(),
            "budgets": {
                "max_model_calls": self.budgets.max_model_calls,
                "max_tool_calls": self.budgets.max_tool_calls,
                "max_visual_calls": self.budgets.max_visual_calls,
                "max_di_calls": self.budgets.max_di_calls,
                "max_iterations": self.budgets.max_iterations,
                "token_budget": self.budgets.token_budget,
                "timeout_seconds": self.budgets.timeout_seconds,
                "model_calls": self.budgets.model_calls,
                "tool_calls": self.budgets.tool_calls,
                "visual_calls": self.budgets.visual_calls,
                "di_calls": self.budgets.di_calls,
                "tokens_used": self.budgets.tokens_used,
                "used": dict(self.budgets._used),
            },
            "contradictions": self.contradictions,
            "recovery_log": self.recovery_log,
            "observations": [
                {
                    "tool": o["tool"],
                    "args": {k: str(v)[:200] for k, v in o["args"].items()},
                    "summary": o["summary"][:300],
                }
                for o in self.observations[-500:]
            ],
        }

    def save_ledger(self, path: str) -> None:
        import json

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_ledger(), indent=2, default=str))

    @classmethod
    def from_ledger(cls, data: Dict[str, Any]) -> "RunStore":
        contracts = [ExtractionContract(**c) for c in data.get("contracts", [])]
        store = cls(contracts)
        store.contract_state.update(data.get("contract_state", {}))
        for entry in data.get("answers", []):
            try:
                store.answers.append(AnswerRecord(**entry))
            except TypeError:
                continue
        saved = data.get("budgets", {})
        for key in (
            "max_model_calls",
            "max_tool_calls",
            "max_visual_calls",
            "max_di_calls",
            "max_iterations",
            "token_budget",
            "timeout_seconds",
            "model_calls",
            "tool_calls",
            "visual_calls",
            "di_calls",
            "tokens_used",
        ):
            if key in saved:
                setattr(store.budgets, key, saved[key])
        if isinstance(saved.get("used"), dict):
            store.budgets._used.update(saved["used"])
        # Fresh clock on resume; counters carry the budget history.
        store.budgets.started_at = time.time()
        store.contradictions = data.get("contradictions", [])
        store.recovery_log = data.get("recovery_log", [])
        return store

    @classmethod
    def load_ledger(cls, path: str) -> "RunStore":
        import json

        return cls.from_ledger(json.loads(Path(path).read_text(encoding="utf-8")))
