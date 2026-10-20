"""Schema gate: output satisfies the requested structure and types."""

from __future__ import annotations

from typing import Any, Dict, List

from ..core.schemas import GateResult, RecoveryTask, make_id
from .base import Gate, VerificationContext

_TYPE_NAMES = {"number": (int, float), "string": (str,), "boolean": (bool,)}


class SchemaGate:
    name = "schema"

    def evaluate(self, ctx: VerificationContext) -> GateResult:
        schema = ctx.goal.output_schema or {}
        required = schema.get("required", [])
        types: Dict[str, str] = schema.get("types", {})
        failures: List[str] = []
        recovery: List[RecoveryTask] = []
        if not ctx.claims and not required:
            failures.append("no claims extracted and no required schema given")
        for req in required:
            subject = str(req.get("subject", "*"))
            attribute = str(req.get("attribute", ""))
            matched = [
                c
                for c in ctx.claims
                if (subject == "*" or c.subject.strip().lower() == subject.lower())
                and c.attribute.strip().lower() == attribute.lower()
            ]
            if not matched:
                failures.append(f"missing required output {subject}/{attribute}")
                recovery.append(
                    RecoveryTask(
                        task_id=make_id("rec"),
                        claim_ids=[],
                        reason=f"required field {subject}/{attribute} has no claim",
                        suggested_actions=[
                            "search_subject_candidates",
                            "read_candidate_pages",
                            "inspect_tables",
                        ],
                    )
                )
        for claim in ctx.claims:
            expected = types.get(claim.attribute)
            if expected in _TYPE_NAMES and not isinstance(
                claim.value, _TYPE_NAMES[expected]
            ):
                if expected == "number" and isinstance(claim.value, bool):
                    failures.append(f"{claim.claim_id}: boolean is not a valid number")
                elif not (
                    expected == "number" and isinstance(claim.value, (int, float))
                ):
                    failures.append(
                        f"{claim.claim_id}: expected {expected}, got "
                        f"{type(claim.value).__name__}"
                    )
                recovery.append(
                    RecoveryTask(
                        task_id=make_id("rec"),
                        claim_ids=[claim.claim_id],
                        reason=f"value type mismatch (expected {expected})",
                        suggested_actions=["inspect_original_region"],
                    )
                )
        details: Dict[str, Any] = {"required": len(required), "claims": len(ctx.claims)}
        return GateResult(
            gate=self.name,
            passed=not failures,
            failures=failures,
            recovery=recovery,
            details=details,
        )
