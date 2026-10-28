"""ObjectiveSpec compiler: arbitrary instructions -> validated typed spec.

Model-driven when a provider is configured (JSON contract), deterministic
heuristic fallback otherwise. The spec — not regex miners — drives the loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from ..models.json_utils import parse_json_response  # noqa: F401 (re-export)

OBJECTIVE_CONTRACT = """\
Compile the user's document-investigation objective into a JSON object with
exactly these keys:
{
  "entities": ["entity kinds to find, e.g. control system, equipment tag"],
  "attributes": ["requested attributes, e.g. AI count, description, deadline"],
  "relationships": ["cross-entity relations, e.g. system contains area"],
  "constraints": ["hard requirements, e.g. every value must have evidence"],
  "exclusions": ["what to skip, e.g. spare, totals"],
  "grouping": ["how to group output, e.g. by system then area"],
  "output_schema": {"required": [{"subject": "*", "attribute": "AI"}], "types": {}},
  "evidence_required": true,
  "completeness": {"sweep_channels": ["lexical", "tables"], "min_verified_ratio": 0.8},
  "ambiguity_policy": "mark-unresolved | best-effort | ask"
}
Rules: entities/attributes are generic kinds, never document-specific values.
exclusions come ONLY from the user's words. ambiguity_policy defaults to
mark-unresolved. Reply with the JSON object only.
Objective: """


@dataclass
class ObjectiveSpec:
    entities: List[str] = field(default_factory=list)
    attributes: List[str] = field(default_factory=list)
    relationships: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    exclusions: List[str] = field(default_factory=list)
    grouping: List[str] = field(default_factory=list)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    evidence_required: bool = True
    completeness: Dict[str, Any] = field(default_factory=dict)
    ambiguity_policy: str = "mark-unresolved"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entities": list(self.entities),
            "attributes": list(self.attributes),
            "relationships": list(self.relationships),
            "constraints": list(self.constraints),
            "exclusions": list(self.exclusions),
            "grouping": list(self.grouping),
            "output_schema": dict(self.output_schema),
            "evidence_required": self.evidence_required,
            "completeness": dict(self.completeness),
            "ambiguity_policy": self.ambiguity_policy,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ObjectiveSpec":
        def strings(key: str) -> List[str]:
            return [str(v) for v in (payload.get(key) or []) if str(v).strip()]

        schema = payload.get("output_schema") or {}
        if not isinstance(schema, dict):
            schema = {}
        completeness = payload.get("completeness") or {}
        if not isinstance(completeness, dict):
            completeness = {}
        policy = str(payload.get("ambiguity_policy", "mark-unresolved"))
        if policy not in ("mark-unresolved", "best-effort", "ask"):
            policy = "mark-unresolved"
        return cls(
            entities=strings("entities"),
            attributes=strings("attributes"),
            relationships=strings("relationships"),
            constraints=strings("constraints"),
            exclusions=strings("exclusions"),
            grouping=strings("grouping"),
            output_schema=schema,
            evidence_required=bool(payload.get("evidence_required", True)),
            completeness=completeness,
            ambiguity_policy=policy,
        )


def validate_spec(spec: ObjectiveSpec) -> List[str]:
    """Spec validation BEFORE investigation begins. Empty list = valid."""
    errors = []
    if not spec.attributes and not spec.entities:
        errors.append("spec has neither entities nor attributes")
    if spec.evidence_required not in (True, False):
        errors.append("evidence_required must be boolean")
    schema = spec.output_schema or {}
    if schema and not isinstance(schema.get("required", []), list):
        errors.append("output_schema.required must be a list")
    return errors


def heuristic_compile(prompt: str) -> ObjectiveSpec:
    """Deterministic fallback: reuse the heuristic goal parser's output."""
    from ..models import heuristic as heuristic_mod

    goal = heuristic_mod.parse_goal(prompt)
    exclusions = sorted({e for n in goal.needs for e in n.exclusions})
    attributes = sorted({n.attribute for n in goal.needs})
    entities = sorted({n.subject for n in goal.needs if n.subject != "*"})
    return ObjectiveSpec(
        entities=entities or ["record"],
        attributes=attributes,
        constraints=list(goal.constraints),
        exclusions=exclusions,
        output_schema=dict(goal.output_schema),
        evidence_required=True,
        completeness={"min_verified_ratio": 0.8},
        ambiguity_policy="mark-unresolved",
    )


def compile_objective(prompt: str, gateway=None) -> ObjectiveSpec:
    """Model-driven when gateway supports completions, else heuristic."""
    if gateway is not None:
        try:
            from ..models.schemas import ModelRequest

            response = gateway.complete(
                ModelRequest(
                    kind="parse_goal",
                    prompt=OBJECTIVE_CONTRACT + prompt,
                    max_tokens=1200,
                    response_format="json_object",
                )
            )
            payload, _ = parse_json_response(response.text)
            if isinstance(payload, dict):
                spec = ObjectiveSpec.from_dict(payload)
                if not validate_spec(spec):
                    return spec
        except Exception:
            pass  # model path is best-effort; heuristic fallback is guaranteed
    return heuristic_compile(prompt)
