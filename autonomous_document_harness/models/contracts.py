"""ExtractionContracts (§REAL03-1/2): typed requested-output fields.

A contract binds one requested answer (sheet + field/question) to the
semantics an observation must satisfy to count toward completion. General
infrastructure: no document or person assumptions anywhere here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from ..models.json_utils import parse_json_response  # noqa: F401 (re-export)

ANSWER_TYPES = frozenset(
    {
        "scalar",
        "boolean",
        "numeric",
        "narrative",
        "list",
        "record",
        "collection",
        "table",
    }
)
TERMINAL_STATES = frozenset({"COMPLETE", "NOT_FOUND", "BLOCKED"})
OPEN_STATES = frozenset({"UNSEEN", "DISCOVERING", "PARTIAL", "VERIFIED_PARTIAL"})

CONTRACT_PROMPT = """\
Compile the requested output field into an extraction contract. Reply with a
JSON object only:
{"target_entity": "<the thing the answer describes>",
 "entity_qualifier": "<scope such as a system, or empty>",
 "target_attribute": "<the property asked for>",
 "answer_type": "scalar|boolean|numeric|narrative|list|record|collection|table",
 "cardinality": "scalar|collection",
 "constraints": ["<exclusions/qualifiers from the request>"]}
Rules: describe the REQUESTED semantics, never any document's content. A whole
schedule/table of records is cardinality collection. Yes/No questions are
boolean. Counts are numeric. Free-form requirements are narrative.
Sheet: {sheet}
Columns: {columns}
Field question: {question}
"""


@dataclass
class ExtractionContract:
    contract_id: str
    sheet: str
    field: str
    question: str
    context: str = ""
    target_entity: str = ""
    entity_qualifier: str = ""
    target_attribute: str = ""
    answer_type: str = "narrative"
    cardinality: str = "collection"
    constraints: List[str] = field(default_factory=list)
    exclusions: List[str] = field(default_factory=list)
    evidence_requirement: str = "source evidence required"
    completion_policy: str = "saturation"
    state: str = "UNSEEN"
    # Runtime ledger (§8), never part of the compiled spec.
    candidate_regions: List[Dict[str, Any]] = field(default_factory=list)
    inspected_regions: List[str] = field(default_factory=list)
    candidate_answers: List[str] = field(default_factory=list)
    rejected_observations: int = 0
    ambiguities: List[str] = field(default_factory=list)
    strategies_attempted: List[str] = field(default_factory=list)
    answers_this_round: int = 0
    quiet_rounds: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "contract_id": self.contract_id,
            "sheet": self.sheet,
            "field": self.field,
            "question": self.question,
            "context": self.context,
            "target_entity": self.target_entity,
            "entity_qualifier": self.entity_qualifier,
            "target_attribute": self.target_attribute,
            "answer_type": self.answer_type,
            "cardinality": self.cardinality,
            "constraints": list(self.constraints),
            "exclusions": list(self.exclusions),
            "evidence_requirement": self.evidence_requirement,
            "completion_policy": self.completion_policy,
            "state": self.state,
        }

    def debug_trace(self) -> str:
        lines = [
            f"Contract: {self.field} / {self.context or self.sheet}",
            f"  entity={self.target_entity!r} attribute={self.target_attribute!r} "
            f"type={self.answer_type} cardinality={self.cardinality}",
            f"  candidates={len(self.candidate_regions)} "
            f"inspected={len(self.inspected_regions)} "
            f"rejected={self.rejected_observations}",
            f"  answers={len(self.candidate_answers)} verified-state={self.state}",
            f"  strategies={sorted(set(self.strategies_attempted))}",
        ]
        if self.ambiguities:
            lines.append(f"  ambiguity: {self.ambiguities[-1][:160]}")
        return "\n".join(lines)


def validate_contracts(contracts: List[ExtractionContract]) -> List[str]:
    """Pre-investigation validation. Empty list = valid batch."""
    errors = []
    if not contracts:
        errors.append("no contracts compiled")
    for contract in contracts:
        if not contract.target_entity:
            errors.append(f"{contract.contract_id}: empty target_entity")
        if not contract.target_attribute:
            errors.append(f"{contract.contract_id}: empty target_attribute")
        if contract.answer_type not in ANSWER_TYPES:
            errors.append(
                f"{contract.contract_id}: bad answer_type {contract.answer_type!r}"
            )
        if contract.cardinality not in ("scalar", "collection"):
            errors.append(f"{contract.contract_id}: bad cardinality")
    return errors


def _heuristic_contract(
    contract_id: str, sheet: str, field: str, entity_columns: List[str]
) -> ExtractionContract:
    """Deterministic fallback from header vocabulary (no model, no domain)."""
    lowered = field.lower()
    if any(word in lowered for word in ("count", "number", "qty", "quantity", "total")):
        answer_type = "numeric"
    elif any(
        word in lowered
        for word in ("yes", "no", "applicable", "required", "specified", "requirement")
    ):
        answer_type = "boolean"
    else:
        answer_type = "narrative"
    entity_bits = [
        c for c in entity_columns if c.lower() not in ("count", "value", "expected")
    ]
    return ExtractionContract(
        contract_id=contract_id,
        sheet=sheet,
        field=field,
        question=f"{sheet}: {field} by {' / '.join(entity_bits) or 'record'}",
        context=sheet,
        target_entity=" / ".join(entity_bits) or "record",
        target_attribute=field,
        answer_type=answer_type,
        cardinality="collection",
    )


def compile_contracts(
    schema_fields: List[Dict[str, Any]], gateway=None
) -> List[ExtractionContract]:
    """Compile requested fields into validated contracts.

    schema_fields: [{sheet, field, entity_columns}] from a SANITIZED schema
    (headers only — never GT values). Model-inferred when available, heuristic
    otherwise. Raises ValueError when validation fails.
    """
    contracts = []
    for index, entry in enumerate(schema_fields):
        sheet = str(entry.get("sheet", ""))
        field_name = str(entry.get("field", ""))
        entity_columns = list(entry.get("entity_columns", []))
        contract = None
        if gateway is not None:
            try:
                from ..models.schemas import ModelRequest

                response = gateway.complete(
                    ModelRequest(
                        kind="chat",
                        max_tokens=500,
                        response_format="json_object",
                        prompt=CONTRACT_PROMPT.format(
                            sheet=sheet,
                            columns=", ".join(entity_columns + [field_name]),
                            question=f"{sheet}: {field_name}",
                        ),
                    )
                )
                payload, _ = parse_json_response(response.text)
                if isinstance(payload, dict) and payload.get("target_entity"):
                    contract = ExtractionContract(
                        contract_id=f"contract_{index + 1:03d}",
                        sheet=sheet,
                        field=field_name,
                        question=f"{sheet}: {field_name}",
                        context=sheet,
                        target_entity=str(payload.get("target_entity", "")),
                        entity_qualifier=str(payload.get("entity_qualifier", "")),
                        target_attribute=str(
                            payload.get("target_attribute", "") or field_name
                        ),
                        answer_type=str(payload.get("answer_type", "narrative")),
                        cardinality=str(payload.get("cardinality", "collection")),
                        constraints=[str(c) for c in payload.get("constraints", [])],
                    )
            except Exception:
                contract = None
        if contract is None:
            contract = _heuristic_contract(
                f"contract_{index + 1:03d}", sheet, field_name, entity_columns
            )
        contracts.append(contract)
    errors = validate_contracts(contracts)
    if errors:
        raise ValueError(f"invalid contracts: {errors}")
    return contracts


class ContractIndex:
    """Candidate contract retrieval by token overlap (§5)."""

    def __init__(self, contracts: List[ExtractionContract]) -> None:
        self.contracts = list(contracts)

    @staticmethod
    def _terms(contract: ExtractionContract) -> set:
        from ..workspace.indexing import tokenize

        terms = set()
        for text in (
            contract.target_entity,
            contract.entity_qualifier,
            contract.target_attribute,
            contract.sheet,
            contract.field,
        ):
            terms.update(tokenize(str(text)))
        return terms

    def find_candidates(
        self, subject: str, attribute: str, top_k: int = 5
    ) -> List[ExtractionContract]:
        from ..workspace.indexing import tokenize

        obs_tokens = set(tokenize(f"{subject} {attribute}"))
        scored = []
        for contract in self.contracts:
            overlap = len(obs_tokens & self._terms(contract))
            if overlap:
                scored.append((overlap, contract))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [contract for _, contract in scored[:top_k]]
