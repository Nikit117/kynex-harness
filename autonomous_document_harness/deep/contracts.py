"""Extraction Contracts for the Deep Agents runtime (§9/§10).

Every requested output field becomes an explicit typed ExtractionContract.
The compiler is generic: it reads a sanitized request schema (headers only,
no ground-truth values) and produces contracts with entity columns, the
value column, answer type and cardinality. No CPQ-specific logic lives here.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from pydantic import BaseModel, Field

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

# Columns that are evaluation/bookkeeping annotations rather than requested
# answers. Detection is name-based and generic (works for any request schema).
_ANNOTATION_RE = re.compile(
    r"reason|verdict|gt\s+document|gt\s+page|gt\s+section|clause|nikit|albert|"
    r"extraction|comment|note|source|reference|basis",
    re.IGNORECASE,
)
_NUMERIC_FIELD_RE = re.compile(r"count|number|qty|quantity|total|sum", re.IGNORECASE)
_BOOLEAN_FIELD_RE = re.compile(
    r"yes|no|applicable|required|specified|included", re.IGNORECASE
)
# Value-column names that label the answer cell rather than describing the
# requested attribute (benchmark workbooks, export templates). Detection is
# name-based and generic (works for any request schema).
META_VALUE_RE = re.compile(
    r"ground.?truth|correct.?interpretation|expected.?semantic",
    re.IGNORECASE,
)


class ExtractionContract(BaseModel):
    """One requested output field, bound to the semantics an answer must satisfy."""

    contract_id: str
    sheet: str
    field: str
    question: str
    context: str = ""
    target_entity: str = ""
    entity_qualifier: str = ""
    target_attribute: str = ""
    answer_type: str = "narrative"
    cardinality: str = "scalar"
    constraints: List[str] = Field(default_factory=list)
    exclusions: List[str] = Field(default_factory=list)
    evidence_requirement: str = "source evidence required"
    completion_policy: str = "saturation"
    # Schema-derived alignment hints (which columns identify the record vs hold
    # the answer). Generic: derived from the sanitized request schema only.
    entity_columns: List[str] = Field(default_factory=list)
    value_column: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


def is_annotation_column(header: str) -> bool:
    """True for bookkeeping/annotation columns, not requested answers."""
    return bool(_ANNOTATION_RE.search(header or ""))


def compile_contracts(
    schema_path: str, sheet_prefix: str = ""
) -> List[ExtractionContract]:
    """Compile a headers-only request schema workbook into contracts.

    For each sheet: annotation columns are dropped; the last remaining column
    is the requested value; the remaining columns identify the record (entity).
    Answer type is inferred from the value-column name. No GT values are read.

    sheet_prefix: when given, only sheets whose names start with the prefix
    (case-insensitive) are compiled. This scopes a multi-sheet schema workbook
    to the single sheet under investigation — the real-world setting where the
    user supplies the request schema for THEIR corpus. Auditable harness
    configuration, not extraction tuning: no values are involved.
    """
    import openpyxl

    contracts: List[ExtractionContract] = []
    workbook = openpyxl.load_workbook(schema_path, read_only=True, data_only=True)
    try:
        for name in workbook.sheetnames:
            if sheet_prefix and not name.lower().startswith(sheet_prefix.lower()):
                continue
            sheet = workbook[name]
            headers: List[str] = []
            for row in sheet.iter_rows(values_only=True):
                headers = [
                    str(c).strip() for c in row if c is not None and str(c).strip()
                ]
                break
            if not headers:
                continue
            content = [h for h in headers if not is_annotation_column(h)]
            if not content:
                continue
            value_column = content[-1]
            entity_columns = content[:-1] or ["record"]
            # Meta-labeled answer columns ("Ground Truth", "Correct
            # Interpretation", ...) name the cell, not the requested attribute:
            # no document will ever state them. The record key anchors
            # admission instead; the attribute is left empty on purpose.
            meta_labeled = bool(META_VALUE_RE.search(value_column))
            if _NUMERIC_FIELD_RE.search(value_column):
                answer_type = "numeric"
            elif _BOOLEAN_FIELD_RE.search(value_column):
                answer_type = "boolean"
            else:
                answer_type = "narrative"
            cardinality = "collection" if len(entity_columns) >= 1 else "scalar"
            contracts.append(
                ExtractionContract(
                    contract_id=f"contract_{len(contracts) + 1:03d}",
                    sheet=name,
                    field=value_column,
                    question=f"{name}: {value_column}",
                    context=name,
                    target_entity=" / ".join(entity_columns),
                    target_attribute="" if meta_labeled else value_column,
                    answer_type=answer_type,
                    cardinality=cardinality,
                    entity_columns=entity_columns,
                    value_column=value_column,
                    constraints=(
                        ["answer column is meta-labeled; record key anchors admission"]
                        if meta_labeled
                        else []
                    ),
                )
            )
    finally:
        workbook.close()
    if not contracts:
        raise ValueError(f"no contracts compiled from schema: {schema_path}")
    return contracts


def compile_contract_text(objective: str) -> List[ExtractionContract]:
    """Fallback compiler for free-form objectives (no schema workbook).

    Treats each newline as one requested field. Generic: no domain knowledge.
    """
    contracts: List[ExtractionContract] = []
    for index, line in enumerate(objective.splitlines(), start=1):
        text = line.strip().lstrip("-0123456789. ").strip()
        if not text or len(text) < 4:
            continue
        if _NUMERIC_FIELD_RE.search(text):
            answer_type = "numeric"
        elif _BOOLEAN_FIELD_RE.search(text):
            answer_type = "boolean"
        else:
            answer_type = "narrative"
        contracts.append(
            ExtractionContract(
                contract_id=f"contract_{index:03d}",
                sheet="Objective",
                field=text[:80],
                question=text,
                context="objective",
                target_entity=text.split()[0] if text.split() else "record",
                target_attribute=text[:80],
                answer_type=answer_type,
                cardinality="scalar",
                entity_columns=[],
                value_column=text[:80],
            )
        )
    return contracts
