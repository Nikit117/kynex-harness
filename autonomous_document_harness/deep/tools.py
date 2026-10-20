"""LangChain tool registry for the Deep Agents runtime (§12).

Wraps the existing DocumentComputer primitives as native Deep Agents tools
plus contract-state tools (submit_candidate_answer, get_unresolved_contracts,
...). Tool outputs are compact JSON-serializable dicts with provenance.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_core.tools import tool

from .admission import admit_observation
from .store import AnswerRecord, RunStore


def _model_available(gateway) -> bool:
    try:
        return bool(gateway.model_reasoning_available())
    except Exception:
        return False


def record_linkage_verdict(snippet: str, subject: str, contract) -> tuple:
    """Record-anchored admission for meta-labeled answer columns.

    Returns ANSWER when the snippet shares at least two substantive tokens
    with the agent's record key AND bears a concrete value of the contract's
    answer type. Glossary definitions and generic prose share no record key
    and stay rejected. Falls back to weak-alignment otherwise (caller then
    runs the standard gate).
    """
    from ..models.admission import _has_concrete_value
    from ..workspace.indexing import tokenize

    key_tokens = {t for t in tokenize(subject or "") if len(t) > 2}
    snip_tokens = set(tokenize(snippet or ""))
    shared = key_tokens & snip_tokens
    if len(shared) >= 2 and _has_concrete_value(snippet or "", contract.answer_type):
        return (
            "ANSWER",
            f"record-linkage: {len(shared)} shared key tokens + value present",
        )
    return "AMBIGUOUS", "weak record linkage"


class DeepToolRegistry:
    """Builds the tool list for the primary investigator Deep Agent."""

    def __init__(
        self,
        computer,
        store: RunStore,
        gateway=None,
        filenames=None,
        max_admission_escalations: int = 500,
        event_sink=None,
    ) -> None:
        self.computer = computer
        self.store = store
        self.gateway = gateway or getattr(computer, "gateway", None)
        self.filenames = filenames or {}
        self._budget = getattr(computer, "budgets", None)
        # Bounded model-judge escalations for gray-zone admission verdicts.
        self.admission_escalations = 0
        self.max_admission_escalations = max_admission_escalations
        # Optional event hook: event_sink(tool_name, phase, payload) where
        # phase is "started" | "finished" | "verdict". UI layers subscribe
        # without touching agent internals. None by default (no overhead).
        self.event_sink = event_sink

    def _admit(self, snippet: str, contract, exclusions=None):
        """Admission with gray-zone model escalation inside a fixed budget.

        Heuristic ANSWERs are trusted without a model call. Other verdicts
        escalate to the model only while escalation budget remains; the
        counter tracks actual model-judge calls.
        """
        from ..models.admission import admit, evaluate

        verdict, reason = evaluate(snippet, contract, exclusions)
        if verdict == "ANSWER":
            return verdict, reason
        if (
            self.gateway is not None
            and self.admission_escalations < self.max_admission_escalations
            and _model_available(self.gateway)
        ):
            self.admission_escalations += 1
            return admit(
                snippet,
                contract,
                gateway=self.gateway,
                exclusions=exclusions,
                escalate_gray_zone=True,
            )
        return verdict, reason

    # -- helpers ------------------------------------------------------------
    # Maximum serialized characters returned to the model per tool call.
    # Bounds context growth and avoids pathological inputs to native
    # tokenizers/renderers downstream. Truncation is marked explicitly.
    MAX_TOOL_CHARS = 60000

    def _call(self, func, *args, **kwargs) -> Dict[str, Any]:
        """Invoke a DocumentComputer primitive, embedding provenance."""
        import time as _time

        tool_name = self._derive_tool_name(func, args)
        started = _time.perf_counter()
        self._emit(tool_name, "started", {"args": self._safe_args(kwargs, args)})
        try:
            result = func(*args, **kwargs)
            # registry.call returns an Observation; direct methods return (data, prov)
            if hasattr(result, "data") and hasattr(result, "provenance"):
                data = (
                    dict(result.data)
                    if isinstance(result.data, dict)
                    else {"data": result.data}
                )
                if result.provenance:
                    data["_provenance"] = result.provenance[:5]
                out = self._truncate(data)
            else:
                data, prov = result
                if isinstance(data, dict):
                    data = dict(data)
                    if prov:
                        data["_provenance"] = prov[:5]
                out = self._truncate(data)
            self._emit(
                tool_name,
                "finished",
                {
                    "status": "ok",
                    "latency_ms": round((_time.perf_counter() - started) * 1000.0, 1),
                    "summary": self._summarize(tool_name, out),
                },
            )
            return out
        except AttributeError as exc:
            self._emit(
                tool_name,
                "finished",
                {"status": "unavailable", "error": str(exc)[:200]},
            )
            return {"error": str(exc), "status": "unavailable"}
        except Exception as exc:
            self._emit(
                tool_name,
                "finished",
                {"status": "error", "error": f"{type(exc).__name__}: {exc}"[:200]},
            )
            return {"error": f"{type(exc).__name__}: {exc}", "status": "error"}

    @staticmethod
    def _derive_tool_name(func, args) -> str:
        name = getattr(func, "__name__", "") or ""
        if name == "call" and args and isinstance(args[0], str):
            return args[0]
        return name.lstrip("_") or "tool"

    @staticmethod
    def _safe_args(kwargs, args) -> Dict[str, Any]:
        safe = {k: str(v)[:120] for k, v in (kwargs or {}).items()}
        if args and not (len(args) == 1 and isinstance(args[0], str)):
            safe["_args"] = str(args)[:120]
        return safe

    @staticmethod
    def _summarize(tool_name: str, data: Any) -> str:
        try:
            if isinstance(data, dict):
                for key in ("count", "answers", "admitted", "verdict", "status"):
                    if key in data:
                        return f"{tool_name}: {data[key]}"
                if "hits" in data and isinstance(data["hits"], list):
                    return f"{tool_name}: {len(data['hits'])} hits"
                if "rows" in data and isinstance(data["rows"], list):
                    return f"{tool_name}: {len(data['rows'])} rows"
            return f"{tool_name}: ok"
        except Exception:
            return f"{tool_name}: ok"

    def _emit(self, tool_name: str, phase: str, payload: Dict[str, Any]) -> None:
        sink = self.event_sink
        if sink is None:
            return
        try:
            sink(tool_name, phase, payload)
        except Exception:
            pass

    @classmethod
    def _truncate(cls, data: Any) -> Any:
        """Cap serialized tool output; mark truncation explicitly."""
        import json as _json

        try:
            text = _json.dumps(data, default=str)
        except Exception:
            return data
        if len(text) <= cls.MAX_TOOL_CHARS:
            return data
        if isinstance(data, dict):
            capped = dict(data)
            # Keep the most valuable keys whole; truncate long text fields.
            for key in ("text", "rows", "hits", "headings", "pages", "tables"):
                if key in capped:
                    try:
                        part = _json.dumps(capped[key], default=str)
                        if len(part) > cls.MAX_TOOL_CHARS // 3:
                            capped[key] = str(capped[key])[: cls.MAX_TOOL_CHARS // 3]
                    except Exception:
                        pass
            capped["_truncated"] = True
            return capped
        return data

    def _note(self, tool: str, args: Dict[str, Any], summary: str) -> None:
        self.store.record_observation(tool, args, summary)
        if self._budget is not None:
            try:
                self._budget.tool_calls = getattr(self._budget, "tool_calls", 0) + 1
            except Exception:
                pass

    def note_model_call(self, tokens_in: int = 0, tokens_out: int = 0) -> None:
        """Record one model invocation (called by the agent wrapper)."""
        if self._budget is not None:
            try:
                self._budget.model_calls = getattr(self._budget, "model_calls", 0) + 1
                self._budget.tokens_used = (
                    getattr(self._budget, "tokens_used", 0) + tokens_in + tokens_out
                )
            except Exception:
                pass

    def note(self, tool: str, args: Dict[str, Any], summary: str) -> None:
        """Public alias for tools to record their own observations."""
        self._note(tool, args, summary)

    def submit_one(
        self,
        contract_id: str,
        subject: str,
        attribute: str,
        value: str,
        document_id: str,
        page: Optional[int],
        cell: Optional[str],
        snippet: str,
        method: str,
        units: Optional[str],
    ) -> Dict[str, Any]:
        """Admit + record a single candidate answer. Shared by single/batch."""
        from ..verification.entailment import entails

        store = self.store
        contract = store.get_contract(contract_id)
        if contract is None:
            return {
                "verdict": "IRRELEVANT",
                "reason": f"unknown contract '{contract_id}'",
            }
        snippet = (snippet or "")[:2000]
        from ..models.admission import negative_verdict_ok

        if negative_verdict_ok(str(value), snippet, subject or ""):
            # Explicit negative verdict with search-scope evidence: the
            # absence was established by adequate search, not assumed.
            verdict, reason = (
                "ANSWER",
                "explicit negative verdict with search scope",
            )
            store.record_strategy(contract_id, f"{method}+negative")
        elif not (contract.target_attribute or "").strip():
            # Meta-labeled answer column: the agent's record key anchors
            # admission (shared substantive tokens + concrete value).
            # Falls through to the standard gate when linkage is weak.
            verdict, reason = record_linkage_verdict(snippet, subject or "", contract)
            if verdict != "ANSWER":
                verdict, reason = self._admit(
                    snippet or f"{subject} {attribute} {value}", contract
                )
        else:
            verdict, reason = self._admit(
                snippet or f"{subject} {attribute} {value}", contract
            )
        self.note(
            "submit_candidate_answer",
            {
                "contract_id": contract_id,
                "subject": (subject or "")[:120],
                "attribute": (attribute or "")[:120],
                "value": str(value)[:120],
                "snippet": snippet[:300],
                "method": method,
            },
            f"{verdict}: {reason}",
        )
        store.record_strategy(contract_id, method)
        if verdict != "ANSWER":
            store.record_rejection(contract_id)
            self._emit(
                "submit_candidate_answer",
                "verdict",
                {
                    "verdict": verdict,
                    "reason": reason[:160],
                    "contract_id": contract_id,
                },
            )
            return {"verdict": verdict, "reason": reason}
        verification = "uncertain"
        if snippet and entails(str(value), snippet, 0.34):
            verification = "verified"
        record = AnswerRecord(
            answer_id=f"ans_{len(store.answers) + 1:04d}",
            contract_id=contract_id,
            subject=subject,
            attribute=attribute,
            value=value,
            units=units,
            method=method,
            document_id=document_id,
            page=page,
            cell=cell,
            snippet=snippet,
            admission=verdict,
            verification=verification,
        )
        store.add_answer(record)
        store.record_candidate_region(contract_id, f"{document_id}#p{page}")
        self._emit(
            "submit_candidate_answer",
            "verdict",
            {
                "verdict": verdict,
                "contract_id": contract_id,
                "subject": (subject or "")[:120],
                "value": str(value)[:120],
            },
        )
        return {
            "verdict": verdict,
            "reason": reason,
            "answer_id": record.answer_id,
            "verification": verification,
        }

    def tools(self) -> List[Any]:
        """Return the full tool list for create_deep_agent."""
        computer = self.computer
        store = self.store
        registry = self

        # -- corpus ----------------------------------------------------------
        @tool
        def list_documents() -> Dict[str, Any]:
            """List every document in the workspace corpus with page/table counts."""
            return registry._call(computer._list_documents)

        @tool
        def inspect_document(doc_id: str) -> Dict[str, Any]:
            """Profile one document: pages, headings, sheets, tables, OCR need."""
            return registry._call(computer._inspect_document, doc_id)

        @tool
        def get_document_metadata(doc_id: str) -> Dict[str, Any]:
            """Filename, fingerprint, size and classification of one document."""
            return registry._call(computer._metadata, doc_id)

        # -- retrieval -------------------------------------------------------
        @tool
        def search_text(
            query: str, top_k: int = 10, mode: str = "hybrid"
        ) -> Dict[str, Any]:
            """Search document text. mode: lexical | semantic | hybrid (RRF)."""
            return registry._call(
                computer.registry.call,
                "search_text",
                query=query,
                top_k=top_k,
                mode=mode,
            )

        @tool
        def search_headings(query: str, top_k: int = 10) -> Dict[str, Any]:
            """Search document headings/structure for sections."""
            return registry._call(
                computer.registry.call, "search_headings", query=query, top_k=top_k
            )

        @tool
        def find_supporting_evidence(query: str, top_k: int = 5) -> Dict[str, Any]:
            """Hybrid hits that may support an answer to the given question."""
            return registry._call(
                computer.registry.call,
                "find_supporting_evidence",
                query=query,
                top_k=top_k,
            )

        @tool
        def find_conflicting_evidence(query: str, top_k: int = 10) -> Dict[str, Any]:
            """Cross-document numeric disagreements about the given subject."""
            return registry._call(
                computer.registry.call,
                "find_contradicting_evidence",
                query=query,
                top_k=top_k,
            )

        # -- reading ---------------------------------------------------------
        @tool
        def read_page(doc_id: str, page: int) -> Dict[str, Any]:
            """Native text + structure of one 1-based page."""
            return registry._call(
                computer.registry.call, "read_page", doc_id=doc_id, page=page
            )

        @tool
        def read_section(doc_id: str, heading: str) -> Dict[str, Any]:
            """Text under a heading, across page breaks."""
            return registry._call(
                computer.registry.call, "read_section", doc_id=doc_id, heading=heading
            )

        @tool
        def read_surrounding_context(
            doc_id: str, page: int, window: int = 1
        ) -> Dict[str, Any]:
            """Pages around a target page (neighboring-page exploration)."""
            return registry._call(
                computer.registry.call,
                "read_surrounding_context",
                doc_id=doc_id,
                page=page,
                window=window,
            )

        @tool
        def read_spreadsheet(doc_id: str) -> Dict[str, Any]:
            """Workbook overview: sheets, dims, merged cells, formulas."""
            return registry._call(
                computer.registry.call, "read_spreadsheet", doc_id=doc_id
            )

        @tool
        def read_sheet_range(
            doc_id: str, sheet: str, cell_range: str = "A1:Z50"
        ) -> Dict[str, Any]:
            """Addressed cell range from a spreadsheet sheet."""
            return registry._call(
                computer.registry.call,
                "read_sheet_range",
                doc_id=doc_id,
                sheet=sheet,
                cell_range=cell_range,
            )

        # -- tables ----------------------------------------------------------
        @tool
        def detect_tables(doc_id: str, page: Optional[int] = None) -> Dict[str, Any]:
            """Candidate tables with dimensions and header snippets."""
            return registry._call(
                computer.registry.call, "detect_tables", doc_id=doc_id, page=page
            )

        @tool
        def extract_table(
            doc_id: str,
            table_index: int = 0,
            page: Optional[int] = None,
            sheet: Optional[str] = None,
        ) -> Dict[str, Any]:
            """Full rows of one table (or a whole sheet). Includes a
            `text` field with the table rendered as readable lines so values
            are visible without parsing JSON arrays."""
            data = registry._call(
                computer.registry.call,
                "extract_table",
                doc_id=doc_id,
                table_index=table_index,
                page=page,
                sheet=sheet,
            )
            if isinstance(data, dict) and "rows" in data:
                lines = []
                for i, row in enumerate(data["rows"]):
                    cells = [str(c) for c in row if str(c).strip()]
                    if cells:
                        lines.append(f"r{i}: {' | '.join(cells)}")
                data["text"] = "\n".join(lines)
            return data

        @tool
        def reconstruct_table(
            doc_id: str, page: Optional[int] = None, sheet: Optional[str] = None
        ) -> Dict[str, Any]:
            """Logical tables: header hierarchies, banner groups, continuations."""
            return registry._call(
                computer.registry.call,
                "reconstruct_table",
                doc_id=doc_id,
                page=page,
                sheet=sheet,
            )

        # -- perception ------------------------------------------------------
        @tool
        def render_page(doc_id: str, page: int, dpi: int = 110) -> Dict[str, Any]:
            """Render a PDF page to PNG (escalation ladder step)."""
            return registry._call(computer._render_page, doc_id, page, dpi)

        @tool
        def render_region(
            doc_id: str, page: int, bbox: List[float], dpi: int = 150
        ) -> Dict[str, Any]:
            """Render and crop a bbox region (pixels of the rendered page)."""
            return registry._call(
                computer._render_region, doc_id, page, tuple(bbox), dpi
            )

        @tool
        def run_azure_document_intelligence(
            doc_id: str, pages: Optional[List[int]] = None
        ) -> Dict[str, Any]:
            """Azure DI layout/OCR/tables for explicit pages (scanned docs)."""
            return registry._call(computer._di_analyze, doc_id, pages)

        @tool
        def inspect_page_visually(
            doc_id: str, page: int, question: str
        ) -> Dict[str, Any]:
            """GPT-5.4 Vision reading of a rendered page. Ask a specific question."""
            return registry._call(
                computer._vision_inspect, doc_id, page, None, question
            )

        @tool
        def inspect_region_visually(
            doc_id: str, page: int, bbox: List[float], question: str
        ) -> Dict[str, Any]:
            """GPT-5.4 Vision reading of a targeted crop. Use for complex tables."""
            return registry._call(
                computer._vision_inspect, doc_id, page, tuple(bbox), question
            )

        @tool
        def inspect_table_visually(
            doc_id: str, page: int, table_index: int = 0
        ) -> Dict[str, Any]:
            """GPT-5.4 Vision check of one table region (merged/hierarchical headers)."""
            return registry._call(
                computer._vision_inspect_table, doc_id, page, table_index
            )

        @tool
        def vision_verify_claim(
            doc_id: str,
            page: int,
            subject: str,
            attribute: str,
            value: str,
            bbox: Optional[List[float]] = None,
        ) -> Dict[str, Any]:
            """GPT-5.4 visual verdict on one claim against a render/crop."""
            return registry._call(
                computer._vision_verify,
                doc_id,
                page,
                subject,
                attribute,
                value,
                tuple(bbox) if bbox else None,
                "",
            )

        # -- contract state --------------------------------------------------
        @tool
        def get_unresolved_contracts() -> Dict[str, Any]:
            """Contracts not yet COMPLETE/NOT_FOUND, with counts and hints."""
            unresolved = store.unresolved_contracts()
            return {
                "count": len(unresolved),
                "contracts": [
                    {
                        "contract_id": c.contract_id,
                        "sheet": c.sheet,
                        "field": c.field,
                        "target_entity": c.target_entity,
                        "target_attribute": c.target_attribute,
                        "answer_type": c.answer_type,
                        "cardinality": c.cardinality,
                    }
                    for c in unresolved
                ],
            }

        @tool
        def get_contract_status(contract_id: str) -> Dict[str, Any]:
            """State, candidate count and strategies for one contract."""
            contract = store.get_contract(contract_id)
            if contract is None:
                return {"error": f"unknown contract '{contract_id}'"}
            st = store.contract_state[contract_id]
            return {
                "contract_id": contract.contract_id,
                "sheet": contract.sheet,
                "field": contract.field,
                "target_entity": contract.target_entity,
                "target_attribute": contract.target_attribute,
                "answer_type": contract.answer_type,
                "state": st["state"],
                "answers": st["answers"],
                "candidate_regions": st["candidate_regions"],
                "strategies_attempted": st["strategies_attempted"],
                "queries": st["queries"],
            }

        @tool
        def get_coverage_report() -> Dict[str, Any]:
            """Corpus coverage: pages, candidates, tables, DI/vision usage."""
            return store.coverage.to_dict()

        @tool
        def submit_candidate_answer(
            contract_id: str,
            subject: str,
            attribute: str,
            value: str,
            document_id: str,
            page: Optional[int] = None,
            cell: Optional[str] = None,
            snippet: str = "",
            method: str = "native_text",
            units: Optional[str] = None,
        ) -> Dict[str, Any]:
            """Submit one candidate answer. Semantic admission runs first:
            only ANSWER verdicts may advance a contract. Returns the verdict."""
            return registry.submit_one(
                contract_id,
                subject,
                attribute,
                value,
                document_id,
                page,
                cell,
                snippet,
                method,
                units,
            )

        @tool
        def submit_answer_batch(
            contract_id: str,
            answers: List[Dict[str, Any]],
            method: str = "native_text",
        ) -> Dict[str, Any]:
            """Submit up to 25 candidate answers in ONE call (throughput).

            Use after reading a table or section: one item per row/record with
            subject/attribute/value/document_id/page/cell/snippet (+units).
            EVERY item is independently admission-checked under the same
            invariant — only ANSWER verdicts advance the contract. Returns
            per-item verdicts plus admitted/rejected counts.
            """
            items = answers[:25] if isinstance(answers, list) else []
            results = []
            admitted = rejected = 0
            for index, item in enumerate(items):
                if not isinstance(item, dict):
                    results.append(
                        {
                            "index": index,
                            "verdict": "IRRELEVANT",
                            "reason": "not an object",
                        }
                    )
                    rejected += 1
                    continue
                outcome = registry.submit_one(
                    contract_id,
                    str(item.get("subject", "")),
                    str(item.get("attribute", "")),
                    str(item.get("value", "")),
                    str(item.get("document_id", "")),
                    item.get("page"),
                    item.get("cell"),
                    str(item.get("snippet", "")),
                    str(item.get("method", method or "native_text")),
                    item.get("units"),
                )
                outcome["index"] = index
                results.append(outcome)
                if outcome["verdict"] == "ANSWER":
                    admitted += 1
                else:
                    rejected += 1
            return {"admitted": admitted, "rejected": rejected, "results": results}

        @tool
        def verify_answer(answer_id: str) -> Dict[str, Any]:
            """Independent evidence check: does the cited snippet state the value?"""
            answer = next((a for a in store.answers if a.answer_id == answer_id), None)
            if answer is None:
                return {
                    "verdict": "AMBIGUOUS",
                    "reason": f"unknown answer '{answer_id}'",
                }
            from ..verification.entailment import entails

            ok = bool(answer.snippet) and entails(
                str(answer.value), answer.snippet, 0.34
            )
            answer.verification = "verified" if ok else "unresolved"
            return {
                "verdict": "SUPPORTED" if ok else "NOT_SUPPORTED",
                "answer_id": answer_id,
                "verification": answer.verification,
            }

        @tool
        def request_targeted_recovery(
            contract_id: str, failure: str, next_action: str
        ) -> Dict[str, Any]:
            """Record a targeted recovery task for a failed contract."""
            store.record_recovery(contract_id, failure, next_action)
            store.record_strategy(contract_id, f"recovery:{next_action[:60]}")
            return {"recorded": True, "contract_id": contract_id}

        return [
            list_documents,
            inspect_document,
            get_document_metadata,
            search_text,
            search_headings,
            find_supporting_evidence,
            find_conflicting_evidence,
            read_page,
            read_section,
            read_surrounding_context,
            read_spreadsheet,
            read_sheet_range,
            detect_tables,
            extract_table,
            reconstruct_table,
            render_page,
            render_region,
            run_azure_document_intelligence,
            inspect_page_visually,
            inspect_region_visually,
            inspect_table_visually,
            vision_verify_claim,
            get_unresolved_contracts,
            get_contract_status,
            get_coverage_report,
            submit_candidate_answer,
            submit_answer_batch,
            verify_answer,
            request_targeted_recovery,
        ]
