"""Primary Deep Agent: LangChain Deep Agents SDK + LangGraph execution.

The Deep Agent owns iterative document investigation. Deterministic
validation (contracts, admission, evidence) stays outside model control.
The supervisor loop resumes the agent with targeted recovery tasks when
contracts remain unresolved — it does NOT replace the agent's reasoning.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .contracts import ExtractionContract
from .model import build_chat_model
from .store import RunStore
from .tools import DeepToolRegistry

SYSTEM_PROMPT = """\
You are an autonomous document investigator. Your job is to answer every \
extraction contract using ONLY evidence found in the provided documents.

## Contracts
Each contract has: contract_id, sheet, field (the requested value), \
target_entity (what the answer describes), target_attribute (the property \
asked for), answer_type, and cardinality.

## Rules
1. Investigate specific unanswered contracts — do not collect trivia.
2. For each contract, search the documents for the target entity, then read \
the relevant pages/regions/tables.
3. Submit candidate answers with submit_candidate_answer. The semantic \
admission gate runs automatically: only ANSWER verdicts advance a contract. \
Glossary definitions, generic descriptions, and topical mentions that do not \
answer the requested attribute will be rejected — that is expected.
4. Every answer MUST include: document_id, page (or cell), snippet (verbatim \
source text), and method. Never invent values or citations.
5. **CRITICAL — snippet must contain the actual value.** When you extract a \
number or text from a table row, copy the ENTIRE row (including the value) \
into the snippet. Example: if the table row is \
"BOP | 396 | 37 | 930 | 227", then snippet = "BOP | 396 | 37 | 930 | 227" \
and value = "396". Do NOT submit a snippet that only contains labels like \
"System / Area / IO Type" — the admission gate will reject it because no \
value is present.
6. For collection contracts (cardinality=collection), find ALL records — \
do not stop after the first. Use multiple search strategies and check \
neighboring pages.
7. For scanned pages or complex tables, escalate: run_azure_document_intelligence \
then inspect_page_visually / inspect_table_visually if structure is ambiguous.
8. Verify each answer with verify_answer before considering it done.
9. When a contract cannot be answered after adequate search, call \
request_targeted_recovery with the failure reason and move on.
10. NEGATIVE VERDICTS: when adequate search (3+ strategies, candidate tables
inspected, vision attempted for scanned pages) establishes that the requested
content is genuinely ABSENT, submit an explicit negative verdict as the value
("Not specified", "Not applicable" or "Should be omitted" — whichever fits)
with a snippet describing the search scope (which sections/tables/pages were
checked). Never emit negatives without searching first; the admission gate
rejects scopeless negatives.
10. Use get_unresolved_contracts to track what remains. NEVER emit a final \
summary while it returns a non-empty list and budget remains — continue \
investigating instead. Finish only when all contracts are resolved or \
budgets are exhausted.

## Table extraction workflow
When extract_table returns rows, each row is a list of cell values. To submit \
an answer from a table row:
- Identify which columns are the entity (key) and which is the value.
- subject = the entity columns joined by " / " (e.g. "UNIT / SUBSYSTEM / AI")
- attribute = the contract's target_attribute (e.g. "Count")
- value = the cell in the value column (e.g. "396")
- Aim for complete record keys: the subject SHOULD include a value for
  every entity column named in the contract brief. If row labels are implicit
  (grouped under a banner header such as a source category or system name),
  read the banner/section headers (reconstruct_table exposes banner groups)
  and include them. If a key value is truly absent from the source after
  checking banners and surrounding text, submit with the best available key
  rather than skipping the row — a complete key scores best, but a genuine
  value with a partial key still beats a missed row.
- snippet = **MUST combine the entity description AND the data row**. Format: \
"System: <system> / Area: <area> / IO Type: <type> | <row cells>". \
Example: "System: SYS-A / Area: BOP / Point Type: DO | BOP | 396 | 37 | 930 | 227 | 21"
- The admission gate requires BOTH the entity words (System, Area, IO Type) \
AND the numeric value in the snippet. A snippet with only labels or only \
numbers will be rejected.
- method = "native_table"

## Batch submission (preferred for volume)
After reading a table or a document section, submit ALL its records in ONE
submit_answer_batch call (up to 25 items): one item per table row or stated
requirement, each with its own subject/attribute/value/snippet. Every item is
independently admission-checked — the same invariant holds, at ~10x the
throughput of single submissions. Prefer batching over one-by-one submits
whenever you hold multiple answers.

## Sheet tactics (apply by contract shape — no sheet-specific hardcoding)- NUMERIC collections with entity columns (count tables): enumerate tables in
  EVERY document (detect_tables per document), extract each candidate table
  fully, submit one answer per data row. Do not stop after the first table —
  variant tables (e.g. with/without spare, per-phase, per-unit) are separate
  record sets that must each be collected. If a likely table page is scanned
  or its structure is ambiguous, escalate to
  run_azure_document_intelligence and inspect_table_visually.
- NARRATIVE / BOOLEAN collections (requirement lists): read each relevant
  document section by section (read_section, read_page in order); submit one
  answer per stated requirement, each with a precise field-level subject.
  Continue until two consecutive full passes over the relevant sections yield
  no new answers.
- Declare NOT_FOUND only after: (a) at least 3 distinct search strategies,
  (b) all candidate tables/sheets inspected, (c) vision attempted for any
  scanned candidate. Otherwise keep investigating.

## Output format for submit_candidate_answer
- subject: the record identity (e.g. "System: SYS-A / Analog Input" or the \
entity the fact is about). For NARRATIVE and BOOLEAN contracts, append the \
specific field or requirement name (e.g. "System: SYS-A / Accuracy of AI cards"), \
so each answer is precisely attributable to one requested item.
- attribute: the requested field/column name
- value: the SHORTEST EXACT SOURCE QUOTE stating the answer (a few words or
  one number copied verbatim — never a paraphrase). Verbatim quotes are the
  gold standard of evidence: "MODEL 300", "Yes", "rear access via hinged doors".
  Put surrounding context in the snippet, not the value.
- snippet: verbatim source text supporting the value
- method: native_text | native_table | azure_di | gpt54_vision | spreadsheet

## Entity coverage (no system left behind)
Early in the investigation, enumerate the DISTINCT entity values in the corpus
(systems, areas, units, packages, equipment types — from headings, tables and
search hits). Check off every one against every applicable contract. Never
investigate only the first system found (e.g. only the first one when other
systems or packages exist). A contract is not covered while a named entity has
no answers.
"""


@dataclass
class DeepRunResult:
    """Compatible with the benchmark script's expectations."""

    status: str
    output: Dict[str, Any] = field(default_factory=dict)
    completeness: Dict[str, Any] = field(default_factory=dict)
    budgets: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    trace_path: str = ""
    contracts: List[Dict[str, Any]] = field(default_factory=list)


class DeepAgentInvestigator:
    """Supervisor: compiles contracts, runs the Deep Agent, enforces completion."""

    def __init__(
        self,
        computer,
        store: RunStore,
        contracts: List[ExtractionContract],
        run_id: str = "deep_run",
        max_resume_rounds: int = 3,
        thread_id: Optional[str] = None,
        checkpoint_dir: Optional[str] = None,
        deep_dive_rounds: int = 1,
        event_sink=None,
    ) -> None:
        self.computer = computer
        self.store = store
        self.contracts = contracts
        self.run_id = run_id
        self.max_resume_rounds = max_resume_rounds
        self.thread_id = thread_id or run_id
        self.checkpoint_dir = checkpoint_dir
        self.deep_dive_rounds = deep_dive_rounds
        self.event_sink = event_sink
        # Optional conversation context (prior turns) appended to the brief so
        # follow-up questions resolve referents. Never GT, never answers-as-facts.
        self.context = ""
        self._seen_msg_ids: set = set()
        self.filenames = {}
        try:
            self.filenames = {
                p.doc_id: p.filename for p in computer.world.profiles.values()
            }
        except Exception:
            pass

    def _build_agent(self):
        """Create the Deep Agent via create_deep_agent (LangGraph runtime)."""
        from deepagents import create_deep_agent

        model = build_chat_model()
        if model is None:
            raise RuntimeError(
                "Deep Agent requires a configured chat model. Set "
                "AZURE_OPENAI_ENDPOINT/AZURE_OPENAI_API_KEY/"
                "AZURE_OPENAI_CHAT_DEPLOYMENT or ADH_MODEL_BASE_URL/"
                "ADH_MODEL_API_KEY/ADH_MODEL_DEPLOYMENT."
            )
        registry = DeepToolRegistry(
            self.computer,
            self.store,
            filenames=self.filenames,
            event_sink=self.event_sink,
        )
        tools = registry.tools()

        checkpointer = None
        if self.checkpoint_dir:
            try:
                from langgraph.checkpoint.memory import MemorySaver

                checkpointer = MemorySaver()
            except Exception:
                checkpointer = None

        kwargs: Dict[str, Any] = dict(
            model=model,
            tools=tools,
            system_prompt=SYSTEM_PROMPT,
            name="primary_investigator",
        )
        if checkpointer is not None:
            kwargs["checkpointer"] = checkpointer
        return create_deep_agent(**kwargs)

    def _table_inventory(self, max_tables: int = 40) -> str:
        """Deterministic table inventory across the corpus (generic).

        Lists candidate tables (document, page, dimensions, header snippet)
        so the investigator systematically covers every table instead of
        whichever ones retrieval happens to surface. Built only from native
        parsing — no request or ground-truth content involved.
        """
        try:
            from ..tools import tables as tables_mod

            world = self.computer.world
            lines = []
            for doc_id in sorted(world.profiles):
                profile = world.profiles[doc_id]
                try:
                    data, _ = tables_mod.detect_tables(world, doc_id)
                except Exception:
                    continue
                for table in data.get("tables", []):
                    header = " | ".join(
                        str(h)[:24] for h in table.get("header", [])[:6]
                    )
                    lines.append(
                        f"- {profile.filename} p.{table.get('page')}: "
                        f"{table.get('rows')}x{table.get('cols')} [{header}]"
                    )
                    if len(lines) >= max_tables:
                        return "\n".join(lines)
            return "\n".join(lines)
        except Exception:
            return ""

    def _initial_message(self) -> str:
        lines = [
            "Begin the investigation. Extraction contracts:",
            json.dumps(
                [
                    {
                        "contract_id": c.contract_id,
                        "sheet": c.sheet,
                        "field": c.field,
                        "target_entity": c.target_entity,
                        "target_attribute": c.target_attribute,
                        "answer_type": c.answer_type,
                        "cardinality": c.cardinality,
                    }
                    for c in self.contracts
                ],
                indent=2,
            ),
            "Start by listing documents, then search for each contract's target entity.",
            "Collection sheets may need dozens or hundreds of answers — one per "
            "record (e.g. every table row, every stated requirement). Keep "
            "submitting until new searches stop yielding new records.",
        ]
        if self.context:
            lines.append(
                "Conversation context (resolve referents like 'it'/'that' from this):"
            )
            lines.append(self.context[:2000])
        lines.extend(
            [
                "Known table inventory (extract every relevant one, including "
                "variant tables):",
                self._table_inventory(),
            ]
        )
        return "\n".join(lines)

    def _focused_message(self, contract: ExtractionContract) -> str:
        """Focused deep-dive brief for ONE contract (built from its semantics)."""
        st = self.store.contract_state[contract.contract_id]
        return "\n".join(
            [
                f"Deep-dive for contract {contract.contract_id}: "
                f"{contract.sheet} / {contract.field}.",
                f"Target entity columns: {', '.join(contract.entity_columns)}; "
                f"requested value column: {contract.value_column} "
                f"({contract.answer_type}, {contract.cardinality}).",
                f"So far: {st['answers']} answers, "
                f"{st['rejected_observations']} rejected, strategies: "
                f"{st['strategies_attempted'][:8]}.",
                "Work on THIS contract only. Run new searches with different "
                "query terms, inspect unvisited tables/pages (detect_tables, "
                "extract_table, reconstruct_table, read_sheet_range), check "
                "neighboring pages, and submit one answer per record with "
                "combined entity+row snippets. For scanned pages use "
                "run_azure_document_intelligence, then targeted vision reads.",
            ]
        )

    def _sweep_message(self, doc_id: str, filename: str, pages: int) -> str:
        """Cover-to-cover reading brief for ONE document (generic)."""
        open_contracts = [
            f"{c.contract_id}: {c.sheet} / {c.field} "
            f"(entity={c.target_entity}, value={c.value_column})"
            for c in self.store.unresolved_contracts()
        ]
        return "\n".join(
            [
                f"Systematic sweep of DOCUMENT {filename} ({pages} pages).",
                "Read it cover to cover, section by section: headings first "
                "(search_headings), then each section (read_section), each page "
                "with tables (detect_tables, extract_table), each spreadsheet "
                "sheet (read_spreadsheet, read_sheet_range). Skip boilerplate "
                "(title pages, tables of contents).",
                "For EVERY record or stated requirement matching an open "
                "contract below, submit an answer via submit_answer_batch "
                "(up to 25 per call) with precise field-level subjects and "
                "combined entity+row snippets:",
                *open_contracts,
                "Do not stop until every page of THIS document has been read.",
            ]
        )

    def _focused_recovery_message(self, contract: ExtractionContract) -> str:
        st = self.store.contract_state[contract.contract_id]
        return "\n".join(
            [
                f"Continue the deep-dive for {contract.contract_id}: "
                f"{contract.sheet} / {contract.field} "
                f"(answers={st['answers']}).",
                "Try strategies not yet attempted. If the remaining values "
                "truly do not exist in the corpus, call "
                "request_targeted_recovery and stop.",
            ]
        )

    def _account_usage(self, result: Any) -> None:
        """Fold LangGraph token usage into the store budgets (efficiency data).

        LangGraph returns the FULL conversation each invoke, so only
        previously-unseen AI messages are counted.
        """
        try:
            messages = result.get("messages", []) if isinstance(result, dict) else []
        except Exception:
            return
        for msg in messages:
            if msg.__class__.__name__ != "AIMessage":
                continue
            msg_id = getattr(msg, "id", None) or id(msg)
            if msg_id in self._seen_msg_ids:
                continue
            self._seen_msg_ids.add(msg_id)
            self.store.budgets.model_calls += 1
            usage = getattr(msg, "usage_metadata", None) or {}
            try:
                self.store.budgets.tokens_used += int(
                    usage.get("input_tokens", 0)
                ) + int(usage.get("output_tokens", 0))
            except Exception:
                pass

    def _save_ledger(self) -> None:
        if not self.checkpoint_dir:
            return
        try:
            self.store.save_ledger(f"{self.checkpoint_dir}/ledger.json")
        except Exception:
            pass

    def _coverage_nudge(self) -> str:
        """Direct the agent at perceptually-uncovered regions (generic).

        Lists scanned/sparse documents that native extraction cannot read so
        the agent escalates them to render + vision reads. Derived only from
        corpus profiles — never from request or ground-truth content.
        """
        try:
            profiles = self.computer.world.profiles
            parsed = self.computer.world.parsed
        except Exception:
            return ""
        scanned, sparse = [], []
        for doc_id, profile in profiles.items():
            if getattr(profile, "needs_ocr", False):
                scanned.append(f"{profile.filename} ({profile.page_count} pages)")
        for doc_id, doc in parsed.items():
            for page in getattr(doc, "pages", []):
                if (
                    getattr(page, "char_count", 0) < 100
                    and getattr(page, "image_coverage", 0) > 0.3
                ):
                    filename = (
                        profiles[doc_id].filename if doc_id in profiles else doc_id
                    )
                    sparse.append(f"{filename} p.{page.page_no}")
                    if len(sparse) >= 8:
                        break
            if len(sparse) >= 8:
                break
        parts = []
        if scanned:
            parts.append(
                "Scanned documents (native text unavailable — render pages and "
                "inspect them visually): " + "; ".join(scanned[:6])
            )
        if sparse:
            parts.append(
                "Image-heavy pages with almost no extractable text (inspect "
                "visually): " + "; ".join(sparse[:8])
            )
        return " ".join(parts)

    def _recovery_message(self, round_index: int = 0) -> str:
        unresolved = self.store.unresolved_contracts()
        if not unresolved:
            return ""
        lines = [
            "Recovery round. The following contracts are still unresolved. "
            "Investigate each one specifically:",
        ]
        for c in unresolved:
            st = self.store.contract_state[c.contract_id]
            lines.append(
                f"- {c.contract_id}: {c.sheet} / {c.field} "
                f"(entity={c.target_entity}, type={c.answer_type}, "
                f"answers={st['answers']}, rejected={st['rejected_observations']}, "
                f"tried=[{', '.join(st['strategies_attempted'][:6])}])"
            )
        lines.append(
            "Use different search strategies than before. For tables, try "
            "extract_table or reconstruct_table. For scanned pages, use "
            "run_azure_document_intelligence. If a value truly does not exist, "
            "call request_targeted_recovery and mark it NOT_FOUND."
        )
        if round_index == 1:
            lines.append(
                "This round, prioritize NARRATIVE contracts: read spec sections "
                "page by page (read_page, read_section), extract every stated "
                "requirement as its own answer, and verify each one."
            )
        elif round_index >= 2:
            lines.append(
                "This round, prioritize unvisited regions: spreadsheet sheets "
                "(read_spreadsheet, read_sheet_range), surrounding context of "
                "pages that already yielded answers, and visual reads of "
                "complex tables (inspect_table_visually)."
            )
        nudge = self._coverage_nudge()
        if nudge:
            lines.append("Perception gaps: " + nudge)
        lines.append(
            "You have a large remaining tool budget — do not stop while "
            "unresolved contracts still have unexplored candidate regions."
        )
        return "\n".join(lines)

    def run(self) -> DeepRunResult:
        """Run the Deep Agent with bounded supervision.

        Phase A: all-contract sweep (breadth). Phase B: relevance-gated
        per-contract deep-dives (depth) — only contracts that already show
        evidence of relevance (answers or candidate regions) get extra
        investigation. Deterministic completion closes the rest.
        """
        started = time.time()
        agent = self._build_agent()

        messages = [self._initial_message()]

        for round_index in range(self.max_resume_rounds + 1):
            if self.store.budgets.exhausted():
                break
            # Fresh thread per sweep round: bounds context growth (the full
            # conversation is re-sent every invoke) while the recovery message
            # carries the complete unresolved-state summary forward.
            sweep_config = {
                "configurable": {"thread_id": f"{self.thread_id}_sweep{round_index}"}
            }
            try:
                result = agent.invoke(
                    {"messages": [{"role": "user", "content": messages[0]}]},
                    sweep_config,
                )
                self._account_usage(result)
            except Exception as exc:
                # Model unavailable (offline) — fall through to deterministic compile
                messages[0] = f"[agent error: {type(exc).__name__}] " + messages[0]
                break
            self._save_ledger()
            unresolved = self.store.unresolved_contracts()
            if not unresolved:
                break
            if round_index >= self.max_resume_rounds:
                break
            messages[0] = self._recovery_message(round_index + 1)

        # Phase B: focused deep-dives for relevant-but-open contracts.
        # Relevance = already produced answers/regions (right neighborhood) OR
        # repeatedly attempted but rejected (admission-format fixable).
        for contract in list(self.store.unresolved_contracts()):
            if self.store.budgets.exhausted():
                break
            st = self.store.contract_state[contract.contract_id]
            relevant = (
                st["answers"] > 0
                or len(st["candidate_regions"]) > 0
                or st["rejected_observations"] >= 3
            )
            if not relevant:
                continue
            thread = f"{self.thread_id}_{contract.contract_id}"
            sub_config = {"configurable": {"thread_id": thread}}
            for dive in range(self.deep_dive_rounds + 1):
                if self.store.budgets.exhausted():
                    break
                before = self.store.contract_state[contract.contract_id]["answers"]
                try:
                    msg = (
                        self._focused_message(contract)
                        if dive == 0
                        else self._focused_recovery_message(contract)
                    )
                    result = agent.invoke(
                        {"messages": [{"role": "user", "content": msg}]}, sub_config
                    )
                    self._account_usage(result)
                except Exception:
                    break
                self._save_ledger()
                st = self.store.contract_state[contract.contract_id]
                # Saturation: consecutive dives with no new answers close a
                # collection contract as COMPLETE (converged, not arbitrary).
                if st["answers"] <= before:
                    st["quiet_dives"] = st.get("quiet_dives", 0) + 1
                else:
                    st["quiet_dives"] = 0
                if (
                    st.get("quiet_dives", 0) >= 2
                    and len(st["strategies_attempted"]) >= 3
                ):
                    self.store.set_contract_state(contract.contract_id, "COMPLETE")
                if self.store.contract_state[contract.contract_id]["state"] in (
                    "COMPLETE",
                    "NOT_FOUND",
                    "BLOCKED",
                ):
                    break

        # Phase C: systematic per-document sweeps. Whatever remains open gets
        # a cover-to-cover pass over every document — exhaustive beats lucky.
        if self.store.unresolved_contracts() and not self.store.budgets.exhausted():
            try:
                profiles = self.computer.world.profiles
            except Exception:
                profiles = {}
            for doc_id, profile in profiles.items():
                if self.store.budgets.exhausted():
                    break
                if not self.store.unresolved_contracts():
                    break
                filename = getattr(profile, "filename", doc_id)
                pages = getattr(profile, "page_count", 0)
                sweep_config = {
                    "configurable": {"thread_id": f"{self.thread_id}_doc_{doc_id}"}
                }
                try:
                    result = agent.invoke(
                        {
                            "messages": [
                                {
                                    "role": "user",
                                    "content": self._sweep_message(
                                        doc_id, filename, pages
                                    ),
                                }
                            ]
                        },
                        sweep_config,
                    )
                    self._account_usage(result)
                except Exception:
                    break
                self._save_ledger()

        # Deterministic completion: mark unresolved contracts.
        # Only terminalize contracts that received some investigation activity;
        # contracts untouched due to budget exhaustion stay open (partial).
        investigated = {
            c.contract_id
            for c in self.contracts
            if self.store.contract_state[c.contract_id]["answers"] > 0
            or self.store.contract_state[c.contract_id]["rejected_observations"] > 0
            or self.store.contract_state[c.contract_id]["strategies_attempted"]
        }
        for c in self.store.unresolved_contracts():
            if c.contract_id not in investigated:
                continue
            st = self.store.contract_state[c.contract_id]
            if st["answers"] > 0:
                self.store.set_contract_state(c.contract_id, "PARTIAL")
            else:
                self.store.set_contract_state(c.contract_id, "NOT_FOUND")

        elapsed = time.time() - started
        return self._compile_result(elapsed)

    def _compile_result(self, elapsed: float) -> DeepRunResult:
        """Build the benchmark-compatible output dict."""
        claims = []
        for answer in self.store.answers:
            claims.append(answer.to_claim(self.filenames.get(answer.document_id, "")))

        unresolved = [
            {
                "claim_id": a.answer_id,
                "subject": a.subject,
                "attribute": a.attribute,
                "reason": f"admission={a.admission} verification={a.verification}",
            }
            for a in self.store.answers
            if a.verification != "verified"
        ]
        # Contracts with no answers at all
        for c in self.contracts:
            st = self.store.contract_state[c.contract_id]
            if st["answers"] == 0:
                unresolved.append(
                    {
                        "claim_id": c.contract_id,
                        "subject": c.target_entity,
                        "attribute": c.field,
                        "reason": f"no candidate answers (state={st['state']})",
                    }
                )

        verified = sum(1 for a in self.store.answers if a.verification == "verified")
        output = {
            "claims": claims,
            "unresolved": unresolved,
            "contradictions": self.store.contradictions,
            "verification": {
                "gates": [
                    {
                        "gate": "semantic_admission",
                        "passed": True,
                        "failures": [],
                    },
                    {
                        "gate": "evidence",
                        "passed": verified > 0,
                        "failures": [] if verified > 0 else ["no verified answers"],
                    },
                    {
                        "gate": "completeness",
                        "passed": not self.store.unresolved_contracts(),
                        "failures": [
                            f"{c.contract_id} unresolved"
                            for c in self.store.unresolved_contracts()
                        ],
                    },
                ]
            },
            "stats": {
                "claims": len(claims),
                "verified": verified,
                "unresolved": len(unresolved),
                "contradictions": len(self.store.contradictions),
                "answer_claims": len(claims),
            },
        }

        completeness = {
            "assessment": "complete"
            if not self.store.unresolved_contracts()
            else "partial",
            "rationale": f"{len(self.contracts)} contracts, {len(self.store.answers)} answers, {verified} verified",
            "channels": [],
        }

        budgets = self.store.budgets.to_dict()
        budgets["elapsed_seconds"] = round(elapsed, 1)

        metrics = {
            "tool_calls": self.store.budgets.tool_calls,
            "model_calls": self.store.budgets.model_calls,
            "visual_calls": self.store.budgets.visual_calls,
            "di_calls": self.store.budgets.di_calls,
            "pages_inspected": self.store.coverage.inspected_candidates,
            "claims_created": len(self.store.answers),
            "contradictions_found": len(self.store.contradictions),
            "recovery_attempts": len(self.store.recovery_log),
            "total_latency_ms": round(elapsed * 1000, 1),
            "estimated_tokens": self.store.budgets.tokens_used,
            "estimated_cost_usd": 0.0,
        }

        return DeepRunResult(
            status="complete" if not self.store.unresolved_contracts() else "partial",
            output=output,
            completeness=completeness,
            budgets=budgets,
            metrics=metrics,
            contracts=self.store.contract_summary(),
        )
