"""Investigation planner: goal -> gaps -> candidate tool actions.

Planning queries against the world model are free; only executed tools cost.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from ..core.schemas import Claim, InformationNeed, InvestigationGoal, make_id
from ..models.heuristic import need_search_queries
from ..tools.computer import DocumentComputer
from ..workspace.world_model import WorldModel

MAX_CANDIDATES = 10

_ACTION_COST = {
    "list_documents": 0.2,
    "inspect_document": 0.3,
    "get_document_metadata": 0.2,
    "search_text": 0.5,
    "semantic_search": 0.6,
    "hybrid_search": 0.6,
    "search_headings": 0.4,
    "read_text": 0.6,
    "read_page": 0.7,
    "read_section": 0.7,
    "read_surrounding_context": 0.7,
    "render_page": 1.2,
    "render_region": 1.5,
    "inspect_page_visually": 3.0,
    "inspect_region_visually": 3.0,
    "detect_layout": 0.5,
    "detect_tables": 0.6,
    "extract_table": 0.8,
    "reconstruct_table": 0.9,
    "inspect_table_visually": 3.0,
    "run_ocr": 2.0,
    "ocr_region": 2.0,
    "di_analyze_document": 2.0,
    "vision_verify_claim": 3.0,
    "read_spreadsheet": 0.4,
    "read_sheet_range": 0.7,
    "find_similar_sections": 0.6,
    "find_supporting_evidence": 0.6,
    "find_contradicting_evidence": 0.7,
    "compare_sources": 0.4,
    "verify_claim": 1.0,
    "calculate": 0.2,
    "run_python_analysis": 0.8,
}


@dataclass
class Action:
    action_id: str
    tool: str
    args: Dict[str, Any]
    need_ids: List[str] = field(default_factory=list)
    reason: str = ""
    rationale: str = ""
    prior_gain: float = 0.5
    recovery: bool = False
    target: str = ""

    @property
    def cost(self) -> float:
        return _ACTION_COST.get(self.tool, 1.0)


@dataclass
class Gaps:
    unresolved_needs: List[InformationNeed] = field(default_factory=list)
    open_contradictions: List[Any] = field(default_factory=list)
    unverified_claims: List[Claim] = field(default_factory=list)
    recovery_tasks: List[Any] = field(default_factory=list)
    searched_queries: Set[str] = field(default_factory=set)
    visited_pages: Set[Tuple[str, int]] = field(default_factory=set)
    visited_sheets: Set[Tuple[str, str]] = field(default_factory=set)
    claims_count: int = 0
    # Perception targets for visual escalation: {doc_id, page, bbox, claim_ids, reason}
    vision_targets: List[Dict[str, Any]] = field(default_factory=list)
    tables_detected: Set[str] = field(default_factory=set)


def _need_vision_question(need: InformationNeed) -> str:
    """Need-directed reading instruction for the vision model."""
    attr = need.attribute.strip().lower()
    if attr in ("count", "counts", "io_count"):
        return (
            "Read this page and list every system with its I/O subtype counts "
            "(AI, AO, DI, DO). Emit each fact as its own line: "
            "CLAIM: <system> | <code> | <number>."
        )
    if attr in ("clause", "requirement"):
        return (
            "Read this page and list each contractual clause or requirement. "
            "Emit each as its own line: CLAIM: <topic> | <clause> | <summary>."
        )
    subject = "" if need.subject == "*" else f" for {need.subject}"
    return (
        f"Read this page and extract {need.attribute} facts{subject}. "
        "Emit each as its own line: CLAIM: <subject> | <attribute> | <value>."
    )


class InvestigationPlanner:
    def __init__(
        self, world: WorldModel, computer: DocumentComputer, config=None
    ) -> None:
        self.world = world
        self.computer = computer
        self.config = config
        # Episodic strategy hints: tool names that worked on similar corpora.
        # Fact-free by construction (see memory/episodic.py).
        self.boost_tools: Set[str] = set()
        # Candidate ledger (§6), rebuilt by propose() every round (free).
        self.candidate_ledger: Dict[str, List[Dict[str, Any]]] = {}

    def _di_ready(self) -> bool:
        if self.config is None:
            return False
        from ..tools.azure_di import di_configured

        return di_configured(self.config)

    def _vision_ready(self) -> bool:
        if self.config is None:
            return False
        from ..tools.vision_llm import vision_configured

        return vision_configured(self.config)

    # -- entry ----------------------------------------------------------------
    def propose(self, goal: InvestigationGoal, gaps: Gaps) -> List[Action]:
        # Channels compose with round-robin breadth: recovery first, then each
        # unresolved need gets a fair share. Policy arbitrates; ordering only
        # guarantees breadth (no single need starves the others).
        self.candidate_ledger = self.build_candidates(goal, gaps)
        actions: List[Action] = []
        for task in gaps.recovery_tasks:
            actions.extend(self._from_recovery(task, goal, gaps))
            if len(actions) >= MAX_CANDIDATES:
                return actions[:MAX_CANDIDATES]
        for contra in gaps.open_contradictions:
            actions.extend(self._from_contradiction(contra, gaps))
            if len(actions) >= MAX_CANDIDATES:
                return actions[:MAX_CANDIDATES]
        per_need = [self._from_need(need, gaps) for need in gaps.unresolved_needs]
        index = 0
        while len(actions) < MAX_CANDIDATES:
            progressed = False
            for bucket in per_need:
                if index < len(bucket):
                    actions.append(bucket[index])
                    progressed = True
                    if len(actions) >= MAX_CANDIDATES:
                        return actions[:MAX_CANDIDATES]
            if not progressed:
                break
            index += 1
        if not actions:
            actions.extend(self._discovery(gaps))
        return actions[:MAX_CANDIDATES]

    def build_candidates(
        self, goal: InvestigationGoal, gaps: Gaps
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Candidate regions per need from every free channel (§6).

        Lexical + semantic + hybrid + structural hits, table inventory,
        sheet-cell hits, neighbor-page expansion, and unscanned fallbacks.
        Planning queries are free; only executed tools cost budget.
        """
        ledger: Dict[str, List[Dict[str, Any]]] = {}
        for need in gaps.unresolved_needs:
            candidates: List[Dict[str, Any]] = []
            seen = set()
            queries = list(need_search_queries(need))
            if self._model_expansion_worthwhile(need, gaps):
                queries.extend(self._expanded_queries(need))
            for query in queries[:8]:
                for channel, search in (
                    ("lexical", self.world.search_lexical),
                    ("semantic", self.world.search_semantic),
                    ("hybrid", self.world.search_hybrid),
                ):
                    try:
                        hits = search(query, top_k=6)
                    except Exception:
                        hits = []
                    for hit in hits:
                        key = (hit.provenance_doc, hit.page, hit.cell)
                        if key in seen:
                            continue
                        seen.add(key)
                        candidates.append(
                            {
                                "doc_id": hit.provenance_doc,
                                "page": hit.page,
                                "cell": hit.cell,
                                "channel": channel,
                                "score": hit.score,
                                "query": query,
                            }
                        )
            for heading in (
                self.world.search_headings(need.attribute, top_k=4)
                if hasattr(self.world, "search_headings")
                else []
            ):
                key = (heading.doc_id, heading.page, None)
                if key not in seen:
                    seen.add(key)
                    candidates.append(
                        {
                            "doc_id": heading.doc_id,
                            "page": heading.page,
                            "cell": None,
                            "channel": "structural",
                            "score": heading.score,
                            "query": need.attribute,
                        }
                    )
            candidates.extend(self._table_candidates(need, gaps, seen))
            candidates.extend(self._neighbor_pages(candidates, gaps, seen))
            candidates.extend(self._unscanned_fallback(need, gaps, seen, candidates))
            candidates.sort(key=lambda c: c["score"], reverse=True)
            ledger[need.need_id] = candidates[:24]
        return ledger

    def _model_expansion_worthwhile(self, need: InformationNeed, gaps: Gaps) -> bool:
        return False  # decided per-round in _from_need (budget-aware); kept explicit

    def _expanded_queries(self, need: InformationNeed) -> List[str]:
        terms = need_search_queries(need)[:3]
        try:
            return (
                self.computer.gateway.expand_queries(terms)
                if hasattr(self.computer, "gateway") and self.computer.gateway
                else []
            )
        except Exception:
            return []

    def _table_candidates(
        self, need: InformationNeed, gaps: Gaps, seen: set
    ) -> List[Dict[str, Any]]:
        out = []
        for doc_id, profile in self.world.profiles.items():
            if profile.table_candidates <= 0 or doc_id in gaps.tables_detected:
                continue
            key = (doc_id, None, None)
            if key in seen:
                continue
            seen.add(key)
            out.append(
                {
                    "doc_id": doc_id,
                    "page": None,
                    "cell": None,
                    "channel": "table",
                    "score": 0.7,
                    "query": need.attribute,
                }
            )
        return out

    def _neighbor_pages(
        self, candidates: List[Dict[str, Any]], gaps: Gaps, seen: set
    ) -> List[Dict[str, Any]]:
        out = []
        for cand in list(candidates):
            page, doc_id = cand.get("page"), cand.get("doc_id", "")
            if page is None:
                continue
            profile = self.world.profiles.get(doc_id)
            if profile is None:
                continue
            for neighbor in (page - 1, page + 1):
                if not 1 <= neighbor <= profile.page_count:
                    continue
                key = (doc_id, neighbor, None)
                if key in seen or key in gaps.visited_pages:
                    continue
                seen.add(key)
                out.append(
                    {
                        "doc_id": doc_id,
                        "page": neighbor,
                        "cell": None,
                        "channel": "neighbor",
                        "score": cand["score"] * 0.5,
                        "query": cand.get("query", ""),
                    }
                )
        return out

    def _unscanned_fallback(
        self,
        need: InformationNeed,
        gaps: Gaps,
        seen: set,
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if any(c.get("page") is not None or c.get("cell") for c in candidates):
            return []
        out = []
        for doc_id, profile in self.world.profiles.items():
            if not profile.needs_ocr:
                continue
            for page_no in range(1, min(profile.page_count, 2) + 1):
                key = (doc_id, page_no, None)
                if key in seen or key in gaps.visited_pages:
                    continue
                seen.add(key)
                out.append(
                    {
                        "doc_id": doc_id,
                        "page": page_no,
                        "cell": None,
                        "channel": "perception",
                        "score": 0.8,
                        "query": need.attribute,
                    }
                )
                if len(out) >= 4:
                    return out
        return out

    def _act(
        self,
        tool: str,
        args: Dict[str, Any],
        need_ids: List[str],
        reason: str,
        gain: float,
        recovery: bool = False,
        target: str = "",
    ) -> Action:
        if tool in self.boost_tools:
            gain += 0.1  # episodic strategy hint, never factual evidence
        return Action(
            action_id=make_id("act"),
            tool=tool,
            args=args,
            need_ids=need_ids,
            reason=reason,
            prior_gain=gain,
            recovery=recovery,
            target=target or str(sorted(args.items())),
        )

    # -- recovery mapping --------------------------------------------------------
    def _from_recovery(self, task, goal: InvestigationGoal, gaps: Gaps) -> List[Action]:
        actions: List[Action] = []
        claim_ids = list(getattr(task, "claim_ids", []))
        # Explicit read targets attached by gates (e.g. coverage) win.
        for read in (getattr(task, "context", None) or {}).get("reads", [])[:4]:
            doc_id = read.get("doc_id", "")
            # Perception routing (§7): unscanned candidates escalate to DI or
            # Vision directly — eligibility never depends on prior claims.
            if read.get("via") == "perception":
                profile = self.world.profiles.get(doc_id)
                if profile is not None and profile.needs_ocr:
                    if self._di_ready():
                        actions.append(
                            self._act(
                                "di_analyze_document",
                                {"doc_id": doc_id, "pages": [read.get("page") or 1]},
                                [],
                                f"recovery {task.task_id}: DI on unscanned page",
                                0.9,
                                recovery=True,
                                target=f"di:{doc_id}",
                            )
                        )
                        continue
                    if self._vision_ready():
                        actions.append(
                            self._act(
                                "inspect_page_visually",
                                {
                                    "doc_id": doc_id,
                                    "page": read.get("page") or 1,
                                    "question": "Read this page precisely and extract "
                                    "all factual records as CLAIM lines.",
                                },
                                [],
                                f"recovery {task.task_id}: vision on unscanned page",
                                0.9,
                                recovery=True,
                                target=f"vpage:{doc_id}",
                            )
                        )
                        continue
            if read.get("cell") and not read.get("page"):
                actions.append(
                    self._act(
                        "read_sheet_range",
                        {
                            "doc_id": doc_id,
                            "sheet": read["cell"].split("!")[0],
                            "cell_range": "A1:Z30",
                        },
                        [],
                        f"recovery {task.task_id}: read unexamined sheet location",
                        0.85,
                        recovery=True,
                        target=f"sheet:{doc_id}:{read['cell']}",
                    )
                )
            elif read.get("page"):
                actions.append(
                    self._act(
                        "read_page",
                        {"doc_id": doc_id, "page": read["page"]},
                        [],
                        f"recovery {task.task_id}: read unexamined page",
                        0.85,
                        recovery=True,
                        target=f"read:{doc_id}:{read['page']}",
                    )
                )
        if actions:
            return actions[:4]
        for suggestion in getattr(task, "suggested_actions", [])[:3]:
            if suggestion in (
                "search_corroboration",
                "search_subject_candidates",
                "expand_retrieval",
                "find_supporting_evidence",
            ):
                query = " ".join(claim_ids) or " ".join(
                    n.attribute for n in goal.needs[:2]
                )
                if not query or not self.world.search_hybrid(query, top_k=8):
                    continue  # free check proves barren; other channels instead
                for mode in ("lexical", "hybrid"):
                    key = f"{mode}:{query}"
                    actions.append(
                        self._act(
                            "search_text",
                            {"query": query or "summary", "top_k": 8, "mode": mode},
                            [],
                            f"recovery {task.task_id}: corroborate via {mode}",
                            0.75,
                            recovery=True,
                            target=key,
                        )
                    )
            elif suggestion in (
                "read_candidate_pages",
                "inspect_original_region",
                "inspect_component_rows",
            ):
                actions.extend(self._read_targets_for_claims(claim_ids, task))
            elif suggestion in (
                "compare_sources",
                "resolve_contradiction",
                "inspect_revisions",
                "search_related_schedule",
            ):
                query = " ".join(claim_ids) or "schedule revision"
                actions.append(
                    self._act(
                        "find_contradicting_evidence",
                        {"query": query, "top_k": 8},
                        [],
                        f"recovery {task.task_id}: gather conflicting sources",
                        0.8,
                        recovery=True,
                        target=f"contra:{query}",
                    )
                )
            elif suggestion in ("inspect_tables", "extract_table"):
                for doc_id in self._candidate_docs(goal)[:2]:
                    actions.append(
                        self._act(
                            "detect_tables",
                            {"doc_id": doc_id},
                            [],
                            f"recovery {task.task_id}: re-scan table candidates",
                            0.7,
                            recovery=True,
                            target=f"tables:{doc_id}",
                        )
                    )
            elif suggestion in (
                "render_region",
                "ocr_region",
                "inspect_region_visually",
                "inspect_original_region",
            ):
                actions.extend(
                    self._sense_targets_for_claims(claim_ids, task, suggestion, gaps)
                )
            elif suggestion == "recompute_total":
                actions.append(
                    self._act(
                        "calculate",
                        {"expression": "0"},
                        [],
                        f"recovery {task.task_id}: placeholder recompute (agent fills operands)",
                        0.3,
                        recovery=True,
                        target=f"calc:{task.task_id}",
                    )
                )
            if len(actions) >= 4:
                break
        return actions[:4]

    def _read_targets_for_claims(self, claim_ids: List[str], task) -> List[Action]:
        # Planner has no store access; re-derive targets from the world index.
        actions: List[Action] = []
        query = " ".join(claim_ids)
        hits = self.world.search_hybrid(query or "summary", top_k=4) if query else []
        for hit in hits[:2]:
            if hit.page is not None:
                actions.append(
                    self._act(
                        "read_page",
                        {"doc_id": hit.provenance_doc, "page": hit.page},
                        [],
                        f"recovery {task.task_id}: inspect cited page",
                        0.8,
                        recovery=True,
                        target=f"read:{hit.provenance_doc}:{hit.page}",
                    )
                )
        return actions

    def _sense_targets_for_claims(
        self, claim_ids: List[str], task, suggestion: str, gaps: Gaps
    ) -> List[Action]:
        """Perception ladder for one region: DI first, then targeted vision."""
        actions: List[Action] = []
        # Agent-supplied vision targets (contradicted/critical regions) win.
        for target in gaps.vision_targets[:2]:
            doc_id = target.get("doc_id", "")
            if not doc_id:
                continue
            if self._vision_ready() and suggestion in (
                "inspect_region_visually",
                "inspect_original_region",
                "render_region",
            ):
                args: Dict[str, Any] = {
                    "doc_id": doc_id,
                    "page": target.get("page", 1),
                    "question": target.get("reason", "read this region precisely"),
                }
                if target.get("bbox"):
                    args["bbox"] = target["bbox"]
                    actions.append(
                        self._act(
                            "inspect_region_visually",
                            args,
                            [],
                            f"recovery {task.task_id}: targeted vision on disputed region",
                            0.9,
                            recovery=True,
                            target=f"vregion:{doc_id}:{args['page']}",
                        )
                    )
                else:
                    actions.append(
                        self._act(
                            "inspect_page_visually",
                            args,
                            [],
                            f"recovery {task.task_id}: targeted vision on disputed page",
                            0.85,
                            recovery=True,
                            target=f"vpage:{doc_id}:{args['page']}",
                        )
                    )
            elif self._di_ready() and suggestion in (
                "ocr_region",
                "run_ocr",
                "inspect_original_region",
            ):
                actions.append(
                    self._act(
                        "di_analyze_document",
                        {"doc_id": doc_id, "pages": [target.get("page", 1)]},
                        [],
                        f"recovery {task.task_id}: DI layout/OCR on disputed page",
                        0.85,
                        recovery=True,
                        target=f"di:{doc_id}",
                    )
                )
        if actions:
            return actions[:2]
        tool = {"render_region": "render_page", "ocr_region": "run_ocr"}.get(
            suggestion, "render_page"
        )
        # Ladder: prefer Azure DI over the legacy local-OCR stub when configured.
        if self._di_ready() and tool == "run_ocr":
            tool = "di_analyze_document"
        actions: List[Action] = []
        for doc_id, profile in self.world.profiles.items():
            if profile.needs_ocr or profile.classification in ("scanned", "hybrid"):
                if tool == "di_analyze_document":
                    args = {"doc_id": doc_id, "pages": [1]}
                    target = f"di:{doc_id}:1"
                    reason = f"recovery {task.task_id}: DI on unscanned page"
                else:
                    args = {"doc_id": doc_id, "page": 1}
                    target = f"{tool}:{doc_id}:1"
                    reason = f"recovery {task.task_id}: sense visual page"
                actions.append(
                    self._act(
                        tool,
                        args,
                        [],
                        reason,
                        0.6,
                        recovery=True,
                        target=target,
                    )
                )
                if len(actions) >= 2:
                    break
        return actions

    # -- contradictions / needs / discovery ---------------------------------------
    def _candidate_action(
        self, need: InformationNeed, cand: Dict[str, Any], gaps: Gaps
    ) -> Optional[Action]:
        """One ledger candidate -> the cheapest tool that inspects it."""
        doc_id = cand.get("doc_id", "")
        channel = cand.get("channel", "")
        if channel == "table":
            if doc_id in gaps.tables_detected:
                return None
            return self._act(
                "detect_tables",
                {"doc_id": doc_id},
                [need.need_id],
                f"need {need.need_id}: inventory tables in {doc_id}",
                0.8,
                target=f"tables:{doc_id}",
            )
        if cand.get("cell") and cand.get("page") is None:
            if (doc_id, cand["cell"]) in gaps.visited_sheets:
                return None
            return self._act(
                "read_sheet_range",
                {
                    "doc_id": doc_id,
                    "sheet": cand["cell"].split("!")[0],
                    "cell_range": "A1:Z30",
                },
                [need.need_id],
                f"need {need.need_id}: inspect sheet hit {cand['cell']}",
                0.8,
                target=f"sheet:{doc_id}:{cand['cell']}",
            )
        page = cand.get("page")
        if page is not None:
            if (doc_id, page) in gaps.visited_pages:
                return None
            if channel == "perception":
                profile = self.world.profiles.get(doc_id)
                needs_ocr = profile.needs_ocr if profile else True
                if needs_ocr and self._di_ready():
                    return self._act(
                        "di_analyze_document",
                        {"doc_id": doc_id, "pages": [page]},
                        [need.need_id],
                        f"need {need.need_id}: DI on unscanned page {page}",
                        0.9,
                        target=f"di:{doc_id}:{page}",
                    )
                if needs_ocr and self._vision_ready():
                    return self._act(
                        "inspect_page_visually",
                        {
                            "doc_id": doc_id,
                            "page": page,
                            "question": _need_vision_question(need),
                        },
                        [need.need_id],
                        f"need {need.need_id}: vision-read unscanned page {page}",
                        1.1,
                        target=f"vpage:{doc_id}:{page}",
                    )
            return self._act(
                "read_page",
                {"doc_id": doc_id, "page": page},
                [need.need_id],
                f"need {need.need_id}: read candidate page {page} ({channel})",
                0.8,
                target=f"read:{doc_id}:{page}",
            )
        return None

    def _from_contradiction(self, contra, gaps: Gaps) -> List[Action]:
        subject = getattr(contra, "subject", "")
        attribute = getattr(contra, "attribute", "")
        query = f"{subject} {attribute}".strip() or "revision"
        actions = [
            self._act(
                "find_contradicting_evidence",
                {"query": query, "top_k": 8},
                [],
                f"contradiction {contra.contradiction_id}: gather both sources",
                0.85,
                target=f"contra:{query}",
            )
        ]
        # Escalate disputed regions to targeted vision when configured.
        if self._vision_ready():
            for target in gaps.vision_targets[:2]:
                if set(getattr(contra, "claim_ids", [])) & set(
                    target.get("claim_ids", [])
                ):
                    args: Dict[str, Any] = {
                        "doc_id": target["doc_id"],
                        "page": target.get("page", 1),
                        "question": (
                            f"Resolve this conflict: {subject}/{attribute} is claimed "
                            f"as {getattr(contra, 'values', [])}. Read the exact "
                            "values visible in this region."
                        ),
                    }
                    if target.get("bbox"):
                        args["bbox"] = target["bbox"]
                        actions.append(
                            self._act(
                                "inspect_region_visually",
                                args,
                                [],
                                f"contradiction {contra.contradiction_id}: vision on region",
                                0.9,
                                target=f"vregion:{target['doc_id']}",
                            )
                        )
                    else:
                        actions.append(
                            self._act(
                                "inspect_page_visually",
                                args,
                                [],
                                f"contradiction {contra.contradiction_id}: vision on page",
                                0.85,
                                target=f"vpage:{target['doc_id']}",
                            )
                        )
        return actions

    def _from_need(self, need: InformationNeed, gaps: Gaps) -> List[Action]:
        actions: List[Action] = []
        searches = 0
        for query in need_search_queries(need):
            for mode in ("lexical", "hybrid"):
                if searches >= 2:  # breadth first: leave room for reads
                    break
                key = f"{mode}:{query}"
                if key in gaps.searched_queries:
                    continue
                # Free pre-check: never spend tool budget on queries the world
                # model already proves barren — escalate to perception instead.
                if not self.world.search_hybrid(query, top_k=8):
                    continue
                actions.append(
                    self._act(
                        "search_text",
                        {"query": query, "top_k": 8, "mode": mode},
                        [need.need_id],
                        f"need {need.need_id}: {mode} search '{query}'",
                        0.7,
                        target=key,
                    )
                )
                searches += 1
        # Candidate ledger (§6): ranked regions across all free channels.
        for cand in getattr(self, "candidate_ledger", {}).get(need.need_id, [])[:6]:
            action = self._candidate_action(need, cand, gaps)
            if action is not None:
                actions.append(action)
            if len(actions) >= 6:
                return actions
        # Perception ladder: unscanned/image docs escalate to Azure DI (targeted).
        if self._di_ready():
            di_pages = self._di_candidate_pages(need, gaps)
            for doc_id, page in di_pages[:3]:
                actions.append(
                    self._act(
                        "di_analyze_document",
                        {"doc_id": doc_id, "pages": [page]},
                        [need.need_id],
                        f"need {need.need_id}: DI layout/OCR on unscanned page {page}",
                        0.85,
                        target=f"di:{doc_id}:{page}",
                    )
                )
                if len(actions) >= 8:
                    return actions
        # ... or straight to GPT-5.4 Vision when DI is not configured.
        elif self._vision_ready():
            for doc_id, page in self._unscanned_pages(gaps)[:2]:
                actions.append(
                    self._act(
                        "inspect_page_visually",
                        {
                            "doc_id": doc_id,
                            "page": page,
                            "question": _need_vision_question(need),
                        },
                        [need.need_id],
                        f"need {need.need_id}: vision-read unscanned page {page}",
                        1.2,
                        target=f"vpage:{doc_id}:{page}",
                    )
                )
                if len(actions) >= 8:
                    return actions
        # Last resort before giving up: read native pages directly. Bounded.
        for doc_id, page in self._unread_pages(gaps)[:3]:
            actions.append(
                self._act(
                    "read_page",
                    {"doc_id": doc_id, "page": page},
                    [need.need_id],
                    f"need {need.need_id}: direct read of unvisited page {page}",
                    0.75,
                    target=f"read:{doc_id}:{page}",
                )
            )
            if len(actions) >= 8:
                return actions
        # Table inventory: profiled table candidates the agent never inspected.
        for doc_id, profile in self.world.profiles.items():
            if profile.table_candidates <= 0 or doc_id in gaps.tables_detected:
                continue
            actions.append(
                self._act(
                    "detect_tables",
                    {"doc_id": doc_id},
                    [need.need_id],
                    f"need {need.need_id}: inventory tables in {doc_id}",
                    0.8,
                    target=f"tables:{doc_id}",
                )
            )
            if len(actions) >= 8:
                return actions
        return actions

    def _unread_pages(self, gaps: Gaps) -> List[Tuple[str, int]]:
        """Readable pages never opened (last-resort direct reads)."""
        out: List[Tuple[str, int]] = []
        for doc_id, profile in self.world.profiles.items():
            if profile.classification not in (
                "selectable",
                "text",
                "hybrid",
                "spreadsheet",
            ):
                continue
            for page_no in range(1, profile.page_count + 1):
                key = (doc_id, page_no)
                if key in gaps.visited_pages:
                    continue
                out.append(key)
                if len(out) >= 4:
                    return out
        return out

    def _unscanned_pages(self, gaps: Gaps) -> List[Tuple[str, int]]:
        """First unvisited pages of docs with no usable native text layer."""
        out: List[Tuple[str, int]] = []
        for doc_id, profile in self.world.profiles.items():
            if not profile.needs_ocr:
                continue
            for page_no in range(1, profile.page_count + 1):
                key = (doc_id, page_no)
                if key in gaps.visited_pages:
                    continue
                out.append(key)
                if len(out) >= 4:
                    return out
        return out

    def _di_candidate_pages(
        self, need: InformationNeed, gaps: Gaps
    ) -> List[Tuple[str, int]]:
        """Unvisited pages of unscanned docs that lexically match the need."""
        out: List[Tuple[str, int]] = []
        seen = set()
        for query in need_search_queries(need):
            for hit in self.world.lexical.search(query, top_k=20):
                if hit.page is None:
                    continue
                profile = self.world.profiles.get(hit.provenance_doc)
                if profile is None or not profile.needs_ocr:
                    continue
                key = (hit.provenance_doc, hit.page)
                if key in gaps.visited_pages or key in seen:
                    continue
                seen.add(key)
                out.append(key)
        # Fully unscanned documents have no index presence at all: escalate
        # their first unvisited pages directly (never blindly the whole doc).
        for doc_id, profile in self.world.profiles.items():
            if not profile.needs_ocr or profile.classification not in (
                "scanned",
                "image",
                "hybrid",
            ):
                continue
            for page_no in range(1, min(profile.page_count, 2) + 1):
                key = (doc_id, page_no)
                if key in gaps.visited_pages or key in seen:
                    continue
                seen.add(key)
                out.append(key)
                if len(out) >= 4:
                    return out
        return out

    def _candidate_docs(self, goal: InvestigationGoal) -> List[str]:
        terms = [n.attribute for n in goal.needs]
        scored = []
        for doc_id, profile in self.world.profiles.items():
            hits = sum(len(self.world.search_lexical(t, top_k=3)) for t in terms[:2])
            scored.append((hits, doc_id))
        scored.sort(reverse=True)
        return [doc_id for _, doc_id in scored] or list(self.world.profiles)

    def _discovery(self, gaps: Gaps) -> List[Action]:
        actions = [
            self._act("list_documents", {}, [], "map the corpus", 0.6, target="list")
        ]
        for doc_id in list(self.world.profiles)[:3]:
            actions.append(
                self._act(
                    "inspect_document",
                    {"doc_id": doc_id},
                    [],
                    f"profile {doc_id}",
                    0.55,
                    target=f"inspect:{doc_id}",
                )
            )
        return actions
