"""HarnessService: UI-independent boundary over the Deep Agents runtime.

Owns the workspace (staged corpus + world model), conversational Ask runs
(normal prompts — NEVER a workbook), status/evidence/trace/coverage views,
targeted retry and explicit exports. The TUI and scriptable CLI share this.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ...core.config import RuntimeConfig
from ...deep.agent import DeepAgentInvestigator
from ...deep.store import RunStore
from ...tools.computer import DocumentComputer
from ...workspace.corpus import Corpus, SUPPORTED_EXTS, fingerprint_file
from ...workspace.world_model import WorldModel
from .events import (
    RUN_COMPLETED,
    RUN_STARTED,
    EventBus,
)


def _full_sweep_hint_enabled() -> bool:
    """RC-10: allow an exhaustive page sweep for enumeration-style tasks.

    The default strategy prompt says "NEVER try to read the entire document",
    which is right for search-shaped questions and wrong for documents whose
    answer is *every row of a list spread across all pages*: the agent behind
    ``real_ofac_ssi_full`` read one page and returned 11 of 1238 entities.

    The clause is OPT-IN because prompt edits have a measured cost in this
    codebase — two variants were rejected in RC-7 (-0.059 across five sf1449
    documents; ``sf1449_supplies_services_0047`` fell to 0.1304 in 3/3 reps).
    With the flag unset the prompt is byte-identical to before, which the tests
    pin; ``ADH_FULL_SWEEP_HINT=1`` enables it for an A/B measurement.
    """
    return os.environ.get("ADH_FULL_SWEEP_HINT", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _strategy_intro() -> str:
    """Opening line of the investigation strategy block."""
    if _full_sweep_hint_enabled():
        return (
            "NEVER try to read the entire document — EXCEPT when the task asks "
            "for EVERY record of a list/table that spans many pages (a "
            "registry, a schedule, an enumeration): then sweep ALL pages before "
            "answering. "
        )
    return "NEVER try to read the entire document. "


def _qa_budget_limits() -> dict:
    """Tool/model/time limits for the investigation computer.

    These values were hardcoded; they are now tunable per deployment
    (``ADH_QA_MAX_TOOL_CALLS``, ``ADH_QA_MAX_MODEL_CALLS``,
    ``ADH_QA_MAX_VISUAL_CALLS``, ``ADH_QA_TIMEOUT_SECONDS``) with the ORIGINAL
    numbers as defaults, so nothing changes unless an operator asks for it.
    Measured D3 usage sits far below them (highest observed: 42 tool calls,
    376 s), so raising them is headroom — never a scoring change.
    """

    def _limit(name: str, default):
        raw = os.environ.get(name, "").strip()
        if not raw:
            return default
        try:
            value = type(default)(raw)
        except ValueError:
            return default
        return value if value > 0 else default

    return {
        "max_tool_calls": _limit("ADH_QA_MAX_TOOL_CALLS", 120),
        "max_model_calls": _limit("ADH_QA_MAX_MODEL_CALLS", 35),
        "max_visual_calls": _limit("ADH_QA_MAX_VISUAL_CALLS", 20),
        "timeout_seconds": _limit("ADH_QA_TIMEOUT_SECONDS", 300.0),
    }


class HarnessService:
    """One service instance per session."""

    def __init__(
        self,
        session: Dict[str, Any],
        data_dir: Optional[Path] = None,
        bus: Optional[EventBus] = None,
    ) -> None:
        from ...deep.model import ensure_azure_env

        ensure_azure_env()
        self.session = session
        self.bus = bus or EventBus()
        self.data_dir = Path(data_dir) if data_dir else Path.home() / ".harness"
        self.workspace_dir = self.data_dir / "workspaces" / session["id"]
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.config = RuntimeConfig.from_env("")
        self._world: Optional[WorldModel] = None
        self._world_key: str = ""
        self._store: Optional[RunStore] = None
        self._computer: Optional[DocumentComputer] = None
        self._runs = 0
        self._chat_model_ref: Any = None
        self._qa_tokens: int = 0
        self._qa_model_calls: int = 0
        self._qa_tool_calls: int = 0

    # -- workspace ------------------------------------------------------
    def add_documents(
        self, paths: List[str], on_progress=None
    ) -> Dict[str, Any]:
        """Stage files/dirs into the workspace (copies; originals untouched).

        *on_progress*: optional ``(current, total, filename)`` callback for
        live UI feedback while staging files.
        """
        staged, skipped, errors = [], [], []
        all_files: List[Path] = []
        for raw in paths:
            path = Path(raw).expanduser()
            if not path.exists():
                errors.append(f"not found: {raw}")
                continue
            if path.is_file():
                all_files.append(path)
            else:
                all_files.extend(sorted(p for p in path.rglob("*") if p.is_file()))
        total = len(all_files)
        for idx, source in enumerate(all_files):
            if on_progress is not None:
                try:
                    on_progress(idx + 1, total, source.name)
                except Exception:
                    pass
            if source.suffix.lower() not in SUPPORTED_EXTS:
                skipped.append(f"{source.name} (unsupported type)")
                continue
            try:
                digest = fingerprint_file(source)
            except OSError as exc:
                errors.append(f"{source.name}: {exc}")
                continue
            if any(d.get("sha256") == digest for d in self.session["documents"]):
                skipped.append(f"{source.name} (duplicate)")
                continue
            target = self.workspace_dir / source.name
            counter = 1
            while target.exists():
                counter += 1
                target = (
                    self.workspace_dir / f"{source.stem}_{counter}{source.suffix}"
                )
            shutil.copy(source, target)
            self.session["documents"].append(
                {
                    "filename": source.name,
                    "staged_as": target.name,
                    "source": str(source),
                    "sha256": digest,
                    "bytes": target.stat().st_size,
                }
            )
            staged.append(source.name)
        self._world = None  # rebuild lazily
        return {
            "staged": staged,
            "skipped": skipped,
            "errors": errors,
            "documents": self.session["documents"],
        }

    def remove_document(self, filename: str) -> bool:
        docs = self.session["documents"]
        for entry in list(docs):
            if entry["filename"] == filename or entry.get("staged_as") == filename:
                docs.remove(entry)
                staged = self.workspace_dir / entry.get("staged_as", "")
                if staged.exists():
                    staged.unlink()  # staged copy only; original never touched
                self._world = None
                return True
        return False

    def list_documents(self) -> List[Dict[str, Any]]:
        return list(self.session["documents"])

    def ensure_world(self) -> WorldModel:
        key = "|".join(sorted(d.get("sha256", "") for d in self.session["documents"]))
        if self._world is None or self._world_key != key:
            self._world = WorldModel.build(Corpus.load(self.workspace_dir))
            self._world_key = key
        return self._world

    def workspace_stats(self) -> Dict[str, Any]:
        try:
            world = self.ensure_world()
            stats = dict(world.corpus_stats)
        except Exception:
            stats = {
                "total_documents": len(self.session["documents"]),
                "total_pages": 0,
            }
        stats["staged_files"] = len(self.session["documents"])
        return stats

    def _event_sink(self, tool_name: str, phase: str, payload: Dict[str, Any]) -> None:
        """Forward registry tool events into the normalized bus (sanitized)."""
        from ..sanitize import sanitize_text
        from .events import tool_event_to_kind

        try:
            kind, title, detail = tool_event_to_kind(tool_name, phase, payload)
        except Exception:
            return
        self.bus.post(
            kind,
            sanitize_text(title, 160),
            sanitize_text(detail, 300),
            {"tool": tool_name, "phase": phase},
        )

    # -- ask (normal prompts; NEVER a workbook) --------------------------
    # A message is small talk when EVERY word is greeting/social filler —
    # any content word (pump, controller...) forces investigation.
    _SMALL_TALK_WORDS = frozenset(
        {
            "hi",
            "hii",
            "hiya",
            "hello",
            "hey",
            "yo",
            "sup",
            "good",
            "morning",
            "afternoon",
            "evening",
            "day",
            "thanks",
            "thank",
            "you",
            "thx",
            "ty",
            "bye",
            "goodbye",
            "see",
            "ya",
            "how",
            "are",
            "u",
            "what's",
            "whats",
            "what",
            "up",
            "who",
            "nice",
            "cool",
            "ok",
            "okay",
            "great",
            "it",
            "going",
            "doing",
            "there",
            "do",
            "can",
            "i",
            "me",
            "my",
            "the",
            "a",
            "an",
            "is",
            "am",
            "and",
            "so",
            "hows",
            "its",
            "that",
            "this",
            "man",
            "bro",
        }
    )

    def _is_small_talk(self, prompt: str) -> bool:
        text = prompt.strip()
        if not text or len(text.split()) > 6:
            return False
        words = [w.lower().strip("!?..,:;\"'()[]") for w in text.split()]
        return bool(words) and all(w in self._SMALL_TALK_WORDS for w in words)

    def _route_prompt(self, prompt: str) -> tuple:
        """Classify a prompt: ("conversational", reply) or ("document", "").

        Fast path: obvious small-talk word set. Everything else gets ONE
        cheap LLM router call — slang, jokes, and venting ("hi nigga",
        "wtf bro") classify as conversational no matter the phrasing.
        Offline/model-error defaults to document task (previous behavior).
        """
        if self._is_small_talk(prompt):
            return (
                "conversational",
                self._llm_reply(
                    prompt,
                    fallback=(
                        "Hey! I'm KYNEX — I investigate documents and answer "
                        "with evidence. /add <file-or-folder> to load documents, "
                        "then just ask."
                    ),
                ),
            )
        model = self._chat_model()
        if model is None:
            return "document", ""
        from ..sanitize import sanitize_text, scrub_secrets

        try:
            response = model.invoke(
                [
                    {
                        "role": "system",
                        "content": (
                            "You are KYNEX, an autonomous document-intelligence "
                            "terminal assistant. Classify the user message. "
                            + (
                                f"The user has {len(self.session.get('documents', []))} "
                                "document(s) loaded in the workspace. "
                                if self.session.get("documents")
                                else ""
                            )
                            + 'Reply ONLY JSON: {"type": "conversational" | '
                            '"document_task", "reply": "<only when conversational: '
                            'a brief, natural, friendly reply>"}. '
                            "conversational = ONLY greetings ('hi', 'hello'), "
                            "thanks ('thanks', 'ty'), identity questions "
                            "('who are you'), or pure jokes/venting with zero "
                            "document intent. "
                            "document_task = EVERYTHING else when documents are "
                            "loaded: questions, follow-ups, requests, summaries, "
                            "breakdowns, counts, analysis, vague prompts like "
                            "'do it', 'how many', 'okay then', 'show me' — all "
                            "of these are document_task because the user expects "
                            "answers from their documents. When in doubt and "
                            "documents are loaded, choose document_task. "
                            "Conversational replies stay friendly and 1-3 sentences."
                        ),
                    },
                    {"role": "user", "content": scrub_secrets(prompt)[:800]},
                ]
            )
            from ...models.json_utils import parse_json_response

            payload, _ = parse_json_response(str(response.content or ""))
            if isinstance(payload, dict) and payload.get("type") == "conversational":
                reply = sanitize_text(str(payload.get("reply", "")), 800)
                return "conversational", reply or self._llm_reply(
                    prompt, fallback="Hey! How can I help?"
                )
        except Exception:
            pass
        return "document", ""

    def _chat_model(self):
        """Lazy chat model for conversational composition (never extraction)."""
        if self._chat_model_ref is None:
            from ...deep.model import build_tracked_chat_model

            self._chat_model_ref = build_tracked_chat_model(max_tokens=600)
        return self._chat_model_ref

    def _llm_reply(self, instruction: str, fallback: str) -> str:
        """One short conversational LLM call; deterministic fallback offline."""
        from ..sanitize import sanitize_text, scrub_secrets

        model = self._chat_model()
        if model is None:
            return fallback
        try:
            response = model.invoke(
                [
                    {
                        "role": "system",
                        "content": (
                            "You are KYNEX, an autonomous document-intelligence "
                            "terminal assistant (like Claude Code for documents). "
                            "Reply briefly and naturally (1-3 sentences). "
                            + (
                                f"The user has {len(self.session.get('documents', []))} "
                                "document(s) loaded. Answer about the loaded documents. "
                                if self.session.get("documents")
                                else "The user has no documents loaded yet. "
                                "Available: /add <file-or-folder> to load documents. "
                            )
                            + "Never invent document content."
                        ),
                    },
                    {"role": "user", "content": scrub_secrets(instruction)[:800]},
                ]
            )
            text = str(response.content or "").strip()
            return sanitize_text(text, 800) or fallback
        except Exception:
            return fallback

    # ---- Investigative Q&A (tool-calling agent) ----------------------------

    def _ensure_computer(self):
        """Lazy DocumentComputer for tool-based investigation."""
        if self._computer is not None:
            return self._computer
        from ...core.schemas import BudgetLimits, BudgetUsage

        world = self.ensure_world()
        budgets = BudgetUsage(BudgetLimits(**_qa_budget_limits()))
        self._computer = DocumentComputer(world, budgets, config=self.config)
        return self._computer

    def _build_toc(self, computer) -> str:
        """Build a Table-of-Contents string from headings and table spans."""
        toc_lines: list[str] = []
        for doc_id, profile in computer.world.profiles.items():
            parsed = computer.world.parsed.get(doc_id)
            if not parsed:
                continue
            fname = profile.filename
            toc_lines.append(f"\n=== {fname} ({profile.page_count} pages) ===")
            table_pages: list[int] = []
            for page in parsed.pages:
                for heading in page.headings:
                    toc_lines.append(f"  p.{page.page_no}: {heading}")
                if page.tables:
                    table_pages.append(page.page_no)
            if table_pages:
                ranges = self._compress_page_ranges(table_pages)
                toc_lines.append(f"  [Tables on pages: {ranges}]")
        return "\n".join(toc_lines) if toc_lines else "(no headings found)"

    @staticmethod
    def _compress_page_ranges(pages: list[int]) -> str:
        if not pages:
            return ""
        pages = sorted(set(pages))
        ranges: list[str] = []
        start = prev = pages[0]
        for p in pages[1:]:
            if p == prev + 1:
                prev = p
            else:
                ranges.append(f"{start}-{prev}" if prev > start else str(start))
                start = prev = p
        ranges.append(f"{start}-{prev}" if prev > start else str(start))
        return ", ".join(ranges)

    def _build_qa_tools(self, computer):
        """Create LangChain tool functions from DocumentComputer for Q&A."""
        from langchain_core.tools import tool as lc_tool

        doc_ids = list(computer.world.parsed.keys())
        primary = doc_ids[0] if doc_ids else ""

        doc_name_map: dict = {}
        for did in doc_ids:
            profile = computer.world.profiles.get(did)
            if profile:
                doc_name_map[profile.filename.lower()] = did
                stem = profile.filename.rsplit(".", 1)[0].lower()
                doc_name_map[stem] = did

        def _resolve_doc(page_number: int, doc_name: str = "") -> str:
            if doc_name:
                needle = doc_name.strip().lower()
                if needle in doc_name_map:
                    return doc_name_map[needle]
                for key, did in doc_name_map.items():
                    if needle in key or key in needle:
                        return did
            for did in doc_ids:
                parsed = computer.world.parsed.get(did)
                if parsed and any(p.page_no == page_number for p in parsed.pages):
                    return did
            return primary

        def _doc_filename(doc_id: str) -> str:
            profile = computer.world.profiles.get(doc_id)
            return profile.filename if profile else doc_id

        @lc_tool
        def search_document(query: str) -> str:
            """Search all loaded documents for text matching the query. Returns relevant snippets with page numbers and document names."""
            obs = computer.registry.call("search_text", query=query, top_k=15)
            data = obs.data
            for hit in data.get("hits", []):
                prov = hit.get("provenance", {})
                did = prov.get("document_id", "")
                if did:
                    prov["filename"] = _doc_filename(did)
            return json.dumps(data, default=str)

        @lc_tool
        def read_page(page_number: int, doc_name: str = "") -> str:
            """Read the full text and structure of a specific page. Use doc_name to target a specific document when multiple are loaded (e.g. doc_name='IO LIST.pdf')."""
            did = _resolve_doc(page_number, doc_name)
            obs = computer.registry.call("read_page", doc_id=did, page=page_number)
            data = obs.data
            data["filename"] = _doc_filename(did)
            return json.dumps(data, default=str)

        @lc_tool
        def read_nearby_pages(page_number: int, doc_name: str = "") -> str:
            """Read a page plus its neighbors (page before and after) for broader context. Use doc_name to target a specific document."""
            obs = computer.registry.call(
                "read_surrounding_context",
                doc_id=_resolve_doc(page_number, doc_name),
                page=page_number,
                window=1,
            )
            return json.dumps(obs.data, default=str)

        @lc_tool
        def read_section(heading: str, doc_name: str = "") -> str:
            """Read all text under a specific section heading (found via ToC). Works across page breaks. Use doc_name to target a specific document."""
            did = primary
            if doc_name:
                did = _resolve_doc(1, doc_name)
            obs = computer.registry.call(
                "read_section",
                doc_id=did,
                heading=heading,
            )
            return json.dumps(obs.data, default=str)

        @lc_tool
        def get_table_of_contents() -> str:
            """Get the full Table of Contents (all section headings with page numbers) for every loaded document."""
            return self._build_toc(computer)

        @lc_tool
        def get_document_overview(doc_index: int = 0) -> str:
            """Get document profile: classification, headings, tables, sheets, errors. Use to understand document structure."""
            did = doc_ids[doc_index] if doc_index < len(doc_ids) else primary
            obs = computer.registry.call("inspect_document", doc_id=did)
            return json.dumps(obs.data, default=str)

        @lc_tool
        def find_tables(page_number: int = 0, doc_name: str = "") -> str:
            """Detect tables on a specific page (or all pages if page_number=0). Returns table dimensions, headers, and locations. Automatically uses Azure Document Intelligence when native detection finds no tables on a specific page. Use doc_name to target a specific document."""
            did = _resolve_doc(page_number, doc_name) if page_number else primary
            if doc_name and not page_number:
                did = _resolve_doc(1, doc_name)
            obs = computer.registry.call(
                "detect_tables",
                doc_id=did,
                page=page_number or None,
            )
            data = obs.data
            data["filename"] = _doc_filename(did)

            native_count = data.get("count", 0)
            if native_count == 0 and page_number > 0:
                di_result = self._try_di_find_tables(computer, did, page_number)
                if di_result is not None:
                    data = di_result
                    data["filename"] = _doc_filename(did)

            return json.dumps(data, default=str)

        @lc_tool
        def extract_table(
            page_number: int, table_index: int = 0, doc_name: str = ""
        ) -> str:
            """Extract full table data (all rows and columns) from a table on a page. Try multiple table_index values (0, 1, 2...) if the first doesn't contain the data you need. Use doc_name to target a specific document. Falls back to Azure Document Intelligence when native extraction fails."""
            did = _resolve_doc(page_number, doc_name)

            parsed = computer.world.parsed.get(did)
            page_obj = next(
                (
                    p
                    for p in (parsed.pages if parsed else [])
                    if p.page_no == page_number
                ),
                None,
            )
            page_parse_failed = page_obj is not None and page_obj.table_detection_failed

            native_cols = 0
            if page_obj and page_obj.tables:
                idx = min(table_index, len(page_obj.tables) - 1)
                native_cols = page_obj.tables[idx].n_cols

            prefer_di = native_cols > 10 or page_parse_failed

            if prefer_di:
                reason = (
                    f"dense table ({native_cols} cols)"
                    if native_cols > 10
                    else "parse failure"
                )
                self.bus.post(
                    "tool_call",
                    f"[DI] Using Document Intelligence for page {page_number} ({reason})",
                    "azure_di",
                )
                di_result = self._try_di_table(computer, did, page_number, table_index)
                if di_result is not None:
                    return di_result
                self.bus.post(
                    "tool_call",
                    f"[DI] Failed for page {page_number}, falling back to native extraction",
                    "azure_di",
                )

            obs = computer.registry.call(
                "extract_table",
                doc_id=did,
                table_index=table_index,
                page=page_number,
            )
            data = obs.data
            data["filename"] = _doc_filename(did)
            result_str = json.dumps(data, default=str)

            native_failed = (
                obs.status != "ok"
                or not data.get("rows")
                or HarnessService._is_corrupt_table(result_str)
            )

            if native_failed:
                self.bus.post(
                    "tool_call",
                    f"[DI] Native extraction failed for page {page_number}, trying Document Intelligence",
                    "azure_di",
                )
                di_result = self._try_di_table(computer, did, page_number, table_index)
                if di_result is not None:
                    return di_result

            return result_str

        @lc_tool
        def reconstruct_table(page_number: int, doc_name: str = "") -> str:
            """Reconstruct the logical table structure on a page — resolves header hierarchies, banner groups, and multi-page continuations. Use when extract_table returns incomplete or corrupted data. Use doc_name to target a specific document."""
            did = _resolve_doc(page_number, doc_name)

            parsed = computer.world.parsed.get(did)
            page_obj = next(
                (
                    p
                    for p in (parsed.pages if parsed else [])
                    if p.page_no == page_number
                ),
                None,
            )
            native_cols = 0
            if page_obj and page_obj.tables:
                native_cols = max(t.n_cols for t in page_obj.tables)

            if native_cols > 10:
                self.bus.post(
                    "tool_call",
                    f"[DI] Using Document Intelligence for page {page_number} (dense table, {native_cols} cols)",
                    "azure_di",
                )
                di_result = self._try_di_table(computer, did, page_number, 0)
                if di_result is not None:
                    return di_result
                self.bus.post(
                    "tool_call",
                    f"[DI] Failed for page {page_number}, falling back to native reconstruction",
                    "azure_di",
                )

            obs = computer.registry.call(
                "reconstruct_table",
                doc_id=did,
                page=page_number,
            )
            return json.dumps(obs.data, default=str)

        @lc_tool
        def analyze_with_document_intelligence(
            page_number: int, doc_name: str = ""
        ) -> str:
            """Azure Document Intelligence OCR + layout analysis. Heavy-duty fallback for pages where native text extraction is corrupted or tables are unreadable. Returns structured tables, text blocks, and layout. Use doc_name to target a specific document."""
            try:
                obs = computer.registry.call(
                    "di_analyze_document",
                    doc_id=_resolve_doc(page_number, doc_name),
                    pages=[page_number],
                )
                return json.dumps(obs.data, default=str)
            except Exception as exc:
                return f"Document Intelligence unavailable: {exc}"

        @lc_tool
        def search_headings(query: str) -> str:
            """Search document section headings to find where topics are covered."""
            obs = computer.registry.call(
                "search_headings",
                query=query,
                top_k=10,
            )
            return json.dumps(obs.data, default=str)

        @lc_tool
        def list_loaded_documents() -> str:
            """List all loaded documents with their page counts and filenames."""
            obs = computer.registry.call("list_documents")
            return json.dumps(obs.data, default=str)

        @lc_tool
        def inspect_page_visually(
            page_number: int, question: str, doc_name: str = ""
        ) -> str:
            """Use Vision (GPT) to visually inspect a rendered page image. Best for ambiguous tables, charts, or scanned content where text extraction is unreliable. MOST RELIABLE tool for corrupted tables. Use doc_name to target a specific document."""
            try:
                obs = computer.registry.call(
                    "inspect_page_visually",
                    doc_id=_resolve_doc(page_number, doc_name),
                    page=page_number,
                    question=question,
                )
                return json.dumps(obs.data, default=str)
            except Exception as exc:
                return f"Vision unavailable: {exc}"

        return [
            search_document,
            read_page,
            read_nearby_pages,
            read_section,
            get_table_of_contents,
            get_document_overview,
            find_tables,
            extract_table,
            reconstruct_table,
            analyze_with_document_intelligence,
            search_headings,
            list_loaded_documents,
            inspect_page_visually,
        ]

    def _investigative_qa(
        self,
        prompt: str,
        _pre_steps: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Deep investigative Q&A: map-first, targeted-read agent. Scales to any doc size."""
        from ..sanitize import sanitize_text, scrub_secrets
        from langchain_core.messages import (
            HumanMessage,
            SystemMessage,
            ToolMessage,
        )

        started = time.time()
        steps: List[str] = list(_pre_steps or [])
        citations: List[Dict[str, Any]] = []

        computer = self._ensure_computer()
        tools = self._build_qa_tools(computer)
        tool_map = {t.name: t for t in tools}

        from ...deep.model import build_tracked_chat_model

        model = build_tracked_chat_model(max_tokens=32768)
        if model is None:
            return {
                "text": "No LLM model configured. Set API keys in environment.",
                "citations": [],
                "answers": 0,
                "verified": 0,
                "status": "no_model",
                "elapsed_seconds": 0.0,
                "prompt": prompt,
                "contracts": [],
                "tool_steps": steps,
            }

        model_with_tools = model.bind_tools(tools)

        world = computer.world
        doc_names = [
            (world.profiles[d].filename if d in world.profiles else d)
            for d in world.parsed
        ]
        total_pages = sum(len(p.pages) for p in world.parsed.values())

        self.bus.post(RUN_STARTED, "Investigating documents", prompt[:120])

        toc_text = self._build_toc(computer)
        toc_step = f"Built Table of Contents ({toc_text.count(chr(10))} headings)"
        steps.append(toc_step)
        self.bus.post("toc_built", toc_step, "")

        doc_inventory = []
        for did, name in zip(world.parsed, doc_names):
            pages = len(world.parsed[did].pages)
            tables = sum(len(p.tables) for p in world.parsed[did].pages)
            doc_inventory.append(f"  - {name} ({pages} pages, {tables} tables)")
        inventory_text = "\n".join(doc_inventory)

        sys_prompt = (
            "You are KYNEX, a document intelligence assistant.\n"
            f"Loaded: {len(doc_names)} document(s), {total_pages} total pages:\n"
            f"{inventory_text}\n\n"
            f"## Table of Contents\n{toc_text}\n\n"
            "## STRATEGY: Search ALL → Identify → Target → Extract → Validate\n"
            f"{_strategy_intro()}Follow this loop:\n"
            "1. SEARCH ALL: Use search_document(keywords) to search ACROSS "
            "ALL loaded documents. The results show which DOCUMENT and PAGE "
            "contain each hit (look at the 'filename' field). Do NOT assume "
            "data is in any particular document — a file named 'IO LIST' "
            "might have IO counts, or they could be in the specification "
            "document. Search broadly, then narrow down.\n"
            "2. IDENTIFY: From search results, note WHICH document(s) "
            "have the relevant content. The answer may span multiple docs.\n"
            "3. TARGET: read_page(page, doc_name='<filename>') ONLY the "
            "candidate pages in the correct document. ALWAYS pass doc_name "
            "when multiple documents are loaded — without it, you may read "
            "the wrong document's page.\n"
            "4. EXTRACT: For pages with tables, ALWAYS use extract_table(page, "
            "doc_name=...) — NEVER rely on read_page for numeric table data. "
            "extract_table uses Document Intelligence for complex tables and "
            "returns structured rows/columns. Try every table_index (0, 1, 2...) "
            "until you find the data. Use read_page ONLY for narrative text, "
            "NOT for tables with numbers. read_page garbles table columns and "
            "drops values — extract_table is the ONLY reliable way to get "
            "complete table data.\n"
            "5. VALIDATE: Sum individual values and verify they match "
            "stated totals. If sums don't match → text extraction is "
            "corrupt → escalate.\n"
            "6. ESCALATE when extraction fails — MANDATORY FALLBACK CHAIN:\n"
            "   a) extract_table(page, doc_name=..., index) — try ALL table indices (0,1,2,3...)\n"
            "   b) reconstruct_table(page, doc_name=...) — rebuilds table from raw text heuristics\n"
            "   c) analyze_with_document_intelligence(page, doc_name=...) — Azure DI structured extraction\n"
            "   d) inspect_page_visually(page, doc_name=..., question='Read the table and transcribe ALL "
            "rows with ALL column values exactly as shown') — VISION is your LAST RESORT "
            "and MOST RELIABLE tool for corrupted tables.\n"
            "   NEVER give up after one failed tool — ALWAYS try the ENTIRE chain.\n"
            "   If extract_table returns empty/garbled → IMMEDIATELY try reconstruct_table.\n"
            "   If reconstruct_table fails → try analyze_with_document_intelligence.\n"
            "   If DI fails → use inspect_page_visually with a SPECIFIC question.\n"
            "   You have NOT exhausted options until you have tried ALL four methods.\n"
            "7. READ NOTES: Check text around tables for semantics "
            "(e.g. 'includes spares', 'not included in summary', "
            "'per unit'). Notes change the meaning of numbers.\n\n"
            "## CRITICAL RULES\n"
            "- Text extraction is UNRELIABLE for tables. Headers get "
            "scrambled, values drop, entire tables can be missing from "
            "text. Always use extract_table or vision to get numbers.\n"
            "- When a table has dash-separated values (columns show '—' or "
            "are missing), the TEXT layer is CORRUPTED. Do NOT report '—' as "
            "the answer. Escalate to vision immediately.\n"
            "- Do NOT pre-load or read ALL pages. Read only what you need.\n"
            "- MULTI-PAGE TABLES: Engineering tables (IO Summary, Cable Schedule, "
            "Instrument Index, etc.) often SPAN many consecutive pages. The TOC "
            "shows '[Tables on pages: X-Y]' — check ALL pages in the range. If "
            "extract_table on page N returns a table, ALWAYS check page N+1 for "
            "continuation. Keep extracting until you hit a page with NO table or "
            "a different table. Missing pages = incomplete data.\n"
            "- MULTI-DOCUMENT: Each doc has its OWN page numbering. "
            "Page 2 of 'IO LIST.pdf' is DIFFERENT from page 2 of 'GTS.pdf'. "
            "ALWAYS pass doc_name='<filename>' to target the right document. "
            "Use search_document results to identify which doc has the data, "
            "then pass that doc_name to read_page/extract_table/etc.\n"
            "- Cite every number with [doc_name, page X].\n"
            "- Show checksum validation (e.g. 28+82=110 ✓).\n"
            "- If a table is missing from text extraction, say so and "
            "use vision fallback — do NOT guess values.\n"
            "- Separate facts (extracted from doc) from inferences.\n"
            "- Reproduce tables in FULL markdown — ALL rows and columns.\n"
            "- Be thorough: show breakdowns, not just totals.\n"
            "- List what you could NOT verify (honest gaps).\n"
            "- NEVER say 'corrupted' or 'not recoverable' without trying "
            "ALL tools in the escalation chain. Vision can read what text "
            "extraction cannot.\n"
            "- If extract_table returns a table with SOME columns populated "
            "but others showing dashes, empty, or '—', the text layer is "
            "PARTIALLY CORRUPTED. You MUST call inspect_page_visually to "
            "get the missing column values. Do NOT report dashes as final "
            "answers — Vision will read the actual values from the PDF image.\n"
            "- inspect_page_visually is your MOST RELIABLE tool. When in doubt, "
            "use it. Ask specific questions: 'Read ALL rows and ALL columns "
            "of the table on this page, including numeric values.'\n\n"
            "## TEXT FALLBACK (scanned/image pages)\n"
            "Some pages are scanned images — read_page returns 0 chars or "
            "garbled text. When read_page returns empty/minimal text:\n"
            "  a) inspect_page_visually(page, 'Read ALL text on this page') — "
            "Vision reads the rendered image\n"
            "  b) analyze_with_document_intelligence(page) — OCR extraction\n"
            "NEVER report 'no text found' without trying Vision first.\n"
            "If image_coverage > 0.6 and char_count is low, the page is "
            "likely a scan or diagram — use Vision.\n\n"
            "## IMAGE/DIAGRAM HANDLING\n"
            "For pages with diagrams, flowcharts, P&ID drawings, or "
            "architectural drawings:\n"
            "  - inspect_page_visually with a SPECIFIC question about what "
            "to read from the image\n"
            "  - If the user asks about something that could be in an image "
            "(block diagram, network topology, single line diagram), check "
            "image_coverage and use Vision proactively\n"
            "Vision can read labels, annotations, and structured content "
            "from any rendered page image."
        )

        recent = self._recent_context()
        context_block = ""
        if recent.strip():
            context_block = (
                f"## Recent conversation (for follow-up context)\n{recent}\n\n"
            )

        messages = [
            SystemMessage(content=sys_prompt),
            HumanMessage(
                content=(
                    f"{context_block}"
                    f"USER QUESTION: {scrub_secrets(prompt)}\n\n"
                    "Start by using the Table of Contents and "
                    "search_document() to identify candidate pages, then "
                    "read and extract from those targeted pages."
                )
            ),
        ]

        answer_text = ""
        max_rounds = 30
        round_no = 0
        for round_no in range(max_rounds):
            try:
                response = model_with_tools.invoke(messages)
            except Exception as exc:
                answer_text = f"Error: {type(exc).__name__}: {str(exc)[:200]}"
                break

            self._qa_model_calls += 1
            usage = getattr(response, "usage_metadata", None)
            if usage:
                if isinstance(usage, dict):
                    self._qa_tokens += usage.get("input_tokens", 0) + usage.get(
                        "output_tokens", 0
                    )
                else:
                    self._qa_tokens += getattr(usage, "input_tokens", 0) + getattr(
                        usage, "output_tokens", 0
                    )

            if not response.tool_calls:
                answer_text = sanitize_text(str(response.content or ""))
                break

            messages.append(response)
            for tc in response.tool_calls:
                tool_fn = tool_map.get(tc["name"])
                if not tool_fn:
                    messages.append(
                        ToolMessage(
                            content="Unknown tool",
                            tool_call_id=tc["id"],
                        )
                    )
                    continue

                args_str = ", ".join(f"{k}={v!r}" for k, v in tc["args"].items())
                step = f"{tc['name']}({args_str})"
                steps.append(step)
                self._qa_tool_calls += 1
                self.bus.post("tool_call", step, f"round {round_no + 1}")

                try:
                    result = tool_fn.invoke(tc["args"])
                except Exception as exc:
                    result = f"Tool error: {type(exc).__name__}: {str(exc)[:200]}"

                result_str = str(result)
                result_str = self._add_fallback_hint(tc["name"], result_str, tc["args"])

                messages.append(
                    ToolMessage(
                        content=result_str,
                        tool_call_id=tc["id"],
                    )
                )

                if tc["name"] in ("read_page", "read_section", "read_nearby_pages"):
                    page_no = tc["args"].get("page_number", 0)
                    if page_no:
                        citations.append(
                            {
                                "id": len(citations) + 1,
                                "document": doc_names[0] if doc_names else "?",
                                "page": page_no,
                                "method": "investigative_read",
                            }
                        )
        else:
            if not answer_text:
                answer_text = sanitize_text(str(response.content or ""))

        elapsed = time.time() - started
        steps.append(f"Answered in {elapsed:.1f}s ({round_no + 1} rounds)")
        self.bus.post(
            RUN_COMPLETED,
            f"Answer ready ({elapsed:.1f}s, {round_no + 1} rounds)",
            f"{len(steps) - 2} tool calls",
        )

        self.session["turns"].append(
            {
                "prompt": prompt[:500],
                "answer": answer_text[:20000],
                "answers": 1,
                "ts": time.time(),
            }
        )
        return {
            "text": answer_text,
            "citations": citations[:20],
            "answers": 1,
            "verified": 1,
            "status": "answered",
            "elapsed_seconds": round(elapsed, 1),
            "prompt": prompt,
            "contracts": [],
            "tool_steps": steps,
        }

    @staticmethod
    def _di_nearby_pages(computer, did: str, page_number: int) -> list[int]:
        """Find consecutive dense-table pages around page_number.

        Walks outward from page_number collecting pages that also have >10
        columns, stopping at gaps. This batches e.g. pages 54-63 together
        when the agent asks for page 54, but won't include unrelated dense
        pages on page 47 or 100.
        """
        parsed = computer.world.parsed.get(did)
        if not parsed:
            return [page_number]
        dense = set()
        for p in parsed.pages:
            if p.tables and max(t.n_cols for t in p.tables) > 10:
                dense.add(p.page_no)
        if page_number not in dense:
            return [page_number]
        pages = [page_number]
        n = page_number + 1
        while n in dense:
            pages.append(n)
            n += 1
        n = page_number - 1
        while n in dense:
            pages.insert(0, n)
            n -= 1
        return pages

    def _try_di_table(self, computer, did: str, page_number: int, table_index: int):
        """Try Azure DI extraction for a page. Returns JSON string or None.

        Fetches only the requested page (+ nearby pages in the same table
        range from TOC) so we don't waste DI compute on pages the user
        never asked about. Results are cached per-page for reuse.
        """
        import os
        from ...tools.azure_di import AzureDIClient

        endpoint = os.environ.get("ADH_DI_ENDPOINT", "") or getattr(
            getattr(self.config, "model", None), "di_endpoint", ""
        )
        key = os.environ.get("ADH_DI_API_KEY", "") or getattr(
            getattr(self.config, "model", None), "di_key", ""
        )
        if not endpoint or not key:
            return None

        ref = computer.world.refs.get(did)
        if not ref:
            return None

        di_pages = getattr(computer, "_di_page_results", {})
        doc_pages = di_pages.setdefault(did, {})

        if page_number not in doc_pages:
            fetch_pages = self._di_nearby_pages(computer, did, page_number)
            fetch_pages = [p for p in fetch_pages if p not in doc_pages]
            if not fetch_pages:
                fetch_pages = [page_number]

            label = (
                f"page {page_number}"
                if len(fetch_pages) == 1
                else f"{len(fetch_pages)} pages ({fetch_pages[0]}-{fetch_pages[-1]})"
            )
            self.bus.post(
                "tool_call",
                f"[DI] Analyzing {label} via Document Intelligence",
                "azure_di",
            )

            try:
                cache = getattr(computer, "_di_cache", {})
                client = AzureDIClient(endpoint, key, cache=cache)
                mapped = client.analyze_layout(ref.path, fetch_pages)
            except Exception:
                for p in fetch_pages:
                    doc_pages[p] = None
                computer._di_page_results = di_pages
                return None

            for p in fetch_pages:
                page_tables = [t for t in mapped.tables if t.page == p]
                doc_pages[p] = page_tables if page_tables else None
            computer._di_page_results = di_pages

        cached_tables = doc_pages.get(page_number)
        if not cached_tables:
            return None
        page_tables = cached_tables
        if not page_tables:
            return None
        target = page_tables[min(table_index, len(page_tables) - 1)]
        grid = target.grid()
        if not grid:
            return None

        profile = computer.world.profiles.get(did)
        fname = profile.filename if profile else did
        self.bus.post(
            "tool_call",
            f"[DI] page {page_number} -> {target.rows}r x {target.cols}c",
            "azure_di",
        )
        data = {
            "doc_id": did,
            "page": page_number,
            "index": table_index,
            "rows": grid,
            "n_rows": len(grid),
            "n_cols": len(grid[0]) if grid else 0,
            "header": target.header(),
            "source": "azure_di",
            "filename": fname,
        }
        return json.dumps(data, default=str)

    def _try_di_find_tables(self, computer, did: str, page_number: int):
        """Try Azure DI to detect tables on a page. Returns dict or None."""
        import os
        from ...tools.azure_di import AzureDIClient

        endpoint = os.environ.get("ADH_DI_ENDPOINT", "") or getattr(
            getattr(self.config, "model", None), "di_endpoint", ""
        )
        key = os.environ.get("ADH_DI_API_KEY", "") or getattr(
            getattr(self.config, "model", None), "di_key", ""
        )
        if not endpoint or not key:
            return None

        ref = computer.world.refs.get(did)
        if not ref:
            return None

        try:
            cache = getattr(computer, "_di_cache", {})
            client = AzureDIClient(endpoint, key, cache=cache)
            mapped = client.analyze_layout(ref.path, [page_number])
        except Exception:
            return None

        page_tables = [t for t in mapped.tables if t.page == page_number]
        if not page_tables:
            return None

        di_found = []
        for idx, t in enumerate(page_tables):
            _grid = t.grid()
            di_found.append(
                {
                    "page": page_number,
                    "index": idx,
                    "rows": t.rows,
                    "cols": t.cols,
                    "header": t.header()[:8],
                    "source": "azure_di",
                }
            )
        return {
            "doc_id": did,
            "count": len(di_found),
            "tables": di_found,
            "source": "azure_di",
        }

    @staticmethod
    def _is_corrupt_table(result: str) -> bool:
        """Detect table results where extraction clearly failed.

        Corruption = dashes/null/N/A replacing real values. Blank cells and
        "0" are normal in sparse engineering tables (IO summaries, etc.) and
        are NOT treated as corruption.
        """
        import re as _re

        cells = _re.findall(r'"([^"]*)"', result)
        if len(cells) <= 10:
            return False
        corrupt_markers = sum(
            1 for c in cells if c.strip() in ("-", "—", "N/A", "null", "--")
        )
        if corrupt_markers > len(cells) * 0.3:
            return True
        return False

    @staticmethod
    def _is_poor_text(result: str) -> bool:
        """Detect text extraction results that are empty, garbled, or image-heavy."""
        try:
            data = json.loads(result)
        except (json.JSONDecodeError, TypeError):
            data = None

        if isinstance(data, dict):
            char_count = data.get("char_count", -1)
            img_cov = data.get("image_coverage", 0.0)
            text = data.get("text", "")
            if char_count == 0 or (isinstance(text, str) and len(text.strip()) < 30):
                return True
            if img_cov and float(img_cov) > 0.6 and char_count < 200:
                return True
        return False

    @staticmethod
    def _add_fallback_hint(tool_name: str, result: str, args: dict) -> str:
        """Append escalation hints when tools return poor/empty/corrupt results."""
        if not result or len(result.strip()) < 5:
            result = result or "(empty result)"

        _EMPTY_MARKERS = (
            "no table",
            "empty",
            "error",
            "none found",
            "not found",
            "failed",
            "unavailable",
        )
        lower = result.lower()[:500]
        is_poor = (
            len(result.strip()) < 20
            or any(m in lower for m in _EMPTY_MARKERS)
            or lower.startswith("tool error")
        )

        is_table_tool = tool_name in (
            "extract_table",
            "reconstruct_table",
            "analyze_with_document_intelligence",
        )
        is_corrupt = (
            is_table_tool and not is_poor and HarnessService._is_corrupt_table(result)
        )

        is_text_tool = tool_name in ("read_page", "read_section", "read_nearby_pages")
        is_text_poor = (
            is_text_tool and not is_poor and HarnessService._is_poor_text(result)
        )

        if not is_poor and not is_corrupt and not is_text_poor:
            return result

        page = args.get("page_number", "?")
        quality = "corrupt/partial" if is_corrupt else "poor"

        if tool_name == "extract_table":
            if is_poor:
                return (
                    f"{result}\n\n[KYNEX ESCALATION] extract_table returned no data "
                    f"for page {page}. The table may be borderless. "
                    f"Try: 1) read_page(page_number={page}) — the data may be in the "
                    f"text as aligned columns, parse it directly. "
                    f"2) inspect_page_visually(page_number={page}, "
                    f"question='Read the COMPLETE table on this page. For EVERY row, "
                    f"transcribe ALL column values. Present as a markdown table.'). "
                    f"3) reconstruct_table(page_number={page})."
                )
            return (
                f"{result}\n\n[KYNEX ESCALATION] extract_table returned {quality} results "
                f"for page {page}. Many column values are missing, dashes, or empty. "
                f"The text layer is UNRELIABLE for this table. "
                f"IMMEDIATELY call inspect_page_visually(page_number={page}, "
                f"question='Read the COMPLETE table on this page. For EVERY row, "
                f"transcribe ALL column values including Analog Input, Analog Output, "
                f"Digital Input, Digital Output, Digital Output Failsafe, and Total. "
                f"Present as a markdown table.'). "
                f"Also try reconstruct_table(page_number={page}) as a parallel check."
            )
        if tool_name == "reconstruct_table":
            return (
                f"{result}\n\n[KYNEX ESCALATION] reconstruct_table returned {quality} results "
                f"for page {page}. IMMEDIATELY call inspect_page_visually(page_number={page}, "
                f"question='Read the COMPLETE table on this page. For EVERY row, "
                f"transcribe ALL column values. Present as a markdown table.'). "
                f"Vision is the most reliable method for corrupted tables."
            )
        if tool_name == "analyze_with_document_intelligence":
            return (
                f"{result}\n\n[KYNEX ESCALATION] Document Intelligence returned {quality} results "
                f"for page {page}. IMMEDIATELY call inspect_page_visually(page_number={page}, "
                f"question='Read the COMPLETE table on this page. For EVERY row, "
                f"transcribe ALL column values. Present as a markdown table.'). "
                f"Vision is the most reliable fallback — do NOT give up."
            )
        if tool_name == "find_tables" and (
            "no table" in lower or 'count": 0' in lower or len(result.strip()) < 10
        ):
            return (
                f"{result}\n\n[KYNEX ESCALATION] find_tables found no structured tables on page {page}. "
                f"The table may be borderless (no drawn lines) or image-based. "
                f"Try TWO approaches: "
                f"1) call read_page(page_number={page}) and parse the aligned text directly — "
                f"borderless tables still appear as aligned columns in text extraction. "
                f"2) call inspect_page_visually(page_number={page}, "
                f"question='Is there a table on this page? If yes, transcribe it completely "
                f"with all rows and columns as a markdown table')."
            )
        if is_text_tool and (is_poor or is_text_poor):
            reason = (
                "no selectable text (likely a scanned/image page)"
                if is_poor
                else "very little text with high image coverage"
            )
            return (
                f"{result}\n\n[KYNEX ESCALATION] {tool_name} returned {reason} for "
                f"page {page}. This page may be scanned, image-based, or have a corrupted "
                f"text layer. IMMEDIATELY call inspect_page_visually(page_number={page}, "
                f"question='Read ALL text content on this page. Transcribe headings, "
                f"paragraphs, labels, table data, and any visible text exactly as shown.') "
                f"to extract the content visually. Also try "
                f"analyze_with_document_intelligence(page_number={page}) for OCR-based extraction."
            )
        return result

    def ask(
        self, prompt: str, budgets: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Run a conversational investigation; return answer + evidence."""
        # Small talk / identity: conversational reply, NEVER an investigation.
        if self._is_small_talk(prompt):
            text = self._llm_reply(
                prompt,
                fallback=(
                    "Hey! I'm KYNEX — I investigate documents and answer with "
                    "evidence. Use /add <file-or-folder> to load documents, then "
                    "just ask questions."
                ),
            )
            self.session["turns"].append(
                {
                    "prompt": prompt[:500],
                    "answer": text[:20000],
                    "answers": 0,
                    "ts": time.time(),
                }
            )
            return {
                "text": text,
                "citations": [],
                "answers": 0,
                "verified": 0,
                "status": "conversational",
                "elapsed_seconds": 0.0,
                "contracts": [],
            }
        # LLM router: slang/jokes/venting classify conversational regardless
        # of phrasing (word lists can never cover natural language).
        kind, reply = self._route_prompt(prompt)
        if kind == "conversational":
            self.session["turns"].append(
                {
                    "prompt": prompt[:500],
                    "answer": reply[:20000],
                    "answers": 0,
                    "ts": time.time(),
                }
            )
            return {
                "text": reply,
                "citations": [],
                "answers": 0,
                "verified": 0,
                "status": "conversational",
                "elapsed_seconds": 0.0,
                "contracts": [],
            }
        if not self.session.get("documents"):
            return {
                "text": "No documents loaded. Use /add <file-or-folder> first.",
                "citations": [],
                "answers": 0,
                "verified": 0,
                "status": "no_documents",
                "elapsed_seconds": 0.0,
                "contracts": [],
            }
        raw_prompt = re.sub(r"^\[Context:.*?\]\s*", "", prompt, flags=re.DOTALL)
        return self._investigative_qa(raw_prompt)

    def _recent_context(self, turns: int = 3) -> str:
        lines = []
        for turn in self.session.get("turns", [])[-turns:]:
            lines.append(f"Q: {str(turn.get('prompt', ''))[:300]}")
            lines.append(f"A: {str(turn.get('answer', ''))[:500]}")
        return "\n".join(lines)

    def _new_store(
        self, contracts, budgets: Optional[Dict[str, Any]] = None
    ) -> RunStore:
        store = RunStore(contracts)
        budgets = budgets or {}
        store.budgets.max_model_calls = budgets.get("max_model_calls", 120)
        store.budgets.max_tool_calls = budgets.get("max_tool_calls", 600)
        store.budgets.max_visual_calls = budgets.get("max_visual_calls", 60)
        store.budgets.timeout_seconds = budgets.get("timeout_seconds", 1800.0)
        return store

    def _wire_tool_events(self, computer) -> None:
        return None

    def _investigator_kwargs(
        self, run_id: str, thread_id: str, max_resume_rounds: int, deep_dive_rounds: int
    ) -> Dict[str, Any]:
        return {
            "run_id": run_id,
            "max_resume_rounds": max_resume_rounds,
            "thread_id": thread_id,
            "deep_dive_rounds": deep_dive_rounds,
            "event_sink": self._event_sink,
        }

    # -- session state ------------------------------------------------------
    def save_state(self) -> None:
        """Persist ledger + turns into the session dict (no secrets)."""
        if self._store is not None:
            try:
                self.session["ledger"] = self._store.to_ledger()
            except Exception:
                pass
        self.session["metrics"] = self.get_cost()

    def restore_state(self) -> None:
        """Rebuild the store from a saved ledger (world rebuilt lazily)."""
        ledger = self.session.get("ledger")
        if not ledger:
            return
        try:
            self._store = RunStore.from_ledger(ledger)
            world = self.ensure_world()
            self._computer = DocumentComputer(
                world, self._store.budgets, config=self.config
            )
            self._computer.budgets = self._store.budgets
        except Exception:
            self._store = None
            self._computer = None

    def _synthesize(self, prompt: str, store: RunStore) -> Dict[str, Any]:
        from ..sanitize import sanitize_text

        all_answers = [a for a in store.answers if a.admission == "ANSWER"]
        verified = [a for a in all_answers if a.verification == "verified"]
        chosen = verified or all_answers
        if not chosen:
            return {
                "text": "I could not find a supported answer in the documents.",
                "citations": [],
                "answers": 0,
                "verified": 0,
            }
        # Prefer substantive values: pure "Not specified" negatives never
        # crowd out real records in the composed reply.
        negative_re = re.compile(
            r"^\s*(not specified|not applicable|n/?a|none|absent)\b", re.IGNORECASE
        )
        substantive = [a for a in chosen if not negative_re.match(str(a.value or ""))]
        use = substantive or chosen

        def _doc_of(answer) -> str:
            try:
                return self._computer.world.profiles[answer.document_id].filename
            except Exception:
                return answer.document_id

        citations = []
        for index, answer in enumerate(use[:12], start=1):
            citations.append(
                {
                    "id": index,
                    "answer_id": answer.answer_id,
                    "document": _doc_of(answer),
                    "page": answer.page,
                    "method": answer.method,
                    "snippet": answer.snippet[:300],
                    "verification": answer.verification,
                }
            )

        # LLM-composed reply (natural language, tables when asked). The model
        # only FORMATS the verified records — it cannot invent values.
        model = self._chat_model()
        if model is not None:
            try:
                records = [
                    {
                        "subject": a.subject,
                        "attribute": a.attribute,
                        "value": str(a.value),
                        "source": f"{_doc_of(a)} p.{a.page}",
                        "verified": a.verification == "verified",
                    }
                    for a in use[:40]
                ]
                response = model.invoke(
                    [
                        {
                            "role": "system",
                            "content": (
                                "You are KYNEX, a document-intelligence "
                                "assistant. Compose the final answer to the "
                                "user's question using ONLY the extracted "
                                "records provided. Rules: never invent values; "
                                "if the user asked for a table, render a clean "
                                "markdown table from the records; otherwise "
                                "answer in 2-6 sentences or a short bullet "
                                "list; cite sources inline as [doc p.page]; "
                                "if records say 'Not specified', say so "
                                "honestly rather than filling gaps."
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Question: {prompt}\n\nExtracted records "
                                f"(JSON):\n{json.dumps(records, default=str)[:6000]}"
                            ),
                        },
                    ]
                )
                text = sanitize_text(str(response.content or "")).strip()
                if text:
                    return {
                        "text": text,
                        "citations": citations,
                        "answers": len(store.answers),
                        "verified": len(verified),
                    }
            except Exception:
                pass  # fall through to deterministic composition

        # Deterministic fallback (offline / model error).
        lines = [f"{a.subject} — {a.attribute}: {a.value}" for a in use[:8]]
        header = (
            "Based on document evidence:"
            if verified
            else "Candidate findings (unverified):"
        )
        text = header + "\n" + "\n".join(f"- {line}" for line in lines)
        if len(use) > 8:
            text += f"\n(+{len(use) - 8} more findings; see /evidence)"
        return {
            "text": text,
            "citations": citations,
            "answers": len(store.answers),
            "verified": len(verified),
        }

    # -- readouts ---------------------------------------------------------
    def get_status(self) -> Dict[str, Any]:
        store = self._store
        contracts = store.contract_summary() if store else []
        states: Dict[str, int] = {}
        for entry in contracts:
            states[entry["state"]] = states.get(entry["state"], 0) + 1
        budgets = store.budgets.to_dict() if store else {}
        return {
            "contracts": len(contracts),
            "states": states,
            "answers": len(store.answers) if store else 0,
            "verified": sum(1 for a in store.answers if a.verification == "verified")
            if store
            else 0,
            "budgets": budgets,
            "runs": self._runs,
        }

    def get_contracts(self, filter_state: str = "") -> List[Dict[str, Any]]:
        if not self._store:
            return []
        entries = self._store.contract_summary()
        if filter_state:
            entries = [
                e
                for e in entries
                if e["state"].lower().startswith(filter_state.lower())
            ]
        return entries

    def get_evidence(self, query: str = "") -> List[Dict[str, Any]]:
        if not self._store:
            return []
        items = []
        for answer in self._store.answers:
            blob = f"{answer.answer_id} {answer.subject} {answer.attribute} {answer.value} {answer.page}".lower()
            if query and query.lower() not in blob:
                continue
            items.append(answer.to_claim())
        return items

    def get_trace(self, limit: int = 50, kind: str = "") -> List[Dict[str, Any]]:
        if not self._store:
            return []
        observations = list(self._store.observations[-limit:])
        if kind:
            observations = [
                o for o in observations if kind.lower() in o["tool"].lower()
            ]
        return observations

    def get_coverage(self) -> Dict[str, Any]:
        if not self._store:
            return {"contracts": [], "coverage": {}}
        return {
            "contracts": self._store.contract_summary(),
            "coverage": self._store.coverage.to_dict(),
        }

    def get_cost(self) -> Dict[str, Any]:
        store_tokens = 0
        store_model = 0
        store_tool = 0
        store_visual = 0
        store_di = 0
        if self._store:
            budgets = self._store.budgets
            store_tokens = budgets.tokens_used
            store_model = budgets.model_calls
            store_tool = budgets.tool_calls
            store_visual = budgets.visual_calls
            store_di = budgets.di_calls
        return {
            "tokens": store_tokens + self._qa_tokens,
            "model_calls": store_model + self._qa_model_calls,
            "tool_calls": store_tool + self._qa_tool_calls,
            "visual_calls": store_visual,
            "di_calls": store_di,
        }

    # -- retry / export ----------------------------------------------------
    def retry_unresolved(self, contract_id: str = "") -> Dict[str, Any]:
        """Targeted recovery: re-run the investigator over open contracts."""
        if not self._store or not self._computer:
            return {"retried": 0, "reason": "no prior run in this session"}
        open_contracts = [
            c
            for c in self._store.unresolved_contracts()
            if not contract_id or c.contract_id == contract_id
        ]
        if not open_contracts:
            return {"retried": 0, "reason": "nothing unresolved"}
        investigator = DeepAgentInvestigator(
            computer=self._computer,
            store=self._store,
            contracts=list(self._store.contracts.values()),
            **self._investigator_kwargs(
                f"retry_{self._runs + 1}",
                f"{self.session['id']}_retry_{self._runs + 1}",
                1,
                1,
            ),
        )
        result = investigator.run()
        self._runs += 1
        return {
            "retried": len(open_contracts),
            "status": result.status,
            "answers": len(self._store.answers),
        }

    def export(self, fmt: str, path: str = "") -> str:

        if not self._store:
            raise ValueError("nothing to export yet — ask a question first")

        # Reuse the standard compiler via a lightweight claim projection.
        result = {
            "objective": "harness session export",
            "claims": [a.to_claim() for a in self._store.answers],
            "unresolved": [],
            "contradictions": self._store.contradictions,
            "verification": {"gates": []},
            "stats": {"claims": len(self._store.answers)},
        }
        downloads = Path.home() / "Downloads"
        downloads.mkdir(parents=True, exist_ok=True)

        out_path = path or str(
            downloads / f"kynex_export_{int(time.time())}.{fmt}"
        )
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        if fmt == "json":
            import json as _json

            Path(out_path).write_text(_json.dumps(result, indent=2, default=str))
        elif fmt == "markdown" or fmt == "md":
            lines = ["# Harness export", ""]
            for claim in result["claims"]:
                lines.append(
                    f"- {claim['subject']} / {claim['attribute']}: {claim['value']}"
                )
            Path(out_path).write_text("\n".join(lines))
        elif fmt == "evidence":
            import json as _json

            cards = [e for c in result["claims"] for e in c.get("evidence", [])]
            Path(out_path).write_text(_json.dumps(cards, indent=2, default=str))
        elif fmt == "trace":
            import json as _json

            Path(out_path).write_text(
                _json.dumps(self.get_trace(5000), indent=2, default=str)
            )
        else:
            raise ValueError(f"unsupported export format '{fmt}'")
        self.session["exports"].append(
            {"format": fmt, "path": out_path, "ts": time.time()}
        )
        return out_path
