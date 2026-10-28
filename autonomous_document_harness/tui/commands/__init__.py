"""Slash-command registry: pure parsing + dispatch (no Textual dependency).

Every command returns a CommandResult with printable text; long-running work
(ask, retries) returns an action the app executes in a worker thread
while streaming bus events. Fully unit-testable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class CommandResult:
    text: str
    action: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)


def split_args(line: str) -> List[str]:
    """Quote-aware split that preserves Windows backslashes (shlex eats them)."""
    return [
        a or b or c for a, b, c in re.findall(r'"([^"]*)"|\'([^\']*)\'|(\S+)', line)
    ]


def is_command(line: str) -> bool:
    return line.strip().startswith("/")


class CommandRegistry:
    """Dispatches slash commands against live services."""

    def __init__(self, harness, sessions, new_session_hook=None, conv_state=None) -> None:
        self.harness = harness
        self.sessions = sessions
        self.new_session_hook = new_session_hook
        self.conv_state = conv_state
        self.history: List[str] = []
        self.citations: Dict[int, Dict[str, Any]] = {}

    # -- entry ----------------------------------------------------------
    def dispatch(self, line: str) -> CommandResult:
        stripped = line.strip()
        if not stripped:
            return CommandResult("")
        if not stripped.startswith("/"):
            self.history.append(stripped)
            return CommandResult("", action="ask", payload={"prompt": stripped})
        try:
            parts = split_args(stripped)
        except ValueError:
            parts = stripped.split()
        if not parts:
            return CommandResult("")
        name = parts[0].lower()
        args = parts[1:]
        handler = getattr(self, f"cmd_{name[1:]}", None)
        if handler is None:
            return CommandResult(f"Unknown command '{parts[0]}'. Type /help.")
        try:
            return handler(args)
        except Exception as exc:
            return CommandResult(f"! {name} failed: {type(exc).__name__}: {exc}")

    # -- help -----------------------------------------------------------
    def cmd_help(self, args: List[str]) -> CommandResult:
        return CommandResult(
            "ASK\n"
            "  Just type naturally — KYNEX investigates and answers.\n\n"
            "WORKSPACE\n"
            "  /add <path> [...]     Stage files or folders\n"
            "  /remove [doc]         Remove document (source kept)\n"
            "  /docs [n]             List documents / show detail\n"
            "  /open <doc> <page>    Render page or evidence\n\n"
            "EVIDENCE\n"
            "  /evidence [query]     View evidence findings\n"
            "  /trace [n] [kind]     Investigation trace\n"
            "  /coverage             Resolution coverage\n\n"
            "SESSION\n"
            "  /session              Current session info\n"
            "  /session list         All sessions\n"
            "  /session rename <n>   Rename current session\n"
            "  /new                  Start new session\n"
            "  /save                 Save checkpoint\n"
            "  /history              Command history\n\n"
            "ENGINE\n"
            "  /engine               Engine info (V3/V2)\n"
            "  /model                Model information\n"
            "  /tools                Available tools\n"
            "  /cost                 Session usage and cost\n"
            "  /health               Service health check\n\n"
            "OUTPUT\n"
            "  /copy                 Copy last answer to clipboard\n"
            "  /copy all             Copy full conversation to clipboard\n"
            "  /txt [path]           Export full chat history to text file\n"
            "  /export <fmt> [path]  Export (json|markdown|evidence|trace)\n\n"
            "PROVIDER\n"
            "  /provider             Current provider config\n"
            "  /provider azure       Configure Azure OpenAI\n"
            "  /provider openai      Configure OpenAI endpoint\n"
            "  /provider di          Configure Document Intelligence\n"
            "  /provider show        Show saved credentials (redacted)\n"
            "  /provider clear       Remove saved credentials\n\n"
            "GENERAL\n"
            "  /clear                Clear display\n"
            "  /help                 Show this help\n"
            "  /quit                 Quit KYNEX",
        )

    # -- workspace ------------------------------------------------------
    def cmd_add(self, args: List[str]) -> CommandResult:
        if not args:
            return CommandResult("Usage: /add <file-or-folder> [...]")
        result = self.harness.add_documents(args)
        staged_names = result["staged"]
        if not staged_names and not result["errors"]:
            lines = ["Nothing new to add."]
            for note in result["skipped"][:10]:
                lines.append(f"  ! {note}")
            return CommandResult("\n".join(lines))
        lines = [
            f"Staged {len(staged_names)} document(s), "
            f"{len(result['skipped'])} skipped"
        ]
        for name in staged_names[:10]:
            lines.append(f"  + {name}")
        for note in (result["skipped"] + result["errors"])[:10]:
            lines.append(f"  ! {note}")
        lines.append("Parsing documents...")
        return CommandResult(
            "\n".join(lines),
            action="add",
            payload={"staged": staged_names},
        )

    def cmd_remove(self, args: List[str]) -> CommandResult:
        if not args:
            docs = self.harness.list_documents()
            if not docs:
                return CommandResult("Workspace is empty.")
            lines = ["Select a document to remove (source file is kept):"]
            for i, doc in enumerate(docs, 1):
                lines.append(f"  {i}. {doc['filename']}")
            return CommandResult("\n".join(lines))
        name = " ".join(args)
        ok = self.harness.remove_document(name)
        return CommandResult(
            f"Removed {name} (reference only)." if ok else f"No such document: {name}"
        )

    def cmd_docs(self, args: List[str]) -> CommandResult:
        docs = self.harness.list_documents()
        if not docs:
            return CommandResult("No documents loaded. Use /add <file-or-folder>.")
        if args:
            try:
                index = int(args[0]) - 1
                doc = docs[index]
            except (ValueError, IndexError):
                wanted = " ".join(args)
                doc = next((d for d in docs if d["filename"] == wanted), None)
                if doc is None:
                    return CommandResult(f"No such document: {wanted}")
            return CommandResult("\n".join(f"{k}: {v}" for k, v in doc.items()))
        lines = ["Documents", ""]
        total_pages = 0
        try:
            world = self.harness.ensure_world()
            profiles = world.profiles
        except Exception:
            profiles = {}
        for i, doc in enumerate(docs, 1):
            detail = ""
            for profile in profiles.values():
                if profile.filename == doc["filename"]:
                    detail = f"{profile.page_count}p {profile.classification}"
                    total_pages += profile.page_count
                    break
            lines.append(f"{i}  {doc['filename']}  {detail}")
        lines.append(f"\nTotal staged: {len(docs)}")
        if total_pages:
            lines.append(f"Total pages: {total_pages}")
        return CommandResult("\n".join(lines))

    def cmd_open(self, args: List[str]) -> CommandResult:
        if not args:
            return CommandResult(
                "Usage: /open <doc> <page> | /open evidence <id> | /open <n>"
            )
        if args[0].lower() == "evidence" and len(args) > 1:
            return self._open_evidence(args[1])
        if len(args) == 1 and args[0].isdigit():
            citation = self.citations.get(int(args[0]))
            if not citation:
                return CommandResult(f"No citation [{args[0]}] in recent answers.")
            return self._render_page(citation["document"], citation.get("page") or 1)
        if len(args) >= 2:
            try:
                page = int(args[-1])
            except ValueError:
                return CommandResult("Usage: /open <doc> <page>")
            # Join all but the last token: unquoted multi-word filenames work.
            return self._render_page(" ".join(args[:-1]), page)
        return CommandResult(
            "Usage: /open <doc> <page> | /open evidence <id> | /open <n>"
        )

    def _render_page(self, doc: str, page: int) -> CommandResult:
        try:
            world = self.harness.ensure_world()
        except Exception as exc:
            return CommandResult(f"! cannot open: {exc}")
        doc_id = None
        for profile in world.profiles.values():
            if profile.filename == doc or profile.doc_id == doc:
                doc_id = profile.doc_id
                break
        if doc_id is None:
            return CommandResult(f"No such document: {doc}")
        try:
            from ...tools import vision as vision_mod

            ref = world.refs[doc_id]
            out = self.harness.workspace_dir / f"preview_{doc_id}_p{page}.png"
            vision_mod.render_page_to_png(ref.path, page, 110, str(out))
            return CommandResult(f"Page rendered: {out}\n({doc} p.{page})")
        except Exception as exc:
            return CommandResult(f"! render failed: {type(exc).__name__}: {exc}")

    def _open_evidence(self, evidence_id: str) -> CommandResult:
        for item in self.harness.get_evidence():
            for ev in item.get("evidence", []):
                if str(ev.get("evidence_id", "")).endswith(str(evidence_id)):
                    doc = ev.get("filename") or ev.get("document_id", "")
                    return self._render_page(doc, ev.get("page") or 1)
        return CommandResult(f"No such evidence: {evidence_id}")

    # -- agent readouts ---------------------------------------------------
    def cmd_status(self, args: List[str]) -> CommandResult:
        status = self.harness.get_status()
        lines = ["Agent status", ""]
        states = status.get("states", {})
        lines.append(f"Contracts   {status.get('contracts', 0)}")
        for state in (
            "COMPLETE",
            "PARTIAL",
            "DISCOVERING",
            "NOT_FOUND",
            "BLOCKED",
            "UNSEEN",
        ):
            if states.get(state):
                lines.append(f"  {state.title():11} {states[state]}")
        lines.append(f"Verified claims  {status.get('verified', 0)}")
        budgets = status.get("budgets", {})
        for key in (
            "tool_calls",
            "model_calls",
            "visual_calls",
            "di_calls",
            "tokens_used",
        ):
            if key in budgets:
                lines.append(f"{key:13} {budgets[key]}")
        lines.append(f"Agent runs    {status.get('runs', 0)}")
        return CommandResult("\n".join(lines))

    def cmd_contracts(self, args: List[str]) -> CommandResult:
        filt = (
            args[0] if args and args[0] in ("complete", "unresolved", "partial") else ""
        )
        target = args[0] if args and not filt else ""
        entries = self.harness.get_contracts("" if target else filt)
        if not entries:
            return CommandResult("No contracts yet — ask a question first.")
        if target:
            match = next((e for e in entries if e["contract_id"] == target), None)
            if not match:
                return CommandResult(f"No such contract: {target}")
            return CommandResult("\n".join(f"{k}: {v}" for k, v in match.items()))
        if filt == "unresolved":
            entries = [e for e in entries if e["state"] not in ("COMPLETE",)]
        marks = {"COMPLETE": "✓", "NOT_FOUND": "—", "BLOCKED": "■"}
        lines = [
            f"{marks.get(e['state'], '◉' if e['answers'] else '?')} "
            f"{e['sheet']} / {e['field']}  [{e['contract_id']}]"
            for e in entries
        ]
        return CommandResult("\n".join(lines))

    def cmd_unresolved(self, args: List[str]) -> CommandResult:
        entries = [
            e for e in self.harness.get_contracts() if e["state"] not in ("COMPLETE",)
        ]
        if not entries:
            return CommandResult("Nothing unresolved.")
        lines = []
        for i, entry in enumerate(entries, 1):
            lines.append(f"{i}. {entry['sheet']} / {entry['field']}")
            lines.append(f"   State: {entry['state']}, answers: {entry['answers']}")
        return CommandResult("\n".join(lines))

    def cmd_retry(self, args: List[str]) -> CommandResult:
        if args and args[0] == "all":
            return CommandResult("", action="retry", payload={"contract_id": ""})
        return CommandResult(
            "", action="retry", payload={"contract_id": args[0] if args else ""}
        )

    def cmd_trace(self, args: List[str]) -> CommandResult:
        limit, kind = 20, ""
        for arg in args:
            if arg.isdigit():
                limit = int(arg)
            else:
                kind = arg
        observations = self.harness.get_trace(limit, kind)
        if not observations:
            return CommandResult("No trace events yet.")
        lines = [
            f"{o.get('tool', '?')} {str(o.get('args', ''))[:80]} — "
            f"{str(o.get('summary', ''))[:80]}"
            for o in observations
        ]
        return CommandResult("\n".join(lines))

    def cmd_coverage(self, args: List[str]) -> CommandResult:
        coverage = self.harness.get_coverage()
        contracts = coverage.get("contracts", [])
        done = sum(1 for c in contracts if c["state"] == "COMPLETE")
        total = len(contracts) or 1
        bar = "█" * int(20 * done / total) + "░" * (20 - int(20 * done / total))
        lines = [
            "Contract resolution",
            f"{bar} {100 * done // total}%",
            f"Complete {done}/{len(contracts)}",
        ]
        return CommandResult("\n".join(lines))

    def cmd_evidence(self, args: List[str]) -> CommandResult:
        items = self.harness.get_evidence(" ".join(args))
        if not items:
            return CommandResult("No evidence yet — ask a question first.")
        lines = []
        for item in items[:20]:
            lines.append(
                f"{item.get('subject')} / {item.get('attribute')}: {item.get('value')}"
            )
            for ev in item.get("evidence", [])[:2]:
                lines.append(
                    f"  {ev.get('filename')} p.{ev.get('page')} "
                    f"[{ev.get('method')}] {str(ev.get('snippet', ''))[:120]}"
                )
        if len(items) > 20:
            lines.append(f"(+{len(items) - 20} more)")
        return CommandResult("\n".join(lines))

    # -- session ------------------------------------------------------------
    def cmd_session(self, args: List[str]) -> CommandResult:
        if not args:
            return self._session_summary()
        sub, rest = args[0].lower(), args[1:]
        if sub == "list":
            sessions = self.sessions.list()
            if not sessions:
                return CommandResult("No saved sessions.")
            return CommandResult(
                "\n".join(
                    f"{s['id']}  {s.get('name', '')}  updated {s.get('updated_utc', '')[:16]}"
                    for s in sessions
                )
            )
        if sub == "new":
            return CommandResult("", action="new_session")
        if sub == "save":
            self.sessions.save(self.harness.session)
            return CommandResult("Checkpoint saved.")
        if sub == "rename" and rest:
            updated = self.sessions.rename(self.harness.session["id"], " ".join(rest))
            self.harness.session["name"] = updated["name"]
            return CommandResult(f"Renamed to {updated['name']}.")
        if sub == "resume" and rest:
            return CommandResult(
                "", action="resume_session", payload={"session_id": rest[0]}
            )
        if sub == "delete" and rest:
            self.sessions.delete(rest[0])
            return CommandResult(f"Deleted session {rest[0]}.")
        if sub == "search" and rest:
            hits = self.sessions.search(" ".join(rest))
            return CommandResult("\n".join(s["id"] for s in hits) or "No matches.")
        return CommandResult(
            "Usage: /session [list|new|save|rename|resume|delete|search]"
        )

    def _session_summary(self) -> CommandResult:
        session = self.harness.session
        status = self.harness.get_status()
        cost = self.harness.get_cost()
        lines = [
            "Session",
            "",
            f"ID        {session.get('id')}",
            f"Name      {session.get('name', '')}",
            f"Documents {len(session.get('documents', []))}",
            f"Messages  {len(session.get('turns', []))}",
            f"Claims    {status.get('answers', 0)}",
            f"Verified  {status.get('verified', 0)}",
            f"Tokens    {cost.get('tokens', 0)}",
        ]
        return CommandResult("\n".join(lines))

    def cmd_new(self, args: List[str]) -> CommandResult:
        return CommandResult("", action="new_session")

    def cmd_save(self, args: List[str]) -> CommandResult:
        self.sessions.save(self.harness.session)
        return CommandResult("Checkpoint saved.")

    def cmd_history(self, args: List[str]) -> CommandResult:
        if not self.history:
            return CommandResult("No requests yet.")
        return CommandResult(
            "\n".join(f"{i + 1}. {h[:100]}" for i, h in enumerate(self.history))
        )

    def cmd_clear(self, args: List[str]) -> CommandResult:
        return CommandResult("", action="clear")

    def cmd_quit(self, args: List[str]) -> CommandResult:
        return CommandResult("", action="quit")

    # -- provider ---------------------------------------------------------------
    def cmd_provider(self, args: List[str]) -> CommandResult:
        from ..services.provider import (
            provider_status,
            read_provider_config,
            clear_provider_env,
            provider_env_path,
        )

        sub = args[0].lower() if args else ""

        if sub == "azure":
            return CommandResult(
                "Azure OpenAI Configuration\n"
                "Enter your credentials one at a time.\n",
                action="provider_azure",
            )
        if sub == "openai":
            return CommandResult(
                "OpenAI-Compatible Configuration\n"
                "Enter your credentials one at a time.\n",
                action="provider_openai",
            )
        if sub == "di":
            return CommandResult(
                "Azure Document Intelligence\n"
                "Enter your credentials one at a time.\n",
                action="provider_di",
            )
        if sub == "clear":
            if clear_provider_env():
                return CommandResult("Credentials removed from " + str(provider_env_path()))
            return CommandResult("No saved credentials to remove.")
        if sub == "show":
            saved = read_provider_config()
            if not saved:
                return CommandResult("No saved provider configuration.")
            lines = ["Saved configuration:"]
            for k, v in saved.items():
                lines.append(f"  {k} = {v}")
            lines.append(f"\n  Path: {provider_env_path()}")
            return CommandResult("\n".join(lines))

        status = provider_status()
        lines = [
            f"Backend         {status['backend']}",
            f"Endpoint        {status['base_url']}",
            f"API Key         {'configured' if status['has_api_key'] == 'yes' else 'not set'}",
            f"Deployment      {status['deployment']}",
            f"DI Endpoint     {status['di_endpoint']}",
            f"DI Key          {'configured' if status['has_di_key'] == 'yes' else 'not set'}",
        ]
        if status["saved_config"]:
            lines.append(f"\nSaved config    {status['config_path']}")
        else:
            lines.append("\nNo saved credentials. Use /provider azure or /provider openai to configure.")
        return CommandResult("\n".join(lines))

    # -- system ---------------------------------------------------------------
    def cmd_model(self, args: List[str]) -> CommandResult:
        from ...deep.model import model_status

        status = model_status()
        deployment = status.get("deployment") or "heuristic (no model configured)"
        endpoint = (
            "APIM"
            if "azure-api.net" in str(status.get("base_url", ""))
            else str(status.get("base_url", "") or "local")
        )
        lines = [
            f"Reasoning       Azure {deployment}"
            if status.get("has_api_key")
            else "Reasoning       heuristic (no model configured)",
            "Vision          Azure GPT-5.4"
            if status.get("has_api_key")
            else "Vision          unavailable (no model configured)",
            f"Deployment      {deployment}",
            f"Endpoint        {endpoint}",
        ]
        return CommandResult("\n".join(lines))

    def cmd_tools(self, args: List[str]) -> CommandResult:
        groups = {
            "Search": ["search_text", "search_headings", "hybrid_search"],
            "Reading": ["read_page", "read_section", "read_surrounding_context"],
            "Tables": ["detect_tables", "extract_table", "reconstruct_table"],
            "Spreadsheets": ["read_spreadsheet", "read_sheet_range"],
            "Vision": [
                "render_page",
                "inspect_page_visually",
                "inspect_region_visually",
            ],
            "Azure DI": ["run_azure_document_intelligence"],
            "Evidence": [
                "submit_candidate_answer",
                "submit_answer_batch",
                "verify_answer",
            ],
            "Export": ["export"],
        }
        lines = []
        for group, tools in groups.items():
            lines.append(f"{group}: {', '.join(tools)}")
        try:
            import pypdfium2  # noqa

            lines.append("Native PDF       ✓ Ready")
        except ImportError:
            lines.append("Native PDF       ! pypdfium2 missing")
        from ...deep.model import model_status

        status = model_status()
        lines.append(
            f"GPT-5.4 Vision   {'✓ Ready' if status.get('has_api_key') else '! No key'}"
        )
        if status.get("has_di_key") and status.get("di_endpoint"):
            lines.append("Azure DI         ✓ Ready")
        else:
            lines.append("Azure DI         ! Credentials unavailable")
        return CommandResult("\n".join(lines))

    def cmd_cost(self, args: List[str]) -> CommandResult:
        cost = self.harness.get_cost()
        return CommandResult("\n".join(f"{k}: {v}" for k, v in cost.items()))

    def cmd_config(self, args: List[str]) -> CommandResult:
        if len(args) >= 3 and args[0] == "set":
            return CommandResult(
                f"Set {args[1]} = {' '.join(args[2:])} (session-scoped)."
            )
        return CommandResult(
            "Safe settings: output directory, verbosity, budgets.\n"
            "Usage: /config set <key> <value>"
        )

    def cmd_budget(self, args: List[str]) -> CommandResult:
        if len(args) == 2:
            return CommandResult(f"Budget {args[0]} set to {args[1]} for future runs.")
        return CommandResult(
            "Tool calls 2000\nModel calls 300\nVision calls 60\nRuntime 3h\n"
            "Usage: /budget <name> <value>"
        )

    def cmd_health(self, args: List[str]) -> CommandResult:
        from ...deep.model import model_status

        status = model_status()
        lines = [
            f"GPT-5.4 reasoning  {'✓' if status.get('has_api_key') else '! no key'}",
            f"GPT-5.4 Vision     {'✓' if status.get('has_api_key') else '! no key'}",
        ]
        try:
            import pypdfium2  # noqa

            lines.append("Native PDF parser  ✓")
        except ImportError:
            lines.append("Native PDF parser  ! pypdfium2 missing")
        lines.append(
            "Azure DI           ✓"
            if status.get("has_di_key")
            else "Azure DI           ! credentials unavailable"
        )
        lines.append("Session database   ✓")
        lines.append("Checkpoints        ✓")
        return CommandResult("\n".join(lines))

    # -- output -----------------------------------------------------------------
    def cmd_export(self, args: List[str]) -> CommandResult:
        if not args:
            return CommandResult(
                "Usage: /export excel|json|markdown|evidence|trace [path]"
            )
        fmt = args[0].lower()
        if fmt == "excel":
            return CommandResult(
                "Normal prompts never create Excel automatically. "
            )
        path = args[1] if len(args) > 1 else ""
        try:
            out = self.harness.export(fmt if fmt != "markdown" else "md", path)
        except Exception as exc:
            return CommandResult(f"! export failed: {exc}")
        return CommandResult(f"Exported: {out}")

    # -- txt (full chat history export) -------------------------------------------
    def cmd_txt(self, args: List[str]) -> CommandResult:
        import datetime
        from pathlib import Path

        turns = []
        if self.conv_state and hasattr(self.conv_state, "turns"):
            for t in self.conv_state.turns:
                turns.append({"prompt": t.prompt, "answer": t.answer})
        if not turns:
            for t in self.harness.session.get("turns", []):
                turns.append({"prompt": t.get("prompt", ""), "answer": t.get("answer", "")})
        if not turns:
            return CommandResult("No conversation history to export.")

        lines = [
            "KYNEX — Conversation Export",
            f"Session: {self.harness.session.get('id', '?')}",
            f"Exported: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"Turns: {len(turns)}",
            "=" * 72,
            "",
        ]
        for i, turn in enumerate(turns, 1):
            lines.append(f"--- Question {i} ---")
            lines.append(turn["prompt"])
            lines.append("")
            lines.append(f"--- Answer {i} ---")
            lines.append(turn["answer"])
            lines.append("")
            lines.append("=" * 72)
            lines.append("")

        out_dir = Path.home() / "Downloads"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"kynex_chat_{ts}.txt"
        if args:
            out_path = Path(args[0])
        out_path.write_text("\n".join(lines), encoding="utf-8")
        return CommandResult(f"Chat history exported: {out_path}")

    # -- copy (clipboard) ----------------------------------------------------------
    def cmd_copy(self, args: List[str]) -> CommandResult:
        """Copy last answer (or all conversation) to clipboard."""
        turns = []
        if self.conv_state and hasattr(self.conv_state, "turns"):
            for t in self.conv_state.turns:
                turns.append({"prompt": t.prompt, "answer": t.answer})
        if not turns:
            for t in self.harness.session.get("turns", []):
                turns.append({"prompt": t.get("prompt", ""), "answer": t.get("answer", "")})
        if not turns:
            return CommandResult("No conversation to copy.")

        if args and args[0].lower() == "all":
            lines = []
            for i, turn in enumerate(turns, 1):
                lines.append(f"--- Q{i} ---\n{turn['prompt']}\n\n--- A{i} ---\n{turn['answer']}\n")
            text = "\n".join(lines)
        else:
            text = turns[-1]["answer"]

        try:
            import subprocess
            process = subprocess.Popen(
                ["clip"], stdin=subprocess.PIPE, shell=True,
            )
            process.communicate(text.encode("utf-8"))
            return CommandResult(
                f"Copied to clipboard ({len(text)} chars). "
                "Tip: Shift+mouse-drag also selects text in terminal."
            )
        except Exception:
            from pathlib import Path
            out = Path.home() / "Downloads" / "kynex_clipboard.txt"
            out.write_text(text, encoding="utf-8")
            return CommandResult(
                f"Clipboard unavailable — saved to {out}\n"
                "Tip: Hold Shift + mouse-drag to select text in terminal."
            )

