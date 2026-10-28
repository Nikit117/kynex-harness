"""KYNEX — Autonomous Document Computer TUI.

Polished terminal-native experience for document intelligence.
Launch with ``kynex`` (interactive) or ``kynex ask`` (scriptable).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .sanitize import sanitize_text
from .services.events import EventBus
from .services.harness import HarnessService
from .services.sessions import SessionService


def build_services(session: Dict[str, Any], data_dir=None, bus=None):
    from .services.sessions import sessions_dir

    base = Path(data_dir) if data_dir else sessions_dir().parent
    bus = bus or EventBus()
    harness = HarnessService(session, data_dir=base, bus=bus)
    return harness, None, bus


class HarnessApp:
    """Textual application wrapper. Imported lazily so service tests avoid TTY."""

    def __init__(
        self, session, sessions: SessionService, initial_path: str = ""
    ) -> None:
        self.session = session
        self.sessions = sessions
        self.initial_path = initial_path

    def run(self) -> int:
        app = create_textual_app(self.session, self.sessions, self.initial_path)
        app.run()
        return 0


def create_textual_app(session, sessions, initial_path: str = ""):
    """Build the full Textual App (headless-testable via run_test pilot)."""
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.widgets import Input

    from .commands import CommandRegistry
    from .conversation_state import ConversationState
    from .router import ConversationRouter, Intent, IntentCompiler
    from .widgets.activity import ActivityIndicator
    from .widgets.conversation import ConversationView
    from .widgets.prompt import HarnessInput
    from .widgets.status_bar import StatusBar

    class _App(App):
        CSS = """
        Screen {
            layout: vertical;
        }
        #banner {
            height: auto;
            padding: 1 2;
            background: $primary-background;
        }
        #body {
            height: 1fr;
        }
        #activity {
            height: auto;
            max-height: 6;
        }
        StatusBar {
            dock: top;
            height: 1;
            background: $primary-background;
            color: $text-muted;
        }
        ConversationView {
            height: 1fr;
        }
        HarnessInput {
            dock: bottom;
            margin: 0 1;
        }
        """
        BINDINGS = [
            Binding("ctrl+c", "interrupt", "Cancel/Quit", show=False),
            Binding("ctrl+l", "clear_display", "Clear", show=False),
            Binding("ctrl+p", "command_palette", "Commands", show=False),
        ]

        def __init__(self) -> None:
            super().__init__()
            self.session_obj = session
            self.sessions_obj = sessions
            self.initial_path = initial_path
            self.bus = EventBus()
            self.harness, _, _ = build_services(
                session, bus=self.bus
            )
            self.conv_state = ConversationState()
            self.registry = CommandRegistry(self.harness, sessions, conv_state=self.conv_state)
            self.worker = None
            self.worker_running = False
            self.pending_choice = None
            self.quit_armed = False
            self.verbosity = "normal"
            self.router = ConversationRouter()
            self.intent_compiler = IntentCompiler()
            self._last_intent: Optional[Intent] = None
            self._current_synthesis = False
            self._streaming_bubble = None
            self._streaming_text = ""

        def compose(self) -> ComposeResult:
            yield StatusBar()
            yield ConversationView(id="conversation")
            yield ActivityIndicator(id="activity")
            yield HarnessInput(id="prompt")

        async def on_mount(self) -> None:
            self.title = "KYNEX"
            self.sub_title = "Autonomous Document Computer"
            conv = self.query_one("#conversation", ConversationView)
            status = self.query_one(StatusBar)
            status.session_name = session.get("name") or session.get("id", "")
            status.doc_count = len(session.get("documents", []))

            conv.add_welcome_banner(
                session_name=session.get("name") or session.get("id", ""),
                doc_count=len(session.get("documents", [])),
            )

            if self.initial_path:
                result = self.registry.dispatch(f"/add {self.initial_path}")
                conv.add_system_message(sanitize_text(result.text))
                if result.action == "add":
                    self._run_add_in_background(result.payload.get("staged", []))
                else:
                    self._update_workspace_stats()
            else:
                recent = self.sessions_obj.most_recent()
                if recent and recent["id"] != self.session_obj["id"]:
                    self.pending_choice = {
                        "kind": "resume",
                        "prompt": (
                            f"Recent session: {recent.get('name') or recent['id']} "
                            f"({len(recent.get('documents', []))} docs). "
                            "[R]esume / [N]ew"
                        ),
                        "recent": recent,
                    }
                    conv.add_system_message(self.pending_choice["prompt"])

            self.set_interval(0.15, self.drain_events)
            self.query_one("#prompt", HarnessInput).focus()

        def drain_events(self) -> None:
            try:
                activity = self.query_one("#activity", ActivityIndicator)
            except Exception:
                return
            try:
                conv = self.query_one("#conversation", ConversationView)
            except Exception:
                return
            for event in self.bus.drain(50):
                if event.kind == "answer_stream_started":
                    self._streaming_text = ""
                    self._streaming_bubble = conv.start_streaming_message()
                    continue
                if event.kind == "answer_stream_delta":
                    self._streaming_text += event.detail
                    if self._streaming_bubble is not None:
                        conv.update_streaming_message(
                            self._streaming_bubble, self._streaming_text
                        )
                    continue
                if event.kind in (
                    "answer_stream_completed", "answer_stream_cancelled"
                ):
                    if self._streaming_bubble is not None:
                        citations = event.data.get("citations")
                        conv.finalize_streaming_message(
                            self._streaming_bubble, citations
                        )
                    self._streaming_bubble = None
                    self._streaming_text = ""
                    continue
                if self.verbosity == "compact" and event.kind in (
                    "tool_started", "tool_finished",
                ):
                    continue
                activity.on_event_kind(event.kind, event.title, event.detail)

        async def on_input_submitted(self, message: Input.Submitted) -> None:
            line = message.value
            message.input.value = ""
            self.quit_armed = False

            prompt_widget = self.query_one("#prompt", HarnessInput)
            prompt_widget.push_history(line)

            conv = self.query_one("#conversation", ConversationView)
            if self.pending_choice is not None:
                conv.add_user_message(line)
                await self.resolve_choice(line.strip().lower())
                return
            if not line.strip():
                return

            conv.add_user_message(line)

            intent = self.router.classify(line, self._last_intent)
            self._last_intent = intent

            if intent == Intent.COMMAND or line.strip().startswith("/"):
                result = self.registry.dispatch(line)
                if result.text:
                    conv.add_assistant_message(result.text)
                if result.action in ("ask", "retry"):
                    self.run_long(result)
                elif result.action == "add":
                    self._run_add_in_background(result.payload.get("staged", []))
                elif result.action == "new_session":
                    self.fresh_session()
                elif result.action == "resume_session":
                    self.resume_session(result.payload.get("session_id", ""))
                elif result.action in ("provider_azure", "provider_openai", "provider_di"):
                    self._start_provider_setup(result.action, conv)
                elif result.action == "clear":
                    await self.action_clear_display()
                elif result.action == "quit":
                    await self.action_quit_protected()
                self._update_workspace_stats()
            elif intent == Intent.SYNTHESIS and self.conv_state.has_result_for_synthesis():
                self._handle_synthesis(line)
            elif intent == Intent.CHAT and not self.router.has_documents:
                self._handle_chat(line, conv)
            else:
                if not self.router.has_documents and intent != Intent.CHAT:
                    conv.add_system_message(
                        "No documents loaded. Use /add <path> to add "
                        "documents, then ask your question."
                    )
                    return
                resolved = self.conv_state.resolve_referent(line)
                enriched = self._enrich_prompt(resolved)
                self.run_long_type("ask", {"prompt": enriched})

        def _handle_synthesis(self, prompt: str) -> None:
            """Synthesis fast path: reformat existing results without reinvestigation."""
            ctx = self.conv_state.get_synthesis_context()
            context_prompt = (
                f"Previous question: {ctx['last_question']}\n"
                f"Previous answer:\n{ctx['last_answer']}\n\n"
                f"User request: {prompt}\n\n"
                "Reformat or filter the previous answer as requested. "
                "Do NOT investigate new documents — use only the data above."
            )
            self._current_synthesis = True
            self.run_long_type("ask", {"prompt": context_prompt})

        def _handle_chat(self, prompt: str, conv) -> None:
            """Respond to casual chat when no documents are loaded."""
            lower = prompt.strip().lower()
            greetings = {"hi", "hey", "hello", "sup", "yo"}
            if any(lower.startswith(g) for g in greetings):
                conv.add_assistant_message(
                    "Hey! I'm KYNEX — your document intelligence assistant.\n\n"
                    "Add documents with `/add <path>` and I'll investigate them for you. "
                    "Type `/help` to see all commands."
                )
                return
            doc_words = {"document", "file", "pdf", "summary", "extract", "find", "search"}
            if any(w in lower for w in doc_words):
                conv.add_assistant_message(
                    "I'd love to help with that, but no documents are loaded yet.\n\n"
                    "Use `/add <path>` to add a PDF or folder, then ask me anything about it."
                )
                return
            conv.add_assistant_message(
                "I'm a document intelligence engine — I work best when you "
                "load documents with `/add <path>`.\n\n"
                "Once loaded, I can answer questions, extract structured data, "
                "compare documents and interrogate them. "
                "Type `/help` for all commands."
            )

        def _enrich_prompt(self, prompt: str) -> str:
            """Prepend document context for vague prompts when docs are loaded."""
            if not self.router.has_documents:
                return prompt
            docs = self.harness.session.get("documents", [])
            if not docs:
                return prompt
            doc_names = [d.get("filename", "?") for d in docs[:5]]
            doc_list = ", ".join(doc_names)
            return (
                f"[Context: {len(docs)} document(s) loaded: {doc_list}. "
                f"Answer based on these documents.]\n\n{prompt}"
            )

        async def resolve_choice(self, choice: str) -> None:
            conv = self.query_one("#conversation", ConversationView)
            if self.pending_choice and self.pending_choice["kind"] == "provider":
                self._handle_provider_input(choice, conv)
                return
            pending, self.pending_choice = self.pending_choice, None
            if pending["kind"] == "resume":
                if choice.startswith("r"):
                    self.resume_session(pending["recent"]["id"])
                else:
                    conv.add_system_message("New session started.")

        _PROVIDER_FIELDS = {
            "provider_azure": [
                ("ADH_MODEL_BASE_URL", "Endpoint", "https://your-resource.openai.azure.com"),
                ("ADH_MODEL_API_KEY", "API Key", ""),
                ("ADH_MODEL_DEPLOYMENT", "Deployment", "gpt-5.4"),
                ("ADH_MODEL_API_VERSION", "API Version (Enter for 2024-10-21)", "2024-10-21"),
            ],
            "provider_openai": [
                ("ADH_MODEL_BASE_URL", "Base URL", "https://api.openai.com/v1"),
                ("ADH_MODEL_API_KEY", "API Key", ""),
                ("ADH_MODEL_DEPLOYMENT", "Model", "gpt-5.4"),
            ],
            "provider_di": [
                ("ADH_DI_ENDPOINT", "DI Endpoint", "https://your-resource.cognitiveservices.azure.com"),
                ("ADH_DI_API_KEY", "DI API Key", ""),
            ],
        }

        def _start_provider_setup(self, action: str, conv) -> None:
            fields = self._PROVIDER_FIELDS[action]
            backend = "azure" if action == "provider_azure" else "openai" if action == "provider_openai" else ""
            self.pending_choice = {
                "kind": "provider",
                "action": action,
                "fields": fields,
                "step": 0,
                "collected": {},
                "backend": backend,
            }
            label, placeholder = fields[0][1], fields[0][2]
            hint = f" (default: {placeholder})" if placeholder else ""
            conv.add_system_message(f"  {label}{hint}:")

        def _handle_provider_input(self, value: str, conv) -> None:
            p = self.pending_choice
            fields = p["fields"]
            step = p["step"]
            env_key, label, default = fields[step]

            actual = value.strip() if value.strip() else default
            if not actual and env_key in ("ADH_MODEL_API_KEY", "ADH_DI_API_KEY", "ADH_MODEL_BASE_URL", "ADH_DI_ENDPOINT"):
                conv.add_system_message(f"  {label} is required. Please enter a value:")
                return

            p["collected"][env_key] = actual
            p["step"] += 1

            if p["step"] < len(fields):
                next_label = fields[p["step"]][1]
                next_default = fields[p["step"]][2]
                hint = f" (default: {next_default})" if next_default else ""
                conv.add_system_message(f"  {next_label}{hint}:")
                return

            from .services.provider import save_provider_env
            config = p["collected"]
            if p["backend"]:
                config["ADH_MODEL_BACKEND"] = p["backend"]
            path = save_provider_env(config)
            self.pending_choice = None

            from ..deep.model import ensure_azure_env, model_status
            ensure_azure_env()
            status = model_status()

            deployment = status.get("deployment") or "unknown"
            status_bar = self.query_one(StatusBar)
            status_bar.model_name = deployment

            masked = []
            for k, v in config.items():
                if "KEY" in k.upper():
                    masked.append(f"  {k} = {v[:4]}...{v[-4:]}" if len(v) > 8 else f"  {k} = ****")
                else:
                    masked.append(f"  {k} = {v}")

            conv.add_system_message(
                "Configuration saved.\n\n"
                + "\n".join(masked)
                + f"\n\n  Path: {path}"
                + (f"\n  Model: {deployment}" if status.get("has_api_key") else "")
            )

        def fresh_session(self) -> None:
            self.checkpoint()
            conv = self.query_one("#conversation", ConversationView)
            new_session = self.sessions_obj.create()
            self.session_obj = new_session
            self.harness, _, _ = build_services(
                new_session, bus=self.bus
            )
            self.conv_state = ConversationState()
            self.registry = CommandRegistry(
                self.harness, self.sessions_obj, conv_state=self.conv_state
            )
            self.router = ConversationRouter()
            self._last_intent = None
            self._current_synthesis = False
            self._update_status_bar(new_session)
            conv.add_system_message(
                f"New session {new_session['id']} (previous checkpointed)."
            )

        def resume_session(self, session_id: str) -> None:
            conv = self.query_one("#conversation", ConversationView)
            try:
                loaded = self.sessions_obj.load(session_id)
            except Exception as exc:
                conv.add_system_message(f"! {exc}")
                return
            self.session_obj = loaded
            self.harness, _, _ = build_services(
                loaded, bus=self.bus
            )
            self.harness.restore_state()
            self.registry = CommandRegistry(
                self.harness, self.sessions_obj, conv_state=self.conv_state
            )
            self.registry.history = [
                t.get("prompt", "") for t in loaded.get("turns", [])
            ]
            has_docs = bool(loaded.get("documents"))
            has_state = bool(loaded.get("turns"))
            self.router = ConversationRouter(
                has_documents=has_docs, has_session_state=has_state
            )
            conv_data = loaded.get("conversation_state")
            if conv_data:
                self.conv_state = ConversationState.from_dict(conv_data)
            else:
                self.conv_state = ConversationState()
            self._current_synthesis = False
            self._update_status_bar(loaded)
            conv.add_system_message(
                f"Resumed session {loaded.get('name') or loaded['id']} "
                f"({len(loaded.get('documents', []))} docs)."
            )

        def _update_status_bar(self, sess: Dict[str, Any]) -> None:
            status = self.query_one(StatusBar)
            status.session_name = sess.get("name") or sess.get("id", "")
            status.doc_count = len(sess.get("documents", []))

        def _update_workspace_stats(self) -> None:
            try:
                stats = self.harness.workspace_stats()
            except Exception:
                stats = {}
            status = self.query_one(StatusBar)
            doc_count = stats.get("total_documents", 0)
            if doc_count == 0:
                doc_count = len(self.harness.session.get("documents", []))
            status.doc_count = doc_count
            status.page_count = stats.get("total_pages", 0)
            self.router.update_state(
                has_documents=doc_count > 0
            )

        def _run_add_in_background(self, staged_names: List[str]) -> None:
            activity = self.query_one("#activity", ActivityIndicator)
            status = self.query_one(StatusBar)
            activity.start("Parsing documents")
            status.status_text = "Parsing"

            async def _do_add():
                conv = self.query_one("#conversation", ConversationView)
                try:
                    world = await self.run_in_thread(self.harness.ensure_world)
                    added_pages = 0
                    for doc_id, profile in world.profiles.items():
                        if profile.filename in staged_names:
                            added_pages += profile.page_count
                    self.call_from_thread(
                        conv.add_system_message,
                        f"Ready — {len(staged_names)} document(s), "
                        f"{added_pages} pages parsed.",
                    )
                except Exception as exc:
                    self.call_from_thread(
                        conv.add_system_message,
                        f"Parse warning: {type(exc).__name__}: {str(exc)[:200]}",
                    )
                finally:
                    activity.stop()
                    status.status_text = "Ready"
                    self.call_from_thread(self._update_workspace_stats)

            self.run_worker(_do_add(), thread=True)

        async def action_quit_protected(self) -> None:
            self.checkpoint()
            conv = self.query_one("#conversation", ConversationView)
            conv.add_system_message("Session checkpoint saved.")
            self.exit(0)

        def checkpoint(self) -> None:
            try:
                self.harness.save_state()
            except Exception:
                pass
            self.harness.session["conversation_state"] = self.conv_state.to_dict()
            self.sessions_obj.save(self.harness.session)

        def action_interrupt(self) -> None:
            conv = self.query_one("#conversation", ConversationView)
            activity = self.query_one("#activity", ActivityIndicator)
            if self.worker_running and self.worker is not None:
                try:
                    self.worker.cancel()
                except Exception:
                    pass
                self.worker_running = False
                activity.stop("Interrupted")
                self.checkpoint()
                status = self.query_one(StatusBar)
                status.status_text = "Ready"
                conv.add_system_message(
                    "^C\nRun interrupted. Checkpoint saved; verified findings preserved."
                )
            elif self.quit_armed:
                self.checkpoint()
                self.exit(0)
            else:
                self.quit_armed = True
                conv.add_system_message(
                    "Press Ctrl+C again to quit (session will checkpoint)."
                )

        async def action_clear_display(self) -> None:
            conv = self.query_one("#conversation", ConversationView)
            await conv.remove_children()
            conv.add_system_message("Display cleared. Session state preserved.")

        async def action_command_palette(self) -> None:
            from .widgets.command_palette import CommandPalette

            result = await self.push_screen_wait(CommandPalette())
            if result:
                prompt = self.query_one("#prompt", HarnessInput)
                prompt.value = result

        def run_long(self, result) -> None:
            self.run_long_type(result.action, result.payload or {})

        def run_long_type(self, kind: str, payload: Dict[str, Any]) -> None:
            conv = self.query_one("#conversation", ConversationView)
            self.worker_running = True
            activity = self.query_one("#activity", ActivityIndicator)
            status = self.query_one(StatusBar)

            if kind == "ask":
                activity.start("Understanding request")
                status.status_text = "Working"
            else:
                activity.start("Working")
                status.status_text = "Working"

            self.worker = self.run_worker(
                self.execute(kind, payload), thread=True
            )

        async def execute(self, kind: str, payload: Dict[str, Any]) -> None:
            conv = self.query_one("#conversation", ConversationView)
            activity = self.query_one("#activity", ActivityIndicator)
            status = self.query_one(StatusBar)
            try:
                if kind == "ask":
                    answer = await self.run_in_thread(
                        self.harness.ask, payload.get("prompt", "")
                    )
                    self.call_from_thread(self.render_answer, answer)
                elif kind == "retry":
                    outcome = await self.run_in_thread(
                        self.harness.retry_unresolved,
                        payload.get("contract_id", ""),
                    )
                    conv.add_system_message(sanitize_text(str(outcome)))
            except Exception as exc:
                conv.add_system_message(
                    f"! {type(exc).__name__}: {sanitize_text(str(exc))[:300]}"
                )
            finally:
                self.worker_running = False
                activity.stop()
                status.status_text = "Ready"
                try:
                    self.checkpoint()
                except Exception:
                    pass

        async def run_in_thread(self, func, *args):
            import asyncio as _asyncio
            return await _asyncio.to_thread(func, *args)

        def render_answer(self, answer: Dict[str, Any]) -> None:
            conv = self.query_one("#conversation", ConversationView)
            citations = answer.get("citations", [])
            text = answer.get("text", "")
            prompt = answer.get("prompt", "")
            tool_steps = answer.get("tool_steps", [])
            if tool_steps:
                elapsed = answer.get("elapsed_seconds", 0)
                summary = f"Document Q&A ({elapsed}s)" if elapsed else "Document Q&A"
                conv.add_collapsible_activity(summary, tool_steps)
            conv.add_assistant_message(text, citations)
            self.registry.history.append(prompt)
            for c in citations:
                self.registry.citations[c["id"]] = c

            reinvestigated = not self._current_synthesis
            self._current_synthesis = False
            intent_str = self._last_intent.value if self._last_intent else ""
            self.conv_state.record_turn(
                prompt=prompt,
                answer=text,
                intent=intent_str,
                citations=citations,
                reinvestigated=reinvestigated,
            )

            self.router.update_state(
                has_session_state=bool(text)
            )
            cost = self.harness.get_cost()
            tokens = cost.get("tokens", 0)
            status = self.query_one(StatusBar)
            if tokens > 0:
                estimated = tokens * 0.000003
                status.cost = f"${estimated:.2f}"

        def show_trace(self, kind: str) -> None:
            conv = self.query_one("#conversation", ConversationView)
            observations = self.harness.get_trace(30)
            if not observations:
                conv.add_assistant_message("No trace events yet.")
                return
            lines = []
            for obs in observations:
                tool = obs.get("tool", "?")
                summary = str(obs.get("summary", ""))[:100]
                lines.append(f"{tool} — {summary}")
            conv.add_assistant_message("\n".join(lines))

    return _App()


# -- Engine command (spec §41) --
def _cmd_engine(registry, args):
    """Show or switch extraction engine."""
    from .commands import CommandResult

    lines = [
        "Engine Information",
        "",
        "Active     investigative (tool-calling)",
        "Q&A        Map → Target → Extract → Validate",
    ]
    return CommandResult("\n".join(lines))


def main(argv=None) -> int:
    """Entry point: ``kynex`` (TUI), ``kynex run|ask`` (scriptable)."""
    from ..deep.model import ensure_azure_env

    ensure_azure_env()
    parser = argparse.ArgumentParser(
        prog="kynex", description="Autonomous document intelligence."
    )
    argv = list(argv if argv is not None else sys.argv[1:])
    initial_path = ""
    if (
        argv
        and not argv[0].startswith("-")
        and argv[0] not in ("run", "ask", "-h", "--help")
    ):
        initial_path = argv.pop(0)
    sub = parser.add_subparsers(dest="command")
    run_p = sub.add_parser("run", help="investigate documents (no workbook)")
    run_p.add_argument("input", help="file or folder")
    run_p.add_argument("--prompt", required=True)
    run_p.add_argument("--session", default="")
    ask_p = sub.add_parser("ask", help="ask one question")
    ask_p.add_argument("prompt")
    ask_p.add_argument("--session", default="")
    args = parser.parse_args(argv)

    sessions = SessionService()
    if args.command in ("run", "ask"):
        sess = sessions.load(args.session) if args.session else sessions.create()
        harness, _, _bus = build_services(sess)
        if args.session:
            harness.restore_state()
        if args.command == "run" and args.input:
            print(harness.add_documents([args.input])["staged"])
        prompt = args.prompt
        answer = harness.ask(prompt)
        print(answer["text"])
        for citation in answer.get("citations", []):
            print(
                f"[{citation['id']}] {citation['document']} p.{citation.get('page')}"
            )
        harness.save_state()
        sessions.save(harness.session)
        return 0
    # Interactive TUI.
    sess = sessions.create()
    return HarnessApp(sess, sessions, initial_path=initial_path).run()


if __name__ == "__main__":
    sys.exit(main())
