"""Primary autonomous document investigator. Owns control flow; tools serve it.

Loop: goal -> gaps -> candidates -> policy -> execute -> mine -> verify ->
contradictions -> gates -> targeted recovery | honest finish.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Set, Tuple

from ..core.config import RuntimeConfig
from ..core.schemas import (
    Claim,
    ExtractionMethod,
    InformationNeed,
    InvestigationGoal,
    SourceRegion,
    VerificationState,
    make_id,
    reset_id_counters,
)
from ..evidence.canonical import CanonicalStore
from ..evidence.claims import ClaimStore
from ..evidence.contradictions import (
    Contradiction,
    detect_contradictions,
    mark_unresolvable,
)
from ..evidence.graph import EvidenceGraph
from ..memory.episodic import EpisodicMemory
from ..memory.evidence import EvidenceMemory
from ..memory.tasks import TaskLedger
from ..memory.working import WorkingMemory
from ..models.gateway import ModelGateway
from ..models.heuristic import is_excluded, need_addressed
from ..observability import events
from ..observability.metrics import RunMetrics
from ..observability.traces import RunTrace
from ..output import compiler as output_compiler
from ..specialists import research as research_specialist
from ..specialists import tables as table_specialist
from ..specialists import verifier as verifier_specialist
from ..tools.computer import DocumentComputer
from ..tools.registry import Observation
from ..verification import (
    VerificationHarness,
    VerificationReport,
    build_default_harness,
)
from ..verification.base import VerificationContext
from ..workspace.world_model import WorldModel
from .budgets import RunBudgets
from .executor import ToolExecutor
from .planner import Action, Gaps, InvestigationPlanner
from .policy import ActionPolicy, PolicyHistory

MAX_MINED_ITEMS_PER_OBS = 40
MAX_RECOVERY_ROUNDS = 3


@dataclass
class InvestigationResult:
    status: str  # complete | budget_exhausted | no_progress
    goal: InvestigationGoal
    claims: List[Claim] = field(default_factory=list)
    contradictions: List[Contradiction] = field(default_factory=list)
    verification: Optional[VerificationReport] = None
    trace: Optional[RunTrace] = None
    metrics: Optional[RunMetrics] = None
    output: Dict[str, Any] = field(default_factory=dict)
    budgets: Dict[str, Any] = field(default_factory=dict)
    completeness: Dict[str, Any] = field(default_factory=dict)


class AutonomousInvestigator:
    def __init__(
        self,
        world: WorldModel,
        config: Optional[RuntimeConfig] = None,
        render_dir: Optional[str] = None,
        run_id: str = "run_001",
        domain_validators: Optional[list] = None,
        ablation=None,
    ) -> None:
        self.world = world
        self.config = config or RuntimeConfig.from_env("")
        self.budgets = RunBudgets.with_limits(self.config.budgets)
        self.gateway = ModelGateway(self.config)
        self.computer = DocumentComputer(
            world,
            self.budgets.usage,
            render_dir=render_dir,
            config=self.config,
            gateway=self.gateway,
        )
        self.planner = InvestigationPlanner(world, self.computer, config=self.config)
        self.policy = ActionPolicy(self.config)
        self.history = PolicyHistory()
        self.executor = ToolExecutor(
            self.computer, max_workers=self.config.max_parallel_tools
        )
        self.harness: VerificationHarness = build_default_harness(
            self.config.entailment_min_overlap,
            self.config.sufficiency_min_verified_ratio,
        )
        self.domain_validators = list(domain_validators or [])
        # Ablation switches (§28): every mechanism can be disabled for measurement.
        from ..eval.ablations.flags import AblationFlags

        self.flags = ablation or AblationFlags()
        if not self.flags.use_coverage:
            self.harness = VerificationHarness(
                [g for g in self.harness.gates if g.name != "coverage"]
            )
        # memory systems
        self.working = WorkingMemory()
        self.ledger = TaskLedger()
        self.evidence_memory = EvidenceMemory()
        self.episodic = EpisodicMemory()
        # evidence systems
        reset_id_counters()
        self.store = ClaimStore()
        self.graph = EvidenceGraph()
        self.contradictions: List[Contradiction] = []
        # observability
        self.trace = RunTrace(run_id=run_id)
        self.metrics = RunMetrics()
        # loop state
        self.searched: Set[str] = set()
        self.visited_pages: Set[Tuple[str, int]] = set()
        self.visited_sheets: Set[Tuple[str, str]] = set()
        self.specialist_docs: Set[str] = set()
        self.tables_inspected: Set[str] = set()
        self._recovery_tool_mark = 0
        self.recovery_rounds = 0
        self.last_failure_signature: Tuple = ()
        self.report: VerificationReport = VerificationReport(passed=False)
        self._used_model_spec = False

    # -- public ---------------------------------------------------------------
    def run(
        self, prompt: str, extra_needs: Optional[List[InformationNeed]] = None
    ) -> InvestigationResult:
        self.trace.objective = prompt
        self.trace.append(events.PLAN, f"goal received: {prompt[:160]}")
        goal = self.gateway.parse_goal(prompt)
        spec = self._compile_spec(prompt)
        goal.spec = spec.to_dict()
        if extra_needs:
            # Structured field needs (e.g. sanitized request schema) join the
            # prompt-parsed needs; both drive the same loop.
            goal.needs.extend(extra_needs)
        self.trace.append(
            events.PLAN,
            f"interpreted {len(goal.needs)} need(s), "
            f"exclusions={self._exclusions(goal)}",
            {
                "needs": [n.attribute for n in goal.needs],
                "spec": spec.to_dict(),
                "spec_source": "model" if self._used_model_spec else "heuristic",
            },
        )
        for need in goal.needs:
            self.ledger.add(
                f"establish {need.subject}/{need.attribute}", need_id=need.need_id
            )
        hints = self.episodic.suggest(self._corpus_signature())
        if hints:
            for hint in hints:
                self.planner.boost_tools.update(hint.get("tools", []))
            self.trace.append(
                events.MEMORY,
                f"episodic hints applied: {[h['strategy'] for h in hints]}",
                {"tools_boosted": sorted(self.planner.boost_tools)},
            )
        self.working.reasoning_state = "investigating"
        no_progress = 0
        status = "complete"

        for iteration in range(self.config.budgets.max_iterations):
            if not self.budgets.usage.consume("iteration"):
                status = "budget_exhausted"
                self.trace.append(events.DONE, "iteration budget exhausted")
                break
            gaps = self._gaps(goal)
            candidates = self.planner.propose(goal, gaps)
            candidates.extend(self._think(goal, gaps, candidates))
            if self.flags.use_policy:
                selected, decisions = self.policy.select(
                    candidates, self.history, self.config.max_parallel_tools
                )
            else:  # ablation D-: blind first-k selection, still logged
                selected = candidates[: self.config.max_parallel_tools]
                decisions = []
            for decision in decisions:
                self.trace.append(
                    events.DECISION,
                    f"{decision.tool} score={decision.score} "
                    f"{'SELECT' if decision.selected else 'skip'}",
                    {"action_id": decision.action_id, **decision.components},
                )
            if not selected:
                no_progress += 1
                self.trace.append(events.PLAN, "policy selected no actions")
                if no_progress >= self.config.no_progress_rounds_stop:
                    status = "no_progress"
                    break
                self._verify_round(goal)
                continue
            pairs = self.executor.execute(selected)
            new_evidence = self._ingest(pairs, goal)
            self._verify_round(goal)
            self._contradiction_round()
            self._update_states(goal, selected)
            progressed = new_evidence or any(obs.status == "ok" for _, obs in pairs)
            no_progress = 0 if progressed else no_progress + 1
            if not self.flags.use_verification:
                continue  # ablation: run blind, measure at the end
            self.report = self._evaluate(goal)
            for result in self.report.results:
                self.trace.append(
                    events.GATE,
                    f"{result.gate} {'PASS' if result.passed else 'FAIL'}: "
                    f"{len(result.failures)} failure(s)",
                    {
                        "gate": result.gate,
                        "passed": result.passed,
                        "failures": result.failures[:5],
                    },
                )
            if self.report.passed:
                status = "complete"
                self._close_recovery_tasks("resolved: verification sufficient")
                self.trace.append(events.DONE, "verification sufficient; finishing")
                break
            if not self.flags.use_recovery:
                status = "no_progress"  # ablation H-: no targeted recovery
                break
            if self.report.recovery and self.budgets.ok():
                self.recovery_rounds += 1
                self.metrics.recovery_attempts += 1
                signature = tuple(sorted(self.report.failed_gates()))
                if (
                    signature == self.last_failure_signature
                    and self.recovery_rounds > 2
                ):
                    self.trace.append(
                        events.RECOVERY,
                        "failures repeating; forcing honest unresolved output",
                    )
                    self._force_honest_finish(goal)
                    self.report = self._evaluate(goal)
                    break
                self.last_failure_signature = signature
                if self.recovery_rounds > MAX_RECOVERY_ROUNDS:
                    self._force_honest_finish(goal)
                    self.report = self._evaluate(goal)
                    break
                for task in self.report.recovery:
                    task.attempts += 1
                    task.cost = self.metrics.tool_calls - self._recovery_tool_mark
                    self.trace.append(
                        events.RECOVERY,
                        f"{task.task_id}: {task.reason}",
                        {"claims": task.claim_ids, "actions": task.suggested_actions},
                    )
                    self.ledger.add(f"recover: {task.reason}")
                self._recovery_tool_mark = self.metrics.tool_calls
            else:
                if not self.budgets.ok():
                    status = "budget_exhausted"
                elif not self.report.passed:
                    status = "no_progress"
                break
            if no_progress >= self.config.no_progress_rounds_stop:
                status = "no_progress"
                break

        self._finalize_unresolved(goal)
        self.report = self._evaluate(goal)
        canonical = CanonicalStore.from_claims(
            self.store.all_claims(), self.store.evidence
        )
        filenames = {d.doc_id: d.filename for d in self.world.refs.values()}
        output = output_compiler.compile_result(
            self.store,
            self.contradictions,
            self.report,
            prompt,
            filenames,
            canonical,
            needs=goal.needs,
        )
        from .completeness import build_sweep_report

        sweep = build_sweep_report(
            self.world,
            goal.needs,
            self.searched,
            self.visited_pages,
            self.visited_sheets,
            self.tables_inspected,
            self.store.all_claims(),
        )
        output["completeness"] = sweep
        self.episodic.record(
            "autonomous-investigation",
            self._corpus_signature(),
            self._tools_used(),
            self.report.passed,
            cost=self.metrics.estimated_cost_usd,
            quality=output["stats"]["verified"] / max(output["stats"]["claims"], 1),
        )
        self.trace.append(
            events.DONE,
            f"status={status} verified={output['stats']['verified']} "
            f"unresolved={output['stats']['unresolved']}",
            {
                "status": status,
                "metrics": self.metrics.to_dict(),
                "completeness": sweep["assessment"],
                "budgets": self.budgets.usage.to_dict(),
                "need_states": {n.need_id: n.state for n in goal.needs},
            },
        )
        return InvestigationResult(
            status=status,
            goal=goal,
            claims=self.store.all_claims(),
            contradictions=list(self.contradictions),
            verification=self.report,
            trace=self.trace,
            metrics=self.metrics,
            output=output,
            budgets=self.budgets.usage.to_dict(),
            completeness=sweep,
        )

    # -- gaps -------------------------------------------------------------------
    def _compile_spec(self, prompt: str):
        from ..models.objective import compile_objective, validate_spec

        use_model = self.flags.use_model_reasoning
        gateway = self.gateway if use_model else None
        spec = compile_objective(prompt, gateway=gateway)
        self._used_model_spec = bool(
            use_model and self.gateway.model_reasoning_available()
        )
        errors = validate_spec(spec)
        if errors:
            self.trace.append(events.PLAN, f"spec validation: {errors}")
        return spec

    def _think(
        self, goal: InvestigationGoal, gaps: Gaps, existing: List[Action]
    ) -> List[Action]:
        """Model-driven action proposals merged with planner candidates."""
        if not self.flags.use_model_reasoning:
            return []
        if not self.gateway.model_reasoning_available():
            return []
        if not self.budgets.usage.consume("model"):
            return []
        valid_tools = {s.name for s in self.computer.registry.specs()}
        summary = json.dumps(
            {
                "needs": [
                    {
                        "subject": n.subject,
                        "attribute": n.attribute,
                        "required": n.required,
                    }
                    for n in gaps.unresolved_needs[:6]
                ],
                "claims": len(self.store.all_claims()),
                "contradictions": len(
                    [c for c in self.contradictions if c.status == "open"]
                ),
                "budgets": self.budgets.usage.to_dict(),
                "prior_actions": [
                    f"{a.tool}({str(sorted(a.args.items()))[:80]})"
                    for a in existing[:10]
                ],
            }
        )
        try:
            proposal = self.gateway.propose_actions(summary)
        except Exception as exc:
            self.trace.append(events.DECISION, f"model think failed: {exc}")
            return []
        actions = []
        for entry in proposal.get("actions", []):
            tool = entry.get("tool", "")
            if tool not in valid_tools or tool in ("calculate",):
                continue
            args = entry.get("args", {})
            if not isinstance(args, dict):
                continue
            args = self._coerce_args(tool, args)
            actions.append(
                Action(
                    action_id=make_id("act"),
                    tool=tool,
                    args=args,
                    reason=f"model: {entry.get('rationale', '')[:120]}",
                    rationale=str(entry.get("rationale", ""))[:300],
                    prior_gain=0.85,
                    target=f"model:{tool}:{str(sorted(args.items()))[:80]}",
                )
            )
        if actions:
            self.trace.append(
                events.DECISION,
                f"model focus: {proposal.get('focus', '')[:120]}",
                {"count": len(actions)},
            )
        return actions[:4]

    @staticmethod
    def _coerce_args(tool: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Light coercion for model-proposed args (str ints, paging params)."""
        coerced = dict(args)
        for key in ("top_k", "page", "window", "table_index", "dpi", "max_chars"):
            if key in coerced and isinstance(coerced[key], str):
                try:
                    coerced[key] = int(float(coerced[key]))
                except ValueError:
                    pass
        for key in (
            "query",
            "question",
            "doc_id",
            "heading",
            "sheet",
            "cell_range",
            "subject",
            "attribute",
            "claim_id",
            "expression",
        ):
            if key in coerced and not isinstance(coerced[key], str):
                coerced[key] = str(coerced[key])
        return coerced

    def _exclusions(self, goal: InvestigationGoal) -> List[str]:
        seen: List[str] = []
        for need in goal.needs:
            for exc in need.exclusions:
                if exc not in seen:
                    seen.append(exc)
        return seen

    def _close_recovery_tasks(self, outcome: str) -> None:
        """Record per-task outcome + approximate tool cost (§18/§32)."""
        for task in self.report.recovery if self.report else []:
            if not task.outcome:
                task.outcome = outcome
                task.cost = self.metrics.tool_calls - self._recovery_tool_mark
        self._recovery_tool_mark = self.metrics.tool_calls

    def _gaps(self, goal: InvestigationGoal) -> Gaps:
        claims = self.store.all_claims()
        unresolved_needs = [n for n in goal.needs if not need_addressed(n, claims)]
        open_contras = [c for c in self.contradictions if c.status == "open"]
        unverified = [
            c
            for c in self.store.all_claims()
            if c.verification == VerificationState.UNVERIFIED
        ]
        return Gaps(
            unresolved_needs=unresolved_needs,
            open_contradictions=open_contras,
            unverified_claims=unverified,
            recovery_tasks=list(self.report.recovery) if self.report else [],
            searched_queries=set(self.searched),
            visited_pages=set(self.visited_pages),
            visited_sheets=set(self.visited_sheets),
            claims_count=len(self.store.all_claims()),
            vision_targets=self._vision_targets(),
            tables_detected=set(self.tables_inspected),
        )

    def _vision_targets(self) -> List[Dict[str, Any]]:
        """Disputed/critical regions worth targeted GPT-5.4 Vision inspection."""
        targets: List[Dict[str, Any]] = []
        seen = set()
        interesting = [
            c
            for c in self.store.all_claims()
            if c.verification
            in (VerificationState.CONTRADICTED, VerificationState.UNCERTAIN)
        ]
        for claim in interesting[:6]:
            for eid in claim.evidence_ids[:2]:
                ev = self.store.get_evidence(eid)
                if ev is None:
                    continue
                region = ev.region
                if region.source_type not in (
                    "table",
                    "table_cell",
                    "region",
                    "page",
                    "image",
                    "figure",
                ):
                    continue
                key = (
                    region.document_id,
                    region.page,
                    tuple(region.bbox) if region.bbox else None,
                )
                if key in seen:
                    continue
                seen.add(key)
                targets.append(
                    {
                        "doc_id": region.document_id,
                        "page": region.page,
                        "bbox": list(region.bbox) if region.bbox else None,
                        "claim_ids": [claim.claim_id],
                        "reason": f"{claim.subject}/{claim.attribute} is "
                        f"{claim.verification.value}",
                    }
                )
        return targets

    # -- ingest --------------------------------------------------------------------
    def _ingest(
        self, pairs: List[Tuple[Action, Observation]], goal: InvestigationGoal
    ) -> bool:
        exclusions = self._exclusions(goal)
        needs = goal.needs
        new_evidence = False
        for action, obs in pairs:
            if self.flags.use_memory:
                self.history.note(action, obs.status)
            spec = next(
                (s for s in self.computer.registry.specs() if s.name == action.tool),
                None,
            )
            self.metrics.record_tool(
                action.tool,
                obs.latency_ms,
                data_chars=len(str(obs.data)),
                needs_model=bool(spec and spec.needs_model),
                needs_visual=bool(spec and spec.needs_visual),
            )
            if action.tool == "read_page" and obs.status == "ok":
                self.visited_pages.add(
                    (action.args.get("doc_id", ""), int(action.args.get("page", 0)))
                )
                self.metrics.pages_inspected += 1
            if action.tool == "read_sheet_range" and obs.status == "ok":
                self.visited_sheets.add(
                    (action.args.get("doc_id", ""), action.args.get("sheet", ""))
                )
            if action.tool == "search_text":
                self.searched.add(
                    f"{action.args.get('mode', 'hybrid')}:"
                    f"{action.args.get('query', '')}"
                )
            if action.tool == "search_headings":
                self.searched.add(f"headings:{action.args.get('query', '')}")
            if (
                action.tool in ("detect_tables", "extract_table", "reconstruct_table")
                and obs.status == "ok"
            ):
                self.tables_inspected.add(action.args.get("doc_id", ""))
            if self.flags.use_memory:
                self.working.add_observation(
                    action.tool, obs.status, (obs.message or str(obs.data)[:120])
                )
            self.trace.append(
                events.TOOL,
                f"{action.tool} -> {obs.status} ({action.reason[:80]})",
                {
                    "tool": action.tool,
                    "status": obs.status,
                    "action_id": action.action_id,
                    "latency_ms": round(obs.latency_ms, 1),
                },
            )
            if obs.status != "ok":
                continue
            if action.tool == "vision_verify_claim":
                if self._adopt_vision_verdict(action, obs):
                    new_evidence = True
                continue  # verdict handled explicitly; rationale is not mined
            items = self._observation_items(obs)
            if not items:
                continue
            candidates, stats = self.gateway.extract_candidates(
                items, needs, exclusions
            )
            if stats.get("skipped_excluded"):
                self.trace.append(
                    events.EVIDENCE,
                    f"excluded {stats['skipped_excluded']} line(s) by rule "
                    f"{exclusions}",
                )
            for cand in candidates[:MAX_MINED_ITEMS_PER_OBS]:
                if self._adopt(cand, goal):
                    new_evidence = True
            if self._mine_with_model(items, goal):
                new_evidence = True
            self._maybe_specialize(action, obs, goal)
        return new_evidence

    def _mine_with_model(
        self, items: List[Dict[str, Any]], goal: InvestigationGoal
    ) -> bool:
        """Model-driven mining (heuristics above stay as candidate generators)."""
        if (
            not self.flags.use_model_reasoning
            or not self.gateway.model_reasoning_available()
        ):
            return False
        if not self.budgets.usage.consume("model"):
            return False
        try:
            found = self.gateway.extract_structured(items, goal.spec or {})
        except Exception as exc:
            self.trace.append(events.EVIDENCE, f"model mining failed: {exc}")
            return False
        adopted = False
        prov_by_snippet: Dict[str, Dict[str, Any]] = {}
        for item in items:
            item_prov = item.get("provenance", {})
            for raw_line in str(item.get("snippet", "")).splitlines():
                line = raw_line.strip()[:600]
                if line and line not in prov_by_snippet:
                    prov_by_snippet[line] = item_prov
        for entry in found:
            snippet = str(entry.get("snippet", ""))[:600]
            if is_excluded(snippet, self._exclusions(goal)):
                continue
            prov = dict(prov_by_snippet.get(snippet) or {})
            if not prov:
                # No source item carries this snippet: attributing it to an
                # arbitrary document would fabricate provenance. Skip it.
                self.trace.append(
                    events.EVIDENCE,
                    "model candidate skipped: snippet matches no observation",
                )
                continue
            cand = SimpleNamespace(
                subject=str(entry.get("subject", "")).strip() or "unknown",
                attribute=str(entry.get("attribute", "")).strip() or "fact",
                value=entry.get("value"),
                units=entry.get("units"),
                snippet=str(entry.get("snippet", snippet))[:600],
                provenance=prov,
                qualifiers=dict(entry.get("qualifiers", {})),
            )
            if cand.value is None or not cand.subject:
                continue
            if self._adopt(cand, goal):
                adopted = True
        if adopted:
            self.trace.append(events.EVIDENCE, "model mining adopted new candidates")
        return adopted

    def _adopt_vision_verdict(self, action: Action, obs: Observation) -> bool:
        """Link a GPT-5.4 visual verdict as first-class evidence on its claim."""
        claim_id = action.args.get("claim_id") or obs.data.get("claim_id")
        claim = self.store.get_claim(claim_id) if claim_id else None
        if claim is None:
            # Fall back to matching subject/attribute.
            matches = [
                c
                for c in self.store.all_claims()
                if c.subject == action.args.get("subject")
                and c.attribute == action.args.get("attribute")
            ]
            claim = matches[0] if matches else None
        if claim is None:
            self.trace.append(
                events.EVIDENCE, "vision verdict arrived with no matching claim"
            )
            return False
        bbox = action.args.get("bbox")
        region = SourceRegion(
            document_id=action.args.get("doc_id", ""),
            page=action.args.get("page"),
            bbox=tuple(float(v) for v in bbox) if bbox else None,
            source_type="region" if bbox else "page",
            extraction_method=ExtractionMethod.GPT54_VISION,
            snippet=str(obs.data.get("rationale", ""))[:300],
        )
        evidence = self.store.add_evidence(
            region=region,
            snippet=str(obs.data.get("rationale", ""))[:600],
            method=ExtractionMethod.GPT54_VISION,
            confidence=float(obs.data.get("confidence") or 0.8),
        )
        self.graph.add_evidence(evidence)
        self.graph.support(evidence.evidence_id, claim.claim_id)
        self.store.add_link(claim.claim_id, evidence.evidence_id)
        verdict = obs.data.get("supported")
        if verdict is True:
            self.store.mark_verified(claim.claim_id)
        elif verdict is False:
            self.store.mark_uncertain(
                claim.claim_id,
                f"GPT-5.4 Vision disagrees: {obs.data.get('rationale', '')[:200]}",
            )
        self.trace.append(
            events.CLAIM, f"vision verdict on {claim.claim_id}: supported={verdict}"
        )
        return True

    def _observation_items(self, obs: Observation) -> List[Dict[str, Any]]:
        data = obs.data or {}
        # Perception tools stamp their method; mined evidence inherits it.
        method_hint = data.get("extraction_method")
        items: List[Dict[str, Any]] = []
        if "hits" in data and isinstance(data["hits"], list):
            for hit in data["hits"][:20]:
                items.append(
                    {
                        "snippet": str(hit.get("snippet", "")),
                        "provenance": dict(hit.get("provenance", {})),
                    }
                )
        if (
            "answer" in data
            and isinstance(data["answer"], str)
            and data["answer"].strip()
        ):
            # GPT-5.4 Vision reading: emits CLAIM: lines plus prose.
            prov = (obs.provenance or [{}])[0]
            items.append({"snippet": data["answer"][:4000], "provenance": dict(prov)})
        if "text" in data and isinstance(data["text"], str) and data["text"].strip():
            prov = (obs.provenance or [{}])[0]
            if method_hint and not prov.get("method"):
                prov = dict(prov, method=method_hint)
            items.append({"snippet": data["text"][:4000], "provenance": dict(prov)})
        if "pages" in data and isinstance(data["pages"], list):
            for page in data["pages"][:6]:
                items.append(
                    {
                        "snippet": str(page.get("text", ""))[:2000],
                        "provenance": {
                            "document_id": data.get("doc_id"),
                            "page": page.get("page"),
                            "source_type": "page",
                            "method": method_hint or "native_text",
                        },
                    }
                )
        if "rows" in data and isinstance(data["rows"], list):
            rows = data["rows"]
            header = [str(c) for c in (rows[0] if rows else [])]
            for row in rows[1:11]:
                line = " | ".join(header) + "  //  " + " | ".join(str(c) for c in row)
                items.append(
                    {
                        "snippet": line,
                        "provenance": {
                            "document_id": data.get("doc_id"),
                            "page": data.get("page"),
                            "cell": data.get("sheet"),
                            "source_type": "table",
                            "method": "spreadsheet"
                            if data.get("sheet")
                            else "native_table",
                        },
                    }
                )
        if "values" in data and isinstance(data["values"], list):
            values = data["values"]
            if values:
                from ..tables.logical import apply_banner_labels, detect_header_layout

                grid = [[str(c) for c in row] for row in values]
                banner, header_at = detect_header_layout(grid)
                header = grid[header_at] if header_at < len(grid) else grid[0]
                if banner:
                    header = [
                        " > ".join(p) for p in apply_banner_labels([header], banner)
                    ]
                for row in grid[header_at + 1 : header_at + 11]:
                    if not any(str(c).strip() for c in row):
                        continue
                    line = (
                        " | ".join(header) + "  //  " + " | ".join(str(c) for c in row)
                    )
                    items.append(
                        {
                            "snippet": line,
                            "provenance": {
                                "document_id": data.get("doc_id"),
                                "page": None,
                                "cell": f"{data.get('sheet')}!{data.get('range')}",
                                "source_type": "sheet_cell",
                                "method": "spreadsheet",
                            },
                        }
                    )
        if "di_pages" in data and isinstance(data["di_pages"], list):
            for page in data["di_pages"][:6]:
                text = str(page.get("text", ""))
                if text.strip():
                    items.append(
                        {
                            "snippet": text[:2000],
                            "provenance": {
                                "document_id": data.get("doc_id"),
                                "page": page.get("page"),
                                "source_type": "text",
                                "method": ExtractionMethod.AZURE_DI,
                            },
                        }
                    )
        if "di_tables" in data and isinstance(data["di_tables"], list):
            for table in data["di_tables"][:4]:
                header = [str(c) for c in table.get("header", [])]
                for row in table.get("rows", [])[1:11]:
                    line = (
                        " | ".join(header) + "  //  " + " | ".join(str(c) for c in row)
                    )
                    items.append(
                        {
                            "snippet": line,
                            "provenance": {
                                "document_id": data.get("doc_id"),
                                "page": table.get("page"),
                                "bbox": table.get("bbox"),
                                "source_type": "table",
                                "method": ExtractionMethod.AZURE_DI,
                            },
                        }
                    )
        if "logical_tables" in data and isinstance(data["logical_tables"], list):
            for table in data["logical_tables"][:4]:
                paths = [" > ".join(p) for p in table.get("header_paths", [])]
                for row in table.get("rows", [])[:12]:
                    values = [str(v) for v in row.get("values", [])]
                    line = " | ".join(paths) + "  //  " + " | ".join(values)
                    items.append(
                        {
                            "snippet": line,
                            "provenance": {
                                "document_id": data.get("doc_id"),
                                "page": row.get("page", data.get("page")),
                                "cell": row.get("sheet", data.get("sheet")),
                                "source_type": "table",
                                "method": data.get("extraction_method", "native_table"),
                            },
                        }
                    )
        return [it for it in items if it["snippet"].strip()]

    def _adopt(self, cand, goal: InvestigationGoal) -> bool:
        prov = cand.provenance or {}
        doc_id = prov.get("document_id", "")
        page = prov.get("page")
        cell = prov.get("cell")
        existing = self.store.find_duplicate(
            cand.subject, cand.attribute, cand.value, doc_id, page, cell
        )
        region = SourceRegion(
            document_id=doc_id,
            page=page,
            bbox=None,
            source_type=prov.get("source_type", "text") or "text",
            extraction_method=prov.get("method", "native_text") or "native_text",
            snippet=cand.snippet[:300],
            cell=cell,
        )
        evidence = self.store.add_evidence(
            region=region,
            snippet=cand.snippet[:600],
            method=region.extraction_method,
            confidence=0.75,
        )
        self.graph.add_evidence(evidence)
        if existing is not None:
            self.store.add_link(existing.claim_id, evidence.evidence_id)
            self.graph.corroborate(evidence.evidence_id, existing.evidence_ids[0])
            self.graph.support(evidence.evidence_id, existing.claim_id)
            self.trace.append(
                events.EVIDENCE,
                f"corroborated {existing.claim_id} "
                f"({existing.subject}/{existing.attribute})",
            )
            return False
        claim = self.store.create_claim(
            subject=cand.subject,
            attribute=cand.attribute,
            value=cand.value,
            evidence_ids=[evidence.evidence_id],
            units=cand.units,
            qualifiers=dict(cand.qualifiers),
        )
        self._link_field(claim, goal)
        self.graph.add_claim(claim)
        self.graph.support(evidence.evidence_id, claim.claim_id)
        self.evidence_memory.index_claim(claim)
        self.metrics.claims_created += 1
        self.trace.append(
            events.CLAIM,
            f"{claim.claim_id}: {claim.subject}/{claim.attribute}={claim.value!r}",
            {"claim_id": claim.claim_id},
        )
        return True

    def _link_field(self, claim: Claim, goal: InvestigationGoal) -> None:
        """Promote observations to answer claims by linking requested fields.

        Collection needs link every adopted claim (discovery results feed
        saturation; the saturation rule — not linkage — decides COMPLETE).
        Single needs link on token overlap with their field terms.
        Unlinked claims stay observations (§4).
        """
        from ..workspace.indexing import tokenize

        claim_tokens = set(tokenize(f"{claim.subject} {claim.attribute}"))
        best, best_overlap = None, 0
        for need in goal.needs:
            field_ref = getattr(need, "field_ref", None) or {}
            if not field_ref:
                continue
            if field_ref.get("kind") == "collection":
                best, best_overlap = need, 2
                break
            terms = set()
            for text in [field_ref.get("sheet", "")] + list(
                field_ref.get("columns", [])
            ):
                terms.update(tokenize(str(text)))
            overlap = len(claim_tokens & terms)
            if overlap > best_overlap:
                best, best_overlap = need, overlap
        if best is not None and (
            best_overlap >= 2 or (best.field_ref or {}).get("kind") == "collection"
        ):
            claim.tier = "answer"
            claim.field_ref = {
                "sheet": (best.field_ref or {}).get("sheet", ""),
                "need_id": best.need_id,
            }
            self.trace.append(
                events.CLAIM,
                f"{claim.claim_id} answers field "
                f"{claim.field_ref['sheet']}:{best.need_id}",
            )

    def _maybe_specialize(
        self, action: Action, obs: Observation, goal: InvestigationGoal
    ) -> None:
        if not self.flags.use_specialists:
            return
        if (
            action.tool == "detect_tables"
            and obs.status == "ok"
            and obs.data.get("count", 0) > 0
        ):
            doc_id = action.args.get("doc_id", "")
            if doc_id and doc_id not in self.specialist_docs:
                self.specialist_docs.add(doc_id)
                detail = table_specialist.investigate_table(self.computer, doc_id)
                self.trace.append(
                    events.SPECIALIST,
                    f"table specialist on {doc_id}: {detail.get('n_rows', 0)} rows",
                    {"doc_id": doc_id},
                )
                rows = detail.get("rows_sample", [])
                logical = detail.get("logical_tables", [])
                if logical:
                    items = []
                    for table in logical[:2]:
                        paths = [" > ".join(p) for p in table.get("header_paths", [])]
                        for row in table.get("rows", [])[:10]:
                            items.append(
                                {
                                    "snippet": " | ".join(paths)
                                    + "  //  "
                                    + " | ".join(str(v) for v in row.get("values", [])),
                                    "provenance": {
                                        "document_id": doc_id,
                                        "page": row.get("page", detail.get("page")),
                                        "cell": row.get("sheet", detail.get("sheet")),
                                        "source_type": "table",
                                        "method": "native_table",
                                    },
                                }
                            )
                elif rows:
                    header = [str(c) for c in rows[0]]
                    items = [
                        {
                            "snippet": " | ".join(header)
                            + "  //  "
                            + " | ".join(str(c) for c in row),
                            "provenance": {
                                "document_id": doc_id,
                                "page": detail.get("page"),
                                "cell": detail.get("sheet"),
                                "source_type": "table",
                                "method": "native_table",
                            },
                        }
                        for row in rows[1:11]
                    ]
                else:
                    items = []
                if items:
                    candidates, _ = self.gateway.extract_candidates(
                        items, goal.needs, self._exclusions(goal)
                    )
                    for cand in candidates:
                        self._adopt(cand, goal)
        if (
            action.tool in ("search_text", "hybrid_search")
            and len(self.world.refs) > 1
            and self.metrics.tool_calls <= 6
        ):
            detail = research_specialist.cross_document_research(
                self.computer, str(action.args.get("query", ""))
            )
            self.trace.append(
                events.SPECIALIST, f"research specialist: {detail.get('documents', [])}"
            )

    # -- verify / contradictions / finish -------------------------------------------
    def _update_states(self, goal: InvestigationGoal, selected: List[Action]) -> None:
        """Need completion contracts (§5): UNSEEN → … → COMPLETE/NOT_FOUND.

        A single claim means PARTIAL for collection needs — never COMPLETE.
        Saturation needs several distinct evidence regions plus exhausted
        candidates; absence after a full sweep means NOT_FOUND.
        """
        proposed = {need_id for action in selected for need_id in action.need_ids}
        ledger = getattr(self.planner, "candidate_ledger", {})
        for need in goal.needs:
            sat = need.saturation
            sat.setdefault("quiet_rounds", 0)
            sat.setdefault("last_claims", 0)
            aligned = [
                c
                for c in self.store.all_claims()
                if (c.field_ref or {}).get("need_id") == need.need_id
                or (
                    not getattr(need, "field_ref", None)
                    and self._loosely_aligned(need, c)
                )
            ]
            verified = [c for c in aligned if c.is_verified()]
            regions = set()
            for claim in aligned:
                for evidence_id in claim.evidence_ids:
                    evidence = self.store.get_evidence(evidence_id)
                    if evidence is not None:
                        region = evidence.region
                        regions.add((region.document_id, region.page, region.cell))
            cands = ledger.get(need.need_id, [])
            unvisited = [c for c in cands if not self._candidate_visited(c)]
            collection = bool(getattr(need, "field_ref", None)) or (
                need.subject == "*"
                and need.attribute.strip().lower()
                in {
                    "count",
                    "counts",
                    "fact",
                    "facts",
                    "value",
                    "values",
                    "number",
                    "numbers",
                    "data",
                    "info",
                    "requirement",
                    "requirements",
                    "table",
                    "io_count",
                }
            )
            if need.state == "UNSEEN" and (need.need_id in proposed or cands):
                need.state = "DISCOVERING"
            if aligned and need.state in ("UNSEEN", "DISCOVERING"):
                need.state = "PARTIAL"
            if verified and need.state in ("PARTIAL", "DISCOVERING", "UNSEEN"):
                need.state = "VERIFIED_PARTIAL"
            if len(aligned) > sat["last_claims"]:
                sat["quiet_rounds"] = 0
            else:
                sat["quiet_rounds"] += 1
            sat["last_claims"] = len(aligned)
            if collection:
                if len(regions) >= 3 and not unvisited:
                    need.state = "COMPLETE"
                elif not unvisited and cands and sat["quiet_rounds"] >= 1 and aligned:
                    need.state = "COMPLETE"
                elif (
                    not unvisited and cands and sat["quiet_rounds"] >= 1 and not aligned
                ):
                    need.state = "NOT_FOUND"
            else:
                if verified:
                    need.state = "COMPLETE"
                elif (
                    not unvisited and cands and sat["quiet_rounds"] >= 1 and not aligned
                ):
                    need.state = "NOT_FOUND"

    @staticmethod
    def _loosely_aligned(need, claim) -> bool:
        from ..workspace.indexing import tokenize

        need_tokens = set(tokenize(f"{need.subject} {need.attribute}"))
        claim_tokens = set(tokenize(f"{claim.subject} {claim.attribute}"))
        return len(need_tokens & claim_tokens) >= 1 and len(claim_tokens) > 0

    def _candidate_visited(self, cand: Dict[str, Any]) -> bool:
        if cand.get("cell") and cand.get("page") is None:
            sheet = str(cand["cell"]).split("!")[0]
            return (cand.get("doc_id", ""), sheet) in self.visited_sheets
        if cand.get("page") is not None:
            return (cand.get("doc_id", ""), cand["page"]) in self.visited_pages
        return False

    def _verify_round(self, goal: InvestigationGoal) -> None:
        gateway = (
            self.gateway
            if self.flags.use_model_reasoning
            and self.gateway.model_reasoning_available()
            else None
        )
        summary = verifier_specialist.verify_batch(
            self.store, min_overlap=self.config.entailment_min_overlap, gateway=gateway
        )
        if summary["checked"]:
            self.trace.append(
                events.CLAIM,
                f"verifier: {summary['verified']}/{summary['checked']} supported",
            )

    def _contradiction_round(self) -> None:
        if not self.flags.use_graph:
            return  # ablation F-: no live contradiction detection
        found = detect_contradictions(self.store)
        known = {c.contradiction_id for c in self.contradictions}
        for contra in found:
            if contra.contradiction_id in known:
                continue
            self.contradictions.append(contra)
            self.metrics.contradictions_found += 1
            for claim_id in contra.claim_ids:
                claim = self.store.get_claim(claim_id)
                if (
                    claim is not None
                    and claim.verification != VerificationState.UNRESOLVED
                ):
                    self.store.mark_contradicted(
                        claim_id, f"conflicts: values {contra.values}"
                    )
            self.trace.append(
                events.CLAIM,
                f"contradiction {contra.contradiction_id}: "
                f"{contra.subject}/{contra.attribute}={contra.values}",
            )
            self._authority_round(contra)

    def _authority_round(self, contra) -> None:
        """Generic precedence reasoning: resolve when signals agree, else wait."""
        from ..evidence.authority import propose_authority
        from ..evidence.contradictions import resolve_contradiction

        filenames = {d.doc_id: d.filename for d in self.world.refs.values()}
        try:
            proposal = propose_authority(contra, self.store, filenames)
        except Exception as exc:
            self.trace.append(events.CLAIM, f"authority analysis failed: {exc}")
            return
        winner = proposal.get("winning_claim_id")
        if not winner:
            self.trace.append(
                events.CLAIM,
                f"authority uncertain for {contra.contradiction_id}: "
                f"{proposal.get('rationale', '')}",
            )
            return
        resolve_contradiction(contra, winner, proposal.get("rationale", ""))
        for claim_id in contra.claim_ids:
            if claim_id == winner:
                continue
            claim = self.store.get_claim(claim_id)
            if claim is not None and claim.verification != VerificationState.UNRESOLVED:
                self.store.mark_unresolved(
                    claim_id, f"superseded: {proposal.get('rationale', '')}"
                )
        champion = self.store.get_claim(winner)
        if (
            champion is not None
            and champion.verification == VerificationState.CONTRADICTED
        ):
            # Its evidence still entails; the contradiction that tainted it resolved.
            self.store.mark_verified(winner)
        self.trace.append(
            events.CLAIM,
            f"authority resolved {contra.contradiction_id}: {winner} wins "
            f"({proposal.get('rationale', '')})",
        )

    def _evaluate(self, goal: InvestigationGoal) -> VerificationReport:
        ctx = VerificationContext(
            goal=goal,
            claims=self.store.all_claims(),
            evidence=dict(self.store.evidence),
            contradictions=list(self.contradictions),
            min_overlap=self.config.entailment_min_overlap,
            min_verified_ratio=self.config.sufficiency_min_verified_ratio,
            domain_validators=list(self.domain_validators),
            world=self.world,
            visited_pages=set(self.visited_pages),
            visited_sheets=set(self.visited_sheets),
        )
        return self.harness.evaluate(ctx)

    def _finalize_unresolved(self, goal: InvestigationGoal) -> None:
        for claim in self.store.all_claims():
            if claim.verification == VerificationState.UNVERIFIED:
                self.store.mark_unresolved(
                    claim.claim_id, "insufficient corroboration within budget"
                )
        budget_dead = not self.budgets.ok()
        for need in goal.needs:
            if not need_addressed(need, self.store.all_claims()) and need.required:
                marker = self.store.create_claim(
                    need.subject if need.subject != "*" else "unknown",
                    need.attribute,
                    None,
                    evidence_ids=[],
                )
                self.store.mark_unresolved(
                    marker.claim_id,
                    f"no {need.subject}/{need.attribute} found in corpus",
                )
                self.graph.add_claim(marker)
                if need.state not in ("COMPLETE",):
                    need.state = "BLOCKED" if budget_dead else "NOT_FOUND"

    def _force_honest_finish(self, goal: InvestigationGoal) -> None:
        for contra in self.contradictions:
            if contra.status == "open":
                mark_unresolvable(
                    contra, "conflicting sources; no authority established"
                )
                for claim_id in contra.claim_ids:
                    claim = self.store.get_claim(claim_id)
                    if (
                        claim is not None
                        and claim.verification != VerificationState.VERIFIED
                    ):
                        self.store.mark_unresolved(
                            claim_id,
                            f"part of unresolved contradiction "
                            f"{contra.contradiction_id}: values {contra.values}",
                        )
        for task in self.report.recovery if self.report else []:
            if not task.outcome:
                task.outcome = "unresolved: verification still failing at budget end"
                task.cost = self.metrics.tool_calls - self._recovery_tool_mark
        self._finalize_unresolved(goal)

    def _corpus_signature(self) -> str:
        kinds = sorted({p.classification for p in self.world.profiles.values()})
        return f"corpus:{'+'.join(kinds)}:docs={len(self.world.profiles)}"

    def _tools_used(self) -> List[str]:
        return sorted(
            {
                d["tool"]
                for d in self.working.recent_observations
                if isinstance(d, dict) and "tool" in d
            }
        )
