"""Model gateway: one interface, swappable backends. Secrets from env only."""

from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List

from ..core.config import RuntimeConfig
from ..core.schemas import InvestigationGoal
from . import heuristic
from .schemas import ModelRequest, ModelResponse, ModelUnavailable

from ..core.schemas import InformationNeed  # noqa: E402  (re-export for callers)


THINK_PROMPT = """\
You direct an autonomous document-investigation agent. Given the state summary,
propose the highest-value next tool calls as a JSON object:
{"actions": [{"tool": "<name>", "args": {}, "rationale": "<one line>"}],
 "focus": "<what uncertainty this resolves>", "stop_hint": "<done|continue>"}
Use ONLY these tools: list_documents, inspect_document, search_text,
semantic_search, hybrid_search, search_headings, read_text, read_page,
read_section, read_surrounding_context, detect_tables, extract_table,
read_spreadsheet, read_sheet_range, find_supporting_evidence,
find_contradicting_evidence, verify_claim, calculate, di_analyze_document,
render_page, inspect_page_visually, inspect_region_visually,
vision_verify_claim.
Prefer cheap reads over vision. Never repeat a listed prior action. Reply with
the JSON object only.
State:
"""

MINE_PROMPT = """\
Extract factual claims from document snippets for the given spec. Reply with a
JSON object: {"candidates": [{"subject": "...", "attribute": "...",
"value": <number|string|null>, "units": null|string, "snippet": "<source line>",
"qualifiers": {}}]}. Rules: subject MUST name the specific entity or record
the fact is about (e.g. "10-P-101A", "Unit Boiler"), never a generic category
like "equipment" or "system" — distinct entities must never share a subject.
Use null when the value is absent. Never invent values. Copy snippets verbatim
from the input. Skip excluded content named in the spec.
Input:
"""

VERIFY_PROMPT = """\
Judge whether the cited snippets support the claimed value. Reply with a JSON
object: {"verdict": "SUPPORTED|CONTRADICTED|AMBIGUOUS", "rationale": "<short>",
"confidence": 0.0-1.0}. SUPPORTED only if a snippet visibly states the value;
CONTRADICTED if snippets state a different value; else AMBIGUOUS.
Input:
"""


class ModelGateway:
    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config

    @property
    def backend(self) -> str:
        return self.config.model.backend

    # -- goal + mining (heuristic-first; model refines only if configured) ----
    def parse_goal(self, prompt: str) -> InvestigationGoal:
        return heuristic.parse_goal(prompt)

    def extract_candidates(
        self,
        items: List[Dict[str, Any]],
        needs: List[InformationNeed],
        exclusions: List[str],
    ):
        return heuristic.extract_candidates(items, needs, exclusions)

    def model_reasoning_available(self) -> bool:
        """True when open-ended completions can actually run."""
        if self.backend not in ("openai-compatible", "openai"):
            return False
        model = self.config.model
        return bool(model.api_key and model.base_url and model.chat_deployment)

    # -- model-driven reasoning (raise ModelUnavailable when heuristic) ---------
    def propose_actions(self, state_summary: str) -> Dict[str, Any]:
        """LLM next-action proposal over a compact state summary (JSON)."""
        from .json_utils import parse_json_response

        response = self.complete(
            ModelRequest(
                kind="chat",
                max_tokens=1500,
                response_format="json_object",
                prompt=THINK_PROMPT + state_summary,
            )
        )
        payload, _ = parse_json_response(response.text)
        if not isinstance(payload, dict) or not isinstance(
            payload.get("actions"), list
        ):
            raise ModelUnavailable("model returned no usable action list")
        return {
            "actions": payload["actions"][:8],
            "focus": str(payload.get("focus", "")),
            "stop_hint": str(payload.get("stop_hint", "")),
        }

    def extract_structured(
        self, items: List[Dict[str, Any]], spec_dict: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """LLM claim mining over observation snippets (JSON candidates)."""
        from .json_utils import parse_json_response

        bundle = json.dumps(
            {
                "spec": spec_dict,
                "snippets": [
                    {
                        "snippet": str(i.get("snippet", ""))[:1200],
                        "provenance": i.get("provenance", {}),
                    }
                    for i in items[:12]
                ],
            }
        )
        response = self.complete(
            ModelRequest(
                kind="extract",
                max_tokens=2000,
                response_format="json_object",
                prompt=MINE_PROMPT + bundle,
            )
        )
        payload, _ = parse_json_response(response.text)
        candidates = payload.get("candidates", []) if isinstance(payload, dict) else []
        return [c for c in candidates if isinstance(c, dict)][:40]

    def verify_claim_structured(
        self, value: Any, snippets: List[str], context: str = ""
    ) -> Dict[str, Any]:
        """Independent model verdict: SUPPORTED | CONTRADICTED | AMBIGUOUS."""
        from .json_utils import parse_json_response

        bundle = json.dumps(
            {
                "value": value,
                "context": context[:500],
                "snippets": [str(s)[:1200] for s in snippets[:4]],
            }
        )
        response = self.complete(
            ModelRequest(
                kind="entail",
                max_tokens=600,
                response_format="json_object",
                prompt=VERIFY_PROMPT + bundle,
            )
        )
        payload, _ = parse_json_response(response.text)
        if not isinstance(payload, dict):
            raise ModelUnavailable("model returned no verdict")
        verdict = str(payload.get("verdict", "AMBIGUOUS")).upper()
        if verdict not in ("SUPPORTED", "CONTRADICTED", "AMBIGUOUS"):
            verdict = "AMBIGUOUS"
        return {
            "verdict": verdict,
            "rationale": str(payload.get("rationale", ""))[:500],
            "confidence": payload.get("confidence"),
        }

    def expand_queries(self, terms: List[str]) -> List[str]:
        """Model-generated retrieval query variants; heuristic fallback.

        Never returns document-specific content — only query strings built
        from the given terms.
        """
        if not self.model_reasoning_available():
            return _heuristic_variants(terms)
        try:
            from .json_utils import parse_json_response

            response = self.complete(
                ModelRequest(
                    kind="chat",
                    max_tokens=400,
                    response_format="json_object",
                    prompt=(
                        "Propose 6 diverse document-search queries covering these "
                        'terms. Reply {"queries": [...]} using only the given '
                        "words and generic synonyms.\nTerms: " + ", ".join(terms[:12])
                    ),
                )
            )
            payload, _ = parse_json_response(response.text)
            queries = [
                str(q)
                for q in payload.get("queries", [])
                if isinstance(q, str) and q.strip()
            ]
            if queries:
                return queries[:8]
        except Exception:
            pass
        return _heuristic_variants(terms)

    def judge_field(self, field: str, expected: Any, prediction: Any) -> Dict[str, Any]:
        """Independent semantic judge: field + GT + prediction ONLY.

        Returns CORRECT | PARTIALLY_CORRECT | INCORRECT | MISSING + reason.
        No traces, no system identity, no extra context.
        """
        from .json_utils import parse_json_response

        if prediction is None or (
            isinstance(prediction, str) and not prediction.strip()
        ):
            return {"verdict": "MISSING", "reason": "no prediction supplied"}
        response = self.complete(
            ModelRequest(
                kind="chat",
                max_tokens=300,
                response_format="json_object",
                prompt=(
                    "Judge whether the prediction satisfies the expected answer "
                    "for the field. Reply in JSON with an object shaped like "
                    '{"verdict": "CORRECT|PARTIALLY_CORRECT|'
                    'INCORRECT|MISSING", "reason": "<concise>"}. Be strict: '
                    "partial credit only when the prediction contains the material "
                    "requirement with minor wording differences.\n"
                    f"Field: {field}\nExpected: {expected}\nPrediction: {prediction}"
                ),
            )
        )
        payload, _ = parse_json_response(response.text)
        verdict = str(payload.get("verdict", "INCORRECT")).upper()
        if verdict not in ("CORRECT", "PARTIALLY_CORRECT", "INCORRECT", "MISSING"):
            verdict = "INCORRECT"
        return {"verdict": verdict, "reason": str(payload.get("reason", ""))[:300]}

    # -- raw completions (optional backends) ------------------------------------
    def complete(self, request: ModelRequest) -> ModelResponse:
        if self.backend == "heuristic":
            raise ModelUnavailable(
                "heuristic backend has no open-ended completions; "
                "use parse_goal/extract_candidates or configure ADH_MODEL_BACKEND"
            )
        if self.backend in ("openai-compatible", "openai"):
            return self._openai_compatible(request)
        raise ModelUnavailable(f"unknown model backend '{self.backend}'")

    def _openai_compatible(self, request: ModelRequest) -> ModelResponse:
        model = self.config.model
        if not model.api_key or not model.base_url or not model.chat_deployment:
            raise ModelUnavailable(
                "model backend needs ADH_MODEL_API_KEY, ADH_MODEL_BASE_URL "
                "and ADH_MODEL_DEPLOYMENT"
            )
        if self.backend == "openai":
            url = model.base_url.rstrip("/") + "/chat/completions"
        else:
            url = (
                model.base_url.rstrip("/")
                + "/openai/deployments/"
                + model.chat_deployment
                + "/chat/completions?api-version="
                + (model.api_version or "2024-10-21")
            )
        payload: Dict[str, Any] = {"max_completion_tokens": request.max_tokens}
        if self.backend == "openai":
            payload["model"] = model.chat_deployment
        if request.images:
            parts: List[Dict[str, Any]] = [{"type": "text", "text": request.prompt}]
            parts.extend(
                {"type": "image_url", "image_url": {"url": url}}
                for url in request.images[:4]
            )
            payload["messages"] = [{"role": "user", "content": parts}]
        else:
            payload["messages"] = [{"role": "user", "content": request.prompt}]
        if request.response_format == "json_object":
            payload["response_format"] = {"type": "json_object"}
        started = time.perf_counter()
        from ..deep.usage import model_request

        with model_request():
            body = self._post_chat(
                url, model.api_key, payload, bearer_auth=self.backend == "openai"
            )
            return self._finish_openai_response(body, started)

    @staticmethod
    def _post_chat(
        url: str, api_key: str, payload: Dict[str, Any], *, bearer_auth: bool = False
    ) -> Dict[str, Any]:
        """POST with max_completion_tokens (GPT-5.x), retrying legacy max_tokens."""
        import urllib.error

        def _send(body: Dict[str, Any]) -> Dict[str, Any]:
            headers = {"Content-Type": "application/json"}
            if bearer_auth:
                headers["Authorization"] = f"Bearer {api_key}"
            else:
                headers["api-key"] = api_key
            req = urllib.request.Request(
                url,
                data=json.dumps(body).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=90) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:300]
                raise ModelUnavailable(
                    f"model call failed: HTTPError {exc.code}: {detail}"
                )

        try:
            return _send(payload)
        except ModelUnavailable as exc:
            message = str(exc)
            if (
                "max_completion_tokens" in message
                and "max_completion_tokens" in payload
            ):
                legacy = dict(payload)
                legacy["max_tokens"] = legacy.pop("max_completion_tokens")
                return _send(legacy)
            raise
        except Exception as exc:
            raise ModelUnavailable(f"model call failed: {type(exc).__name__}: {exc}")

    def _finish_openai_response(
        self, body: Dict[str, Any], started: float
    ) -> ModelResponse:
        from ..deep.usage import record_usage

        usage = body.get("usage", {})
        record_usage(usage)
        try:
            text = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise ModelUnavailable(f"malformed model response: {str(body)[:200]}")
        return ModelResponse(
            text=text,
            backend=self.backend,
            tokens_in=int(usage.get("prompt_tokens", 0)),
            tokens_out=int(usage.get("completion_tokens", 0)),
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )


def _heuristic_variants(terms: List[str]) -> List[str]:
    """Generic term variants: case, splits, singulars, pairs. No domain lists."""
    variants: List[str] = []
    cleaned = [t.strip().lower() for t in terms if t and t.strip()]
    if cleaned:
        variants.append(" ".join(cleaned[:6]))
    for term in cleaned[:8]:
        parts = [p for p in re.split(r"[^a-z0-9]+", term) if p]
        variants.extend(parts)
        if term.endswith("s") and len(term) > 3:
            variants.append(term[:-1])
    seen, out = set(), []
    for variant in variants:
        if variant and variant not in seen:
            seen.add(variant)
            out.append(variant)
    return out[:8]
