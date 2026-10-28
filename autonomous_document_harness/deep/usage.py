"""Token-usage accounting for benchmark runs.

The ExtractBench leaderboard reports cost per page, which needs the token
usage of every model call. The harness counts tokens on one code path only
(the agent loop) and the bulk page-wise extractor counts none at all, so the
usage is collected here instead: a single process-wide, lock-protected
counter that every model response is added to.

The counter is deliberately process-wide rather than per-document: benchmark
documents run concurrently in worker threads, so a per-document delta would
interleave calls from different documents. Per-document latency is already
recorded exactly (`run_metrics.json -> elapsed_seconds`); the total token
usage is what the cost-per-page figure needs.
"""

from __future__ import annotations

import threading
from typing import Any, Optional


class UsageCounter:
    """Thread-safe accumulator of prompt/completion tokens."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.calls = 0

    def add(self, prompt_tokens: int, completion_tokens: int) -> None:
        with self._lock:
            self.prompt_tokens += int(prompt_tokens or 0)
            self.completion_tokens += int(completion_tokens or 0)
            self.calls += 1

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.prompt_tokens + self.completion_tokens,
                "calls": self.calls,
            }

    def reset(self) -> None:
        with self._lock:
            self.prompt_tokens = 0
            self.completion_tokens = 0
            self.calls = 0


COUNTER = UsageCounter()

# Per-thread counters: the agent loop and the JSON converter run on the
# runner's worker thread, so their usage belongs to exactly one document,
# while the bulk page-wise extractor runs its own worker pool and reports its
# usage per window. A process-wide delta cannot separate concurrent
# documents, but a thread-local one can.
_LOCAL = threading.local()


def _local_counter() -> UsageCounter:
    counter = getattr(_LOCAL, "counter", None)
    if counter is None:
        counter = UsageCounter()
        _LOCAL.counter = counter
    return counter


def thread_snapshot() -> dict:
    """Token usage recorded on the calling thread."""
    return _local_counter().snapshot()


def thread_reset() -> None:
    """Forget the calling thread's usage (start of a new document)."""
    _local_counter().reset()


def _tokens_from_usage(usage: Any) -> tuple[int, int]:
    """Extract (prompt, completion) tokens from a LangChain usage payload."""
    if usage is None:
        return 0, 0
    if isinstance(usage, dict):
        prompt = usage.get("input_tokens") or usage.get("prompt_tokens") or 0
        completion = usage.get("output_tokens") or usage.get("completion_tokens") or 0
        return int(prompt), int(completion)
    prompt = getattr(usage, "input_tokens", None)
    if prompt is None:
        prompt = getattr(usage, "prompt_tokens", 0)
    completion = getattr(usage, "output_tokens", None)
    if completion is None:
        completion = getattr(usage, "completion_tokens", 0)
    return int(prompt or 0), int(completion or 0)


def record_response(response: Any) -> None:
    """Add the token usage of one model response to the counters."""
    if response is None:
        return
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        llm_output = getattr(response, "llm_output", None) or {}
        usage = llm_output.get("token_usage") if isinstance(llm_output, dict) else None
    prompt, completion = _tokens_from_usage(usage)
    if prompt or completion:
        COUNTER.add(prompt, completion)
        _local_counter().add(prompt, completion)


def tokens_of(response: Any) -> tuple[int, int]:
    """(prompt, completion) tokens of one model response, without recording."""
    if response is None:
        return 0, 0
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        llm_output = getattr(response, "llm_output", None) or {}
        usage = llm_output.get("token_usage") if isinstance(llm_output, dict) else None
    return _tokens_from_usage(usage)


class TrackedRunnable:
    """Wraps a LangChain runnable and records the usage of every response."""

    def __init__(self, runnable: Any) -> None:
        self._runnable = runnable

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        response = self._runnable.invoke(*args, **kwargs)
        record_response(response)
        return response

    async def ainvoke(self, *args: Any, **kwargs: Any) -> Any:
        response = await self._runnable.ainvoke(*args, **kwargs)
        record_response(response)
        return response

    def stream(self, *args: Any, **kwargs: Any) -> Any:
        last = None
        for chunk in self._runnable.stream(*args, **kwargs):
            last = chunk
            yield chunk
        record_response(last)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._runnable, name)


class TrackedChatModel:
    """Wraps a chat model so every invocation is counted.

    ``bind_tools`` is wrapped too: the harness calls the model through
    ``bind_tools(...)`` and the bound runnable is what actually performs the
    request, so a wrapper that only intercepted ``invoke`` would miss every
    agent turn.
    """

    def __init__(self, model: Any) -> None:
        self._model = model

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        response = self._model.invoke(*args, **kwargs)
        record_response(response)
        return response

    async def ainvoke(self, *args: Any, **kwargs: Any) -> Any:
        response = await self._model.ainvoke(*args, **kwargs)
        record_response(response)
        return response

    def bind_tools(self, *args: Any, **kwargs: Any) -> Any:
        return TrackedRunnable(self._model.bind_tools(*args, **kwargs))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)


def estimate_cost(snapshot: dict) -> Optional[dict]:
    """Cost of a usage snapshot from the configured per-million-token prices.

    Prices come from ``ADH_INPUT_COST_PER_MTOK`` / ``ADH_OUTPUT_COST_PER_MTOK``
    (dollars per million tokens). Without them no cost is reported -- a
    guessed price would be worse than none.
    """
    import os

    def price(name: str) -> Optional[float]:
        raw = os.environ.get(name, "").strip()
        if not raw:
            return None
        try:
            return float(raw)
        except ValueError:
            return None

    p_in = price("ADH_INPUT_COST_PER_MTOK")
    p_out = price("ADH_OUTPUT_COST_PER_MTOK")
    if p_in is None and p_out is None:
        return None
    cost = 0.0
    if p_in is not None:
        cost += snapshot.get("prompt_tokens", 0) / 1_000_000 * p_in
    if p_out is not None:
        cost += snapshot.get("completion_tokens", 0) / 1_000_000 * p_out
    return {
        "cost_usd": round(cost, 6),
        "input_cost_per_mtok": p_in,
        "output_cost_per_mtok": p_out,
    }
