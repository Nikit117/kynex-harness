"""Model and DI usage, with document scopes shared by child workers."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Optional


class UsageCounter:
    """Thread-safe accumulator of prompt/completion tokens."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.calls = 0
        self.cached_prompt_tokens = 0
        self.unpriced_model_calls = 0
        self.pending_model_calls = 0
        self.di_requests = 0
        self.di_pages = 0
        self.di_requests_accepted = 0
        self.di_pages_accepted = 0

    def add(
        self, prompt_tokens: int, completion_tokens: int, cached_prompt_tokens: int = 0
    ) -> None:
        with self._lock:
            self.prompt_tokens += int(prompt_tokens or 0)
            self.completion_tokens += int(completion_tokens or 0)
            self.calls += 1
            self.cached_prompt_tokens += cached_prompt_tokens

    def add_pending(self, delta: int) -> None:
        with self._lock:
            self.pending_model_calls += delta

    def add_unpriced(self) -> None:
        with self._lock:
            self.unpriced_model_calls += 1

    def add_di(self, pages: int, accepted: bool) -> None:
        with self._lock:
            if accepted:
                self.di_requests_accepted += 1
                self.di_pages_accepted += pages
            else:
                self.di_requests += 1
                self.di_pages += pages

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.prompt_tokens + self.completion_tokens,
                "calls": self.calls,
                "cached_prompt_tokens": self.cached_prompt_tokens,
                "unpriced_model_calls": self.unpriced_model_calls,
                "pending_model_calls": self.pending_model_calls,
                "model_usage_complete": self.unpriced_model_calls == 0
                and self.pending_model_calls == 0,
                "di_requests": self.di_requests,
                "di_pages": self.di_pages,
                "di_requests_accepted": self.di_requests_accepted,
                "di_pages_accepted": self.di_pages_accepted,
            }

    def reset(self) -> None:
        with self._lock:
            self.prompt_tokens = 0
            self.completion_tokens = 0
            self.calls = 0
            self.cached_prompt_tokens = 0
            self.unpriced_model_calls = 0
            self.pending_model_calls = 0
            self.di_requests = 0
            self.di_pages = 0
            self.di_requests_accepted = 0
            self.di_pages_accepted = 0


COUNTER = UsageCounter()

# Thread snapshots support per-window deltas; document scopes include all
# workers, including probes whose records do not reach the final prediction.
_LOCAL = threading.local()
_DOCUMENT: ContextVar[UsageCounter | None] = ContextVar("document_usage", default=None)
_REQUEST_RECORDED: ContextVar[list[bool] | None] = ContextVar("request_recorded", default=None)


@contextmanager
def document_usage() -> Iterator[UsageCounter]:
    counter = UsageCounter()
    token = _DOCUMENT.set(counter)
    try:
        yield counter
    finally:
        _DOCUMENT.reset(token)


def document_snapshot() -> dict:
    counter = _DOCUMENT.get()
    if counter is None:
        raise RuntimeError("No document usage scope")
    return counter.snapshot()


def _counters() -> Iterator[UsageCounter]:
    yield COUNTER
    yield _local_counter()
    document = _DOCUMENT.get()
    if document is not None:
        yield document


def record_unpriced_call() -> None:
    """A failed or unmetered invocation may still incur a provider charge."""
    for counter in _counters():
        counter.add_unpriced()


@contextmanager
def model_request() -> Iterator[None]:
    """Keep exported cost incomplete until an invocation's usage is available."""
    counters = tuple(_counters())
    for counter in counters:
        counter.add_pending(1)
    recorded = [False]
    token = _REQUEST_RECORDED.set(recorded)
    try:
        yield
    finally:
        for counter in counters:
            if not recorded[0]:
                counter.add_unpriced()
            counter.add_pending(-1)
        _REQUEST_RECORDED.reset(token)


def record_di_request(pages: int, *, accepted: bool = False) -> None:
    """Track analysis POST attempts and acceptances; exclude polls/cache hits."""
    for counter in _counters():
        counter.add_di(pages, accepted)


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
        record_usage(None)
        return
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        llm_output = getattr(response, "llm_output", None) or {}
        usage = llm_output.get("token_usage") if isinstance(llm_output, dict) else None
    record_usage(usage)


def record_usage(usage: Any) -> None:
    """Record either LangChain or raw Chat Completions usage exactly once."""
    recorded = _REQUEST_RECORDED.get()
    if recorded is not None:
        recorded[0] = True
    input_names = ("input_tokens", "prompt_tokens")
    output_names = ("output_tokens", "completion_tokens")
    if isinstance(usage, dict):
        complete = any(usage.get(k) is not None for k in input_names) and any(
            usage.get(k) is not None for k in output_names
        )
    else:
        complete = any(getattr(usage, k, None) is not None for k in input_names) and any(
            getattr(usage, k, None) is not None for k in output_names
        )
    if not complete:
        record_unpriced_call()
    prompt, completion = _tokens_from_usage(usage)
    if prompt == 0 and completion == 0:
        return
    if isinstance(usage, dict):
        details = usage.get("input_token_details") or usage.get("prompt_tokens_details") or {}
    else:
        details = getattr(usage, "prompt_tokens_details", None) or {}
    if isinstance(details, dict):
        cached = int(details.get("cache_read", details.get("cached_tokens", 0)) or 0)
    else:
        cached = int(getattr(details, "cached_tokens", 0) or 0)
    for counter in _counters():
        counter.add(prompt, completion, cached)


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
        with model_request():
            response = self._runnable.invoke(*args, **kwargs)
            record_response(response)
        return response

    async def ainvoke(self, *args: Any, **kwargs: Any) -> Any:
        with model_request():
            response = await self._runnable.ainvoke(*args, **kwargs)
            record_response(response)
        return response

    def stream(self, *args: Any, **kwargs: Any) -> Any:
        last = None
        with model_request():
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
        with model_request():
            response = self._model.invoke(*args, **kwargs)
            record_response(response)
        return response

    async def ainvoke(self, *args: Any, **kwargs: Any) -> Any:
        with model_request():
            response = await self._model.ainvoke(*args, **kwargs)
            record_response(response)
        return response

    def bind_tools(self, *args: Any, **kwargs: Any) -> Any:
        return TrackedRunnable(self._model.bind_tools(*args, **kwargs))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)


def estimate_cost(snapshot: dict) -> Optional[dict]:
    """LLM cost from configured per-million-token prices (DI priced separately).

    Prices come from ``ADH_INPUT_COST_PER_MTOK`` / ``ADH_OUTPUT_COST_PER_MTOK``
    (dollars per million tokens). Without them no cost is reported -- a
    guessed price would be worse than none. Failed calls retain a known
    subtotal, but the total stays unknown when usage or rates are incomplete.
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
    p_cached = price("ADH_CACHED_INPUT_COST_PER_MTOK")
    if p_in is None and p_out is None and p_cached is None:
        return None
    cost = 0.0
    if p_in is not None:
        cost += snapshot.get("prompt_tokens", 0) / 1_000_000 * p_in
    if p_cached is not None:
        cached = snapshot.get("cached_prompt_tokens", 0)
        cost += cached / 1_000_000 * (p_cached - (p_in if p_in is not None else 0.0))
    if p_out is not None:
        cost += snapshot.get("completion_tokens", 0) / 1_000_000 * p_out
    complete = (
        snapshot.get("unpriced_model_calls", 0) == 0
        and snapshot.get("pending_model_calls", 0) == 0
        and (p_in is not None or snapshot.get("prompt_tokens", 0) == 0)
        and (p_out is not None or snapshot.get("completion_tokens", 0) == 0)
    )
    return {
        "cost_usd": round(cost, 6) if complete else None,
        "known_cost_usd": round(cost, 6),
        "cost_complete": complete,
        "input_cost_per_mtok": p_in,
        "output_cost_per_mtok": p_out,
        "cached_input_cost_per_mtok": p_cached,
    }
