"""Bulk page-wise array extraction (RC-4: massive arrays).

When an ExtractBench schema expects a very large top-level array (tens of
thousands of rows), a single-shot LLM agent answer cannot hold every row and
the item scores ~0 with an empty array. This module attacks that failure mode
directly: it accumulates one giant array by running ONE cheap LLM call per
page window over deterministic PyMuPDF page text, then concatenating the
results.

Design notes
------------
* Page text is extracted deterministically (PyMuPDF). By default the text is
  re-rendered from word bboxes with an INDENT marker on lines that sit right
  of the local column edge — this preserves the "flush name line vs indented
  continuation line" cue that newline-only get_text() destroys.
* The LLM is called once per window (concurrency capped at MAX_WORKERS_CAP =
  2 so a benchmark running in parallel shares the rate limits safely).
* Responses may be huge (tens of thousands of rows -> truncated by
  max_tokens). ``parse_array_response`` salvages the complete prefix of
  elements from a truncated array instead of discarding everything.
* No existing repo file is modified; tested helpers
  (``_extract_json_block``, ``_salvage_truncated_json``) are imported from
  ``run_tui_benchmark_v2`` read-only and used only as object-oriented
  fallbacks.

Usage (CLI):
    python -I autonomous_document_harness/scripts/bulk_array_extract.py \
        --pdf data/long/<item>.pdf --test-json data/long/<item>.test.json \
        --field owners --out <pred_dir>/<item>/prediction.json \
        --max-pages 4 --workers 2 --window 1
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Reuse the tested object-oriented parsers from the (unmodified) v2 benchmark.
from autonomous_document_harness.scripts.run_tui_benchmark_v2 import (  # noqa: E402
    _extract_json_block,
    _salvage_truncated_json,
)

MAX_WORKERS_CAP = 6

FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
PAGE_FIELD_RE = re.compile(r"page", re.IGNORECASE)


# ---------------------------------------------------------------------------
# LLM construction / invocation seams (monkeypatched by unit tests)
# ---------------------------------------------------------------------------


def _llm_timeout_seconds() -> float:
    """Per-request timeout for the batch model (``ADH_LLM_TIMEOUT_SECONDS``).

    1800 s default: generous enough that a legitimate 20k-token page response
    still completes (the largest pages in the corpus hold ~450 records), while a
    stalled request can no longer hang a run forever. ``max_retries=2`` covers a
    transient network failure without silently losing a page.
    """
    raw = os.environ.get("ADH_LLM_TIMEOUT_SECONDS", "").strip()
    if not raw:
        return 1800.0
    try:
        value = float(raw)
    except ValueError:
        return 1800.0
    return value if value > 0 else 1800.0


def _build_model() -> Any:
    """Build the shared Azure chat model (never logs or prints secrets)."""
    from autonomous_document_harness.deep.model import (
        build_tracked_chat_model,
        ensure_azure_env,
    )

    ensure_azure_env()
    return build_tracked_chat_model(
        max_tokens=32768,
        request_timeout=_llm_timeout_seconds(),
        max_retries=2,
    )


def _invoke_llm(model: Any, system: str, user: str) -> str:
    """Invoke the chat model and return its reply as plain text."""
    from langchain_core.messages import HumanMessage, SystemMessage

    response = model.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    return _content_to_text(response.content)


def _content_to_text(content: Any) -> str:
    """Flatten a LangChain message content block into plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(str(block.get("text") or block.get("content") or ""))
        return "\n".join(p for p in parts if p)
    return str(content or "")


# ---------------------------------------------------------------------------
# Deterministic page-text extraction (PyMuPDF)
# ---------------------------------------------------------------------------


def _page_text_with_indent(page: Any) -> str:
    """Re-render one page with an indentation marker per line.

    Lines are grouped by PyMuPDF's native (block, line) ids, binned into
    "columns" by left edge x0 (gap > COLUMN_GAP_PT starts a new column), and
    within a column a line whose x0 sits > INDENT_MIN_PT right of the column
    minimum gets a two-space indent marker. This reconstructs the
    flush-vs-indent distinction (primary record line vs continuation line)
    that plain ``get_text()`` flattens away. Deterministic and doc-agnostic.
    """
    words = page.get_text("words")  # x0, y0, x1, y1, word, block, line, word_no
    if not words:
        return page.get_text()

    # Per native line: key -> (y, x0, tokens sorted by x0)
    y_by: dict[tuple[int, int], float] = {}
    x_by: dict[tuple[int, int], float] = {}
    tok_by: dict[tuple[int, int], list[tuple[float, str]]] = {}
    for w in words:
        key = (w[5], w[6])  # (block_no, line_no)
        if key not in y_by:
            y_by[key] = w[1]
        if key not in x_by or w[0] < x_by[key]:
            x_by[key] = w[0]
        tok_by.setdefault(key, []).append((w[0], w[4]))

    texts: dict[tuple[int, int], str] = {
        k: " ".join(t[1] for t in sorted(v)) for k, v in tok_by.items()
    }

    # Column left edges: cluster line-minimum x0's by gap threshold. Indent
    # (~4.5pt here) stays inside a cluster; a different column does not.
    COLUMN_GAP_PT = 20.0
    edges: list[float] = []
    for x in sorted(x_by.values()):
        if not edges or x - edges[-1] > COLUMN_GAP_PT:
            edges.append(x)

    def assign_col(x: float) -> float:
        return min(edges, key=lambda c: abs(c - x))

    by_col: dict[float, list[tuple[float, str, float]]] = {}
    for key, x0 in x_by.items():
        col = assign_col(x0)
        by_col.setdefault(col, []).append((y_by[key], texts[key], x0))

    INDENT_MIN_PT = 2.0
    chunks: list[str] = []
    for col in sorted(by_col):
        if chunks:
            chunks.append("")
        for _y, text, x0 in sorted(by_col[col], key=lambda t: t[0]):
            indent = "  " if (x0 - col) > INDENT_MIN_PT else ""
            chunks.append(f"{indent}{text}")
    return "\n".join(chunks)


_OCR_BLANK_CHARS = 30


def _di_page_texts(pdf_path: str, pages: list[int]) -> dict[int, str]:
    """Text for pages whose native layer is empty, read by Document Intelligence.

    The page-wise enumerator has its own text path (PyMuPDF), so the staging-time
    OCR fallback of the parser never reached it: on a scanned document the
    enumerator read nothing at all and the long lists stayed empty
    (``real_ftx_full_corrupted``: 521 of 7 554 rows).

    Kill switch: ``ADH_BULK_OCR=0``.
    """
    if os.environ.get("ADH_BULK_OCR", "1").strip().lower() in ("0", "false", "no"):
        return {}
    try:
        from ..deep.model import ensure_azure_env

        # the DI credentials live in the sibling .env, so the environment has
        # to be bridged BEFORE they are read
        ensure_azure_env()
    except Exception:  # noqa: BLE001 -- OCR is best-effort
        pass
    endpoint = os.environ.get("ADH_DI_ENDPOINT", "") or os.environ.get(
        "AZURE_DI_ENDPOINT", ""
    )
    key = os.environ.get("ADH_DI_API_KEY", "") or os.environ.get("AZURE_DI_API_KEY", "")
    if not (endpoint and key):
        return {}
    try:
        from ..tools.azure_di import AzureDIClient

        client = AzureDIClient(endpoint, key, cache={})
    except Exception:  # noqa: BLE001 -- OCR is best-effort
        return {}

    out: dict[int, str] = {}
    for i in range(0, len(pages), 25):
        chunk = pages[i : i + 25]
        try:
            layout = client.analyze_layout(
                str(pdf_path),
                pages=chunk,
                cache_key=f"bulk-ocr:{pdf_path}:{chunk[0]}",
            )
        except Exception:  # noqa: BLE001 -- best effort
            continue
        for page in layout.pages:
            if page.text.strip():
                out[page.page_no] = page.text
    return out


def extract_page_texts(
    pdf_path: str,
    start_page: int = 1,
    end_page: Optional[int] = None,
    *,
    layout: bool = True,
) -> tuple[list[str], int]:
    """Extract deterministic text for a one-indexed inclusive page range.

    Returns (texts, total_pages); texts[k] is the text of page start_page+k.
    Extraction is serial (PyMuPDF is not thread-safe) and happens BEFORE any
    LLM call so worker threads share immutable strings.

    Pages whose native text layer is empty (scanned pages) are read by Document
    Intelligence instead -- otherwise the enumerator sees a blank page and
    extracts nothing from it.
    """
    from fitz import open as pdoc  # PyMuPDF

    doc = pdoc(str(pdf_path))
    try:
        total_pages = doc.page_count
        s = max(1, start_page)
        e = total_pages if end_page is None else min(end_page, total_pages)
        texts: list[str] = []
        for idx in range(s, e + 1):
            page = doc[idx - 1]
            texts.append(_page_text_with_indent(page) if layout else page.get_text())
    finally:
        doc.close()

    blank = [s + k for k, t in enumerate(texts) if len(t.strip()) < _OCR_BLANK_CHARS]
    if blank:
        di_texts = _di_page_texts(pdf_path, blank)
        for page_no in blank:
            replacement = di_texts.get(page_no)
            if replacement:
                texts[page_no - s] = replacement
    return texts, total_pages


# ---------------------------------------------------------------------------
# Array-oriented response parsing
# ---------------------------------------------------------------------------


def parse_array_response(text: str) -> tuple[list[Any], str]:
    """Parse an LLM response that should be a bare JSON array of items.

    Returns (items, mode) where mode is one of:
      "empty"            no content / empty array
      "bare_array"       [...] parses directly (possibly embedded in prose)
      "fenced_array"     ```json [...] ``` or ``` [...] ```
      "wrapped_object"   {"field": [...]} / {"items": [...]} single-list object
      "salvaged_prefix"  truncated array — recovered complete prefix only
      "salvaged_object"  truncated object wrapper — recovered via v2 salvage
      "failed"           nothing parseable
    """
    if not text or not text.strip():
        return [], "empty"

    stripped = text.strip()

    # 1. Direct parse (bare array).
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, list):
            return (parsed, "bare_array") if parsed else ([], "empty")
        if isinstance(parsed, dict):
            return _list_from_object(parsed, "bare_array")
    except json.JSONDecodeError:
        pass

    # 2. Fenced ```json [...] ``` blocks.
    for fence in FENCE_RE.finditer(stripped):
        body = fence.group(1).strip()
        if not body:
            continue
        try:
            parsed = json.loads(body)
            if isinstance(parsed, list):
                return (parsed, "fenced_array") if parsed else ([], "empty")
            if isinstance(parsed, dict):
                return _list_from_object(parsed, "fenced_array")
        except json.JSONDecodeError:
            continue

    # 3. Bare array embedded in prose — first complete top-level [...] block.
    try_block = _scan_array_block(stripped)
    if try_block is not None:
        return (try_block, "bare_array") if try_block else ([], "empty")

    # 4. Truncated array: salvage the complete prefix of elements.
    salvaged = _salvage_truncated_array(stripped)
    if salvaged is not None:
        return (salvaged, "salvaged_prefix") if salvaged else ([], "empty")

    # 5. Truncated object wrapper: reuse the tested v2 object salvage.
    salvaged_obj = _salvage_truncated_json(stripped)
    if isinstance(salvaged_obj, dict):
        return _list_from_object(salvaged_obj, "salvaged_object")

    # 6. Complete object embedded in prose (reuse v2 block extractor).
    obj = _extract_json_block(stripped)
    if isinstance(obj, dict):
        return _list_from_object(obj, "wrapped_object")

    return [], "failed"


def _list_from_object(obj: dict, mode: str) -> tuple[list[Any], str]:
    """Extract the single list value from an object wrapper, else fail."""
    list_vals = [v for v in obj.values() if isinstance(v, list)]
    if len(list_vals) == 1:
        items = list_vals[0]
        if not items and mode == "bare_array":
            return [], "empty"
        return (items, mode) if items else ([], mode)
    return [], "failed"


def _scan_array_block(text: str) -> Optional[list[Any]]:
    """Return the first complete balanced top-level [...] block, or None."""
    start = text.find("[")
    if start < 0:
        return None
    depth = 0
    in_str = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _salvage_truncated_array(fragment: str) -> Optional[list[Any]]:
    """Recover the complete prefix of elements from a truncated JSON array.

    Array-oriented counterpart of ``_salvage_truncated_json``: computes
    candidate element boundaries at depth 1 (right after a closed element,
    or right before a depth-1 comma), then validates cuts back-to-front by
    appending ``]`` and parsing. Returns the longest parseable prefix, or
    None when nothing parses.
    """
    start = fragment.find("[")
    if start < 0:
        return None
    body = fragment[start:]

    depth = 0
    in_str = False
    escaped = False
    candidates: list[int] = []
    for i, ch in enumerate(body):
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
            if depth == 0:
                try:  # array closed cleanly — parse whole
                    parsed = json.loads(body[: i + 1])
                    return parsed if isinstance(parsed, list) else None
                except json.JSONDecodeError:
                    break
            if depth == 1:
                candidates.append(i + 1)  # an element completed at depth 1
        elif ch == "," and depth == 1:
            candidates.append(i)  # cut just before a trailing separator

    # Truncated: validate candidate cuts back-to-front (latest first) so the
    # kept prefix is the longest complete sequence of elements.
    seen: set[int] = set()
    for cut in reversed(candidates[-200:]):
        if cut <= 0 or cut in seen:
            continue
        seen.add(cut)
        tail = body[:cut].rstrip()
        while tail.endswith(","):
            tail = tail[:-1].rstrip()
        if not tail:
            continue
        try:  # mid-string / mid-scalar cuts simply fail to parse
            parsed = json.loads(tail + "]")
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list):
            return parsed
    return None


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def _page_field_name(item_schema: dict) -> Optional[str]:
    props = item_schema.get("properties") or {}
    for key in props:
        if PAGE_FIELD_RE.search(str(key)):
            return str(key)
    return None


def build_prompts(
    item_schema: dict,
    field_name: str,
    page_first: int,
    page_last: int,
    page_texts: list[str],
    total_pages: int,
) -> tuple[str, str]:
    """Build (system, user) prompts for one page window.

    page_texts[k] is the text of PDF page page_first + k.
    """
    schema_str = json.dumps(item_schema, indent=2, ensure_ascii=False)
    page_field = _page_field_name(item_schema)

    system = (
        "You are a deterministic data-extraction engine reading pages of a PDF. "
        "Your ONLY output is a bare JSON array (no prose, no markdown code "
        "fences, no explanation). Every array element must be one JSON object "
        "matching the item schema exactly. RULES:\n"
        "1) Transcribe text EXACTLY as printed — keep uppercase, punctuation, "
        "odd spellings and abbreviations; do not normalize or title-case.\n"
        "2) Extract EVERY matching record on the given pages; do not skip any.\n"
        "3) Do not invent records that are not on the pages.\n"
        "4) If the pages contain no matching records, output [].\n"
        "5) Output is long; be terse: no extra whitespace."
    )

    sections: list[str] = []
    for offset, text in enumerate(page_texts):
        page_no = page_first + offset
        sections.append(f"=== PAGE {page_no} ===\n{text}\n=== END PAGE {page_no} ===")

    range_desc = (
        f"a single page (page {page_first} of {total_pages})"
        if page_first == page_last
        else f"pages {page_first} to {page_last} of {total_pages}"
    )

    if page_field and page_first == page_last:
        page_rule = (
            f'\n- Set "{page_field}" to the integer {page_first} for every '
            f"item — all records here come from PDF page {page_first} "
            "(one-indexed)."
        )
    elif page_field:
        page_rule = (
            f'\n- Set "{page_field}" to the one-indexed integer PDF page each '
            f"item appears on. Chunk boundaries are marked; every item in the "
            f"=== PAGE N === section comes from exactly page N. Only the "
            f"integers {page_first}..{page_last} are possible."
        )
    else:
        page_rule = ""

    user = (
        f'### TASK\nExtract every item of field "{field_name}" from the page '
        f"chunk below ({range_desc}).\n\n"
        f"### ITEM SCHEMA — each array element must match\n{schema_str}\n\n"
        f"### LAYOUT NOTES\n"
        "- Lines that begin with exactly two extra leading spaces are indented "
        "continuation lines; they belong to the nearest preceding "
        "non-indented line's record (typically address/locale lines).\n"
        "- The primary line of a record (e.g. a person's or company's name) is "
        "flush with the record's left edge.\n"
        "- Guide text (column/county headers between records) is not an item.\n"
        f"{page_rule}\n\n"
        f"### PAGE TEXT\n" + "\n".join(sections) + "\n\n"
        "### OUTPUT\nBare JSON array only. Output [] if none."
    )
    return system, user


# ---------------------------------------------------------------------------
# Item normalization
# ---------------------------------------------------------------------------


def _schema_prop_type(prop: dict) -> str:
    """Effective scalar type of a property spec (first non-null)."""
    if not isinstance(prop, dict):
        return ""
    typ = prop.get("type")
    if isinstance(typ, list):
        non_null = [t for t in typ if t != "null"]
        return str(non_null[0]) if non_null else ""
    if isinstance(prop.get("anyOf"), list):
        for branch in prop["anyOf"]:
            if isinstance(branch, dict) and branch.get("type") not in (None, "null"):
                return str(branch["type"])
    return str(typ or "")


def _has_value(v: Any) -> bool:
    """False for None / blank text / empty containers."""
    if v is None:
        return False
    if isinstance(v, str):
        return bool(v.strip())
    if isinstance(v, (list, dict)):
        return bool(v)
    return True


def _coerce_items(items: list[Any], item_schema: dict) -> tuple[list[dict], int]:
    """Keep dict items; coerce types; drop keys not in the schema.

    Missing schema properties become None (the scorer distinguishes missing
    from null cells, and the GT fills every schema field).
    Returns (normalized_items, dropped_count).

    An item in which EVERY schema field is blank is dropped as well: it is not a
    record. Pages that hold no line items at all -- a glossary page, an
    accounting-data page -- still make the model emit one empty object each, and
    counting those as rows is what turned ``dd1155_schedule_continuation_0011``
    (56 pages, ground truth 20 rows) into a 57-row array: the scale probe read 56
    empty objects from two prose pages, extrapolated 1568 rows over the document,
    the gate swapped the agent's correct 20 rows for 57 empty ones and the
    document scored 0.1864 against 0.9680 before. Blank rows can never match a
    ground-truth record, so dropping them is lossless by construction.
    """
    props: dict = item_schema.get("properties") or {}
    required: list = item_schema.get("required") or list(props)
    types = {
        k: _schema_prop_type(v if isinstance(v, dict) else {}) for k, v in props.items()
    }
    out: list[dict] = []
    dropped = 0
    for item in items:
        if not isinstance(item, dict):
            dropped += 1
            continue
        norm: dict = {}
        for key in props:
            if key in item:
                val = item[key]
                stype = types[key]
                if stype == "integer" and isinstance(val, str):
                    try:
                        val = int(val.strip())
                    except (ValueError, AttributeError, TypeError):
                        pass
                elif stype == "number" and isinstance(val, str):
                    try:
                        val = float(val.strip())
                    except (ValueError, AttributeError, TypeError):
                        pass
                norm[key] = val
            elif key in required:
                norm[key] = None
        if not any(_has_value(v) for v in norm.values()):
            dropped += 1
            continue
        out.append(norm)
    return out, dropped


# ---------------------------------------------------------------------------
# Window planning + orchestration
# ---------------------------------------------------------------------------


def plan_windows(
    start_page: int,
    end_page: Optional[int],
    total_pages: int,
    page_window: int,
) -> list[tuple[int, int]]:
    """Inclusive one-indexed windows [(s, e), ...] clamped to page count."""
    if page_window < 1:
        raise ValueError("page_window must be >= 1")
    s0 = max(1, start_page)
    e0 = total_pages if end_page is None else min(end_page, total_pages)
    if s0 > e0:
        return []
    return [(w, min(w + page_window - 1, e0)) for w in range(s0, e0 + 1, page_window)]


def extract_array_pagewise(
    pdf_path: str,
    item_schema: dict,
    field_name: str,
    *,
    page_window: int = 1,
    max_workers: int = 2,
    start_page: int = 1,
    end_page: Optional[int] = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Accumulate one giant array field with one LLM call per page window.

    Returns {"items": [...], "pages_processed": int, "pages_failed": int,
    "stats": {...}}: per-window detail under stats["window_details"].
    """
    model = _build_model()
    if model is None:
        raise RuntimeError(
            "No chat model configured (set AZURE_OPENAI_* / ADH_MODEL_* env)."
        )

    texts, total_pages = extract_page_texts(
        pdf_path, start_page=start_page, end_page=end_page, layout=True
    )
    s0 = max(1, start_page)
    e0 = min(end_page, total_pages) if end_page is not None else total_pages

    return _run_windows(
        model,
        item_schema,
        field_name,
        texts,
        first_page=s0,
        total_pages=total_pages,
        page_window=page_window,
        max_workers=max_workers,
        verbose=verbose,
        page_range=(s0, e0) if texts else None,
        pdf_path=str(pdf_path),
    )


def probe_array_scale(
    pdf_path: str,
    item_schema: dict,
    field_name: str,
    *,
    sample_pages: int = 2,
    max_workers: int = 2,
    verbose: bool = False,
) -> dict[str, Any]:
    """Estimate how many records ``field_name`` holds in the WHOLE document.

    Why this exists
    ---------------
    The bulk gate fired only when the agent's array was EMPTY. That misses the
    failure it was written for: ``real_ofac_ssi_full`` came back with 11 of 1238
    rows, so the array was *not* empty, the page-wise fallback never ran, and the
    document scored 0.0141 instead of ~0.890.

    Why a sampled extraction instead of a cheap layout statistic
    ------------------------------------------------------------
    Layout proxies do not separate the two cases. Measured over all 49 D3
    documents: ofac yields 0 extractable tables and sits mid-pack on line
    length (median 39 chars, 59% short lines) and density (166 lines/page),
    while ``gov_clin_schedule_0005`` looks far more list-like (77 lines/page)
    yet genuinely holds 13 records across 66 pages. Only the page CONTENT tells
    them apart, so this runs the same page-wise extractor on a couple of evenly
    spaced pages and extrapolates.

    Returns ``{"pages", "sample_pages", "records", "per_page",
    "expected_records", "failed", "elapsed_seconds"}``.
    """
    from fitz import open as pdoc  # PyMuPDF

    with pdoc(str(pdf_path)) as doc:
        total_pages = doc.page_count
    if total_pages <= 0:
        return {
            "pages": 0,
            "sample_pages": [],
            "records": 0,
            "per_page": 0.0,
            "expected_records": 0,
            "failed": 0,
            "elapsed_seconds": 0.0,
        }

    k = max(1, min(int(sample_pages), total_pages))
    if k >= total_pages:
        picks = list(range(1, total_pages + 1))
    else:
        # Interior pages on purpose: page 1 is often a cover or a table of
        # contents, which would underestimate the document.
        picks = sorted(
            {
                min(total_pages, max(1, ((i + 1) * total_pages) // (k + 1)))
                for i in range(k)
            }
        )

    def _one(page: int) -> dict[str, Any]:
        return extract_array_pagewise(
            pdf_path,
            item_schema,
            field_name,
            start_page=page,
            end_page=page,
            page_window=1,
            max_workers=1,
            verbose=verbose,
        )

    t0 = time.time()
    workers = max(1, min(int(max_workers), MAX_WORKERS_CAP, len(picks)))
    if workers == 1:
        outs = [_one(p) for p in picks]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            outs = [f.result() for f in [pool.submit(copy_context().run, _one, p) for p in picks]]

    records = 0
    failed = 0
    for res in outs:
        records += len(list(res.get("items") or []))
        if res.get("pages_failed"):
            failed += 1

    per_page = records / max(1, len(outs))
    return {
        "pages": total_pages,
        "sample_pages": picks,
        "records": records,
        "per_page": round(per_page, 2),
        "expected_records": int(round(per_page * total_pages)),
        "failed": failed,
        "elapsed_seconds": round(time.time() - t0, 1),
    }


def _progress_enabled(verbose: bool = False) -> bool:
    """Window-level progress lines, ON by default for batch jobs.

    A 59-page document with ~450 records per page runs for the better part of an
    hour. With no output there is no way to tell "5 pages done" from "50 done" —
    the stalled 2026-10-19 run was indistinguishable from a healthy one for 40
    minutes, which is why the stall had to be diagnosed from CPU counters.
    ``ADH_BULK_PROGRESS=0`` silences it.
    """
    if verbose:
        return True
    return os.environ.get("ADH_BULK_PROGRESS", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )


def _eta_suffix(done: int, total: int, elapsed: float, workers: int) -> str:
    """Rough remaining-time hint from the average completed-window time."""
    if done <= 0 or done >= total:
        return ""
    per_window = elapsed / done
    remaining_batches = (total - done) / max(1, workers)
    return f" eta~{int(per_window * remaining_batches)}s"


def _run_windows(
    model: Any,
    item_schema: dict,
    field_name: str,
    page_texts: list[str],
    *,
    first_page: int,
    total_pages: int,
    page_window: int,
    max_workers: int,
    verbose: bool = False,
    page_range: Optional[tuple[int, int]] = None,
    pdf_path: str = "",
) -> dict[str, Any]:
    """Core accumulation loop — injectable model, also used by unit tests."""
    t0 = time.time()
    if page_range is None and page_texts:
        # Derive the range from the supplied text. page_texts[i] belongs to page
        # first_page + i — see the offset arithmetic in _process() below — so
        # omitting page_range must not silently produce zero windows.
        page_range = (first_page, first_page + len(page_texts) - 1)
    if page_range is not None:
        windows = plan_windows(page_range[0], page_range[1], total_pages, page_window)
    else:
        windows = []

    workers = max(1, min(int(max_workers), MAX_WORKERS_CAP))
    results: dict[int, dict[str, Any]] = {}

    def _process(window_index: int, w_start: int, w_end: int) -> dict[str, Any]:
        offset0 = w_start - first_page
        chunk = page_texts[offset0 : (w_end - first_page) + 1]
        system, user = build_prompts(
            item_schema, field_name, w_start, w_end, chunk, total_pages
        )
        info: dict[str, Any] = {
            "window": window_index,
            "pages": [w_start, w_end],
            "page_chars": sum(len(c) for c in chunk),
        }
        from autonomous_document_harness.deep.usage import thread_snapshot

        usage_before = thread_snapshot()
        try:
            reply = _invoke_llm(model, system, user)
        except Exception as exc:  # noqa: BLE001 — log and keep going
            info.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
            return info
        usage_after = thread_snapshot()
        window_usage = {
            key: int(usage_after.get(key, 0)) - int(usage_before.get(key, 0))
            for key in ("prompt_tokens", "completion_tokens", "total_tokens", "calls")
        }
        raw_items, mode = parse_array_response(reply)
        norm_items, dropped = _coerce_items(raw_items, item_schema)
        info.update(
            {
                "status": "ok",
                "mode": mode,
                "raw_chars": len(reply),
                "raw_items": len(raw_items),
                "dropped": dropped,
                "items": norm_items,
                "tokens": window_usage,
            }
        )
        return info

    show_progress = _progress_enabled(verbose)
    if windows and show_progress:
        print(
            f"[bulk] {field_name}: {len(windows)} window(s) x {page_window} page(s) "
            f"[{page_range[0]}-{page_range[1]} of {total_pages}] "
            f"with {workers} worker(s)",
            flush=True,
        )

    if windows:
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(copy_context().run, _process, i, s, e): i for i, (s, e) in enumerate(windows)
            }
            for fut in as_completed(futures):
                info = fut.result()
                results[info["window"]] = info
                done += 1
                if show_progress:
                    eta = _eta_suffix(done, len(windows), time.time() - t0, workers)
                    if info["status"] == "ok":
                        print(
                            f"[bulk] {field_name} window "
                            f"{info['window'] + 1}/{len(windows)} "
                            f"pages {info['pages'][0]}-{info['pages'][1]}: "
                            f"mode={info['mode']} items={info.get('raw_items')}{eta}",
                            flush=True,
                        )
                    else:
                        print(
                            f"[bulk] {field_name} window "
                            f"{info['window'] + 1}/{len(windows)} "
                            f"FAILED: {info.get('error')}{eta}",
                            flush=True,
                        )

    # Accumulate in document («window») order.
    items: list[dict] = []
    window_details: list[dict] = []
    pages_processed = 0
    pages_failed = 0
    parse_modes = {
        "empty": 0,
        "bare_array": 0,
        "fenced_array": 0,
        "wrapped_object": 0,
        "salvaged_prefix": 0,
        "salvaged_object": 0,
        "failed": 0,
        "error": 0,
    }
    dropped_total = 0
    for i, (w_start, w_end) in enumerate(windows):
        info = results.get(i, {"status": "error", "error": "no result"})
        window_details.append({k: v for k, v in info.items() if k != "items"})
        items.extend(info.get("items") or [])
        n_pages = w_end - w_start + 1
        if info.get("status") == "ok":
            pages_processed += n_pages
            mode = info.get("mode", "failed")
            if mode in parse_modes:
                parse_modes[mode] += 1
            dropped_total += int(info.get("dropped") or 0)
        else:
            pages_failed += n_pages
            status = info.get("status") or "error"
            if status in parse_modes:
                parse_modes[status] += 1

    bulk_tokens = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "calls": 0,
    }
    for detail in window_details:
        tokens = detail.get("tokens") or {}
        for key in bulk_tokens:
            bulk_tokens[key] += int(tokens.get(key) or 0)

    stats = {
        "pdf_path": str(pdf_path),
        "total_pages": total_pages,
        "page_range": [page_range[0], page_range[1]] if page_range else None,
        "page_window": page_window,
        "workers": workers,
        "windows_total": len(windows),
        "windows_ok": sum(1 for d in window_details if d.get("status") == "ok"),
        "windows_failed": sum(1 for d in window_details if d.get("status") != "ok"),
        "items_extracted": len(items),
        "dropped_non_dict": dropped_total,
        "parse_modes": parse_modes,
        "tokens": bulk_tokens,
        "window_details": window_details,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    return {
        "items": items,
        "pages_processed": pages_processed,
        "pages_failed": pages_failed,
        "stats": stats,
    }


# ---------------------------------------------------------------------------
# Prediction assembly (null skeleton + the extracted array)
# ---------------------------------------------------------------------------


def build_prediction(
    data_schema: dict, field_name: str, items: list[dict]
) -> dict[str, Any]:
    """Prediction with only ``field_name`` populated.

    Mirrors the scorer's null skeleton: non-array scalars -> None, other
    arrays -> [], other objects -> recurse; the extracted array goes under
    ``field_name``.
    """

    def _skeleton(node: dict) -> Any:
        typ = node.get("type")
        if isinstance(typ, list):
            non_null = [t for t in typ if t != "null"]
            typ = non_null[0] if non_null else ""
        if typ == "array":
            return []
        if typ == "object":
            props = node.get("properties") or {}
            return {
                k: _skeleton(sub if isinstance(sub, dict) else {})
                for k, sub in sorted(props.items())
            }
        return None

    out: dict[str, Any] = {}
    for key, spec in sorted((data_schema.get("properties") or {}).items()):
        if key == field_name:
            out[key] = items
        else:
            out[key] = _skeleton(spec if isinstance(spec, dict) else {})
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def run_cli(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Bulk page-wise accumulation extraction of one huge array field "
            "(one LLM call per page window)."
        )
    )
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--test-json", required=True, dest="test_json")
    ap.add_argument("--field", required=True)
    ap.add_argument("--out", required=True, help="prediction.json output path")
    ap.add_argument(
        "--max-pages",
        type=int,
        default=0,
        help="number of pages to process starting at --start-page (0 = all)",
    )
    ap.add_argument("--start-page", type=int, default=1)
    ap.add_argument("--workers", type=int, default=MAX_WORKERS_CAP)
    ap.add_argument("--window", type=int, default=1, help="pages per LLM call")
    ap.add_argument(
        "--raw-text",
        action="store_true",
        help="disable the layout-indent rendering and use plain get_text()",
    )
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    case = json.loads(Path(args.test_json).read_text(encoding="utf-8"))
    data_schema = case.get("data_schema") or {}
    field_spec = (data_schema.get("properties") or {}).get(args.field)
    if not isinstance(field_spec, dict) or "items" not in field_spec:
        print(
            f"ERROR: field '{args.field}' is not an array in data_schema",
            file=sys.stderr,
            flush=True,
        )
        return 2
    item_schema = field_spec["items"]

    end_page = (
        args.start_page + args.max_pages - 1
        if args.max_pages and args.max_pages > 0
        else None
    )

    result = extract_array_pagewise(
        args.pdf,
        item_schema,
        args.field,
        page_window=args.window,
        max_workers=args.workers,
        start_page=args.start_page,
        end_page=end_page,
        verbose=args.verbose,
    )

    prediction = build_prediction(data_schema, args.field, result["items"])
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(prediction, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    report = {
        **result["stats"],
        "field": args.field,
        "prediction_path": str(out_path),
        "pages_processed": result["pages_processed"],
        "pages_failed": result["pages_failed"],
    }
    report_path = out_path.with_name(out_path.stem + ".extraction_run.json")
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "items": len(result["items"]),
                "pages_processed": result["pages_processed"],
                "pages_failed": result["pages_failed"],
                "out": str(out_path),
                "report": str(report_path),
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(run_cli())
