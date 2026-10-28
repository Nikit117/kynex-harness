"""Unit tests for bulk_array_extract — NO LLM calls (model seams patched)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autonomous_document_harness.scripts import bulk_array_extract as bae  # noqa: E402

ITEM_SCHEMA = {
    "type": "object",
    "required": ["name", "locale", "source_page"],
    "properties": {
        "name": {"type": "string"},
        "locale": {"type": ["string", "null"]},
        "source_page": {"type": "integer"},
    },
}


def _patch_llm(monkeypatch, responder):
    """Patch both model seams; responder(system, user) -> reply text."""
    monkeypatch.setattr(bae, "_build_model", lambda: object())
    monkeypatch.setattr(
        bae, "_invoke_llm", lambda model, system, user: responder(system, user)
    )


# ---------------------------------------------------------------------------
# Window planning
# ---------------------------------------------------------------------------


def test_plan_windows_single_page_chunks():
    assert bae.plan_windows(1, None, 59, 1) == [(p, p) for p in range(1, 60)]


def test_plan_windows_multi_page_chunks():
    wins = bae.plan_windows(1, None, 10, 4)
    assert wins == [(1, 4), (5, 8), (9, 10)]


def test_plan_windows_clamps_to_page_count():
    wins = bae.plan_windows(1, 100, 59, 3)
    assert wins[-1] == (58, 59)
    assert all(e <= 59 for s, e in wins)


def test_plan_windows_maxpages_window():
    wins = bae.plan_windows(3, 6, 59, 1)
    assert wins == [(3, 3), (4, 4), (5, 5), (6, 6)]


def test_plan_windows_invalid_start_returns_empty():
    assert bae.plan_windows(60, None, 59, 1) == []


def test_plan_windows_rejects_zero_window():
    with pytest.raises(ValueError):
        bae.plan_windows(1, None, 5, 0)


# ---------------------------------------------------------------------------
# Array-JSON parsing (no LLM involved)
# ---------------------------------------------------------------------------

ARRAY = [{"name": "A B", "locale": None, "source_page": 1}]


def test_parse_bare_array():
    items, mode = bae.parse_array_response(json.dumps(ARRAY))
    assert mode == "bare_array" and items == ARRAY


def test_parse_empty_array_and_empty_text():
    assert bae.parse_array_response("[]") == ([], "empty")
    assert bae.parse_array_response("") == ([], "empty")
    assert bae.parse_array_response("   \n ") == ([], "empty")


def test_parse_fenced_array():
    text = "```json\n" + json.dumps(ARRAY) + "\n```"
    items, mode = bae.parse_array_response(text)
    assert mode == "fenced_array" and items == ARRAY


def test_parse_bare_array_embedded_in_prose():
    text = "Sure! Here are the records:\n["
    text += json.dumps(ARRAY)[1:] + "]\nHope that helps."
    items, mode = bae.parse_array_response(text)
    assert mode == "bare_array" and items == ARRAY


def test_parse_object_wrapped_array():
    text = json.dumps({"owners": ARRAY})
    items, mode = bae.parse_array_response(text)
    assert items == ARRAY and mode == "bare_array"


def test_parse_object_wrapped_in_prose_finds_inner_array():
    # The bare-array-in-prose scan finds the complete inner [...] first:
    prose = "result: " + json.dumps({"owners": ARRAY}) + " (end)"
    items, mode = bae.parse_array_response(prose)
    assert items == ARRAY and mode == "bare_array"


def test_parse_truncated_object_salvages_inner_array_prefix():
    # Truncated object wrapper: missing the array's ']' plus the object's '}'
    # (`{"meta": 1, "owners": [{...name...1`), i.e. cut after the element.
    body = json.dumps({"meta": 1, "owners": ARRAY})[:-2]
    items, mode = bae.parse_array_response(body)
    assert mode == "salvaged_prefix"
    assert items == ARRAY


def test_list_from_object_helpers():
    # single list value -> passes through with the caller's mode
    items, mode = bae._list_from_object({"owners": ARRAY}, "wrapped_object")
    assert items == ARRAY and mode == "wrapped_object"
    # multiple list values is ambiguous -> failed
    items, mode = bae._list_from_object({"a": [1], "b": [2]}, "salvaged_object")
    assert items == [] and mode == "failed"
    # salvaged empty list keeps the salvaged mode (not "empty")
    items, mode = bae._list_from_object({"owners": []}, "salvaged_object")
    assert items == [] and mode == "salvaged_object"


def test_parse_truncated_array_salvage_complete_prefix():
    rows = [
        {"name": f"NAME {i}", "locale": f"ADDR {i}", "source_page": i}
        for i in range(1, 51)
    ]
    full = json.dumps(rows, separators=(", ", ": "))
    body = full[: -len('"source_page": 50}')]  # cut inside the LAST element
    assert "50" in body[-30:]
    items, mode = bae.parse_array_response(body)
    assert mode == "salvaged_prefix"
    assert len(items) == 49  # 50th element incomplete -> dropped
    assert items[0] == rows[0] and items[-1] == rows[48]


def test_parse_truncated_array_with_all_complete_elements_after_cut():
    rows = [{"name": f"N{i}", "locale": None, "source_page": i} for i in range(5)]
    body = json.dumps(rows)[:-20]  # cut partway through element 4
    items, mode = bae.parse_array_response(body)
    assert mode == "salvaged_prefix"
    assert len(items) == 4
    assert items[3] == rows[3]


def test_parse_garbage_fails():
    assert bae.parse_array_response("I could not find any records.") == ([], "failed")


def test_parse_truncated_object_via_v2_salvage():
    # Truncated object wrapper whose list is cut. The array-prefix salvage wins
    # ("salvaged_prefix" = a complete-element prefix was recovered), and the one
    # complete element must survive.
    obj = '{"owners": [{"name": "X Y", "locale": null, "source_page": 1}'
    items, mode = bae.parse_array_response(obj)
    assert mode == "salvaged_prefix"
    assert [i["name"] for i in items] == ["X Y"]
    assert items[0]["source_page"] == 1


# ---------------------------------------------------------------------------
# Item coercion / normalization
# ---------------------------------------------------------------------------


def test_coerce_integer_from_string_and_fill_missing():
    items, dropped = bae._coerce_items(
        [
            {"name": "A", "locale": None, "source_page": "3", "junk": 1},
            {"name": "B"},  # source_page & locale missing -> None / None
            "not a dict",
        ],
        ITEM_SCHEMA,
    )
    assert dropped == 1
    assert items[0] == {"name": "A", "locale": None, "source_page": 3}
    assert items[1] == {"name": "B", "locale": None, "source_page": None}


def test_coerce_keeps_non_digit_strings_for_integers():
    items, _ = bae._coerce_items(
        [{"name": "A", "locale": "7-ELEVEN TR", "source_page": "page 3?"}],
        ITEM_SCHEMA,
    )
    assert items[0]["source_page"] == "page 3?"


def test_coerce_drops_items_with_no_populated_field():
    """A prose page makes the model emit one empty object; that is not a record.

    dd1155_schedule_continuation_0011 (56 pages, ground truth 20 rows): two
    glossary/accounting pages yielded 56 empty objects, the scale probe read them
    as ~1568 extrapolated rows, and the gate replaced the agent's correct 20-row
    array with 57 empty ones -- 0.9680 -> 0.1864.
    """
    items, dropped = bae._coerce_items(
        [
            {},
            {"name": None, "locale": None, "source_page": None},
            {"name": "   ", "locale": ""},
        ],
        ITEM_SCHEMA,
    )
    assert items == []
    assert dropped == 3


def test_coerce_keeps_a_row_with_a_single_populated_field():
    items, dropped = bae._coerce_items(
        [{"name": None, "locale": None, "source_page": 3}], ITEM_SCHEMA
    )
    assert dropped == 0
    assert items == [{"name": None, "locale": None, "source_page": 3}]


# ---------------------------------------------------------------------------
# Accumulation, merge and failure accounting (stubbed model)
# ---------------------------------------------------------------------------


def test_run_windows_accumulates_in_page_order_with_failures(monkeypatch):
    responses = {
        "page1": json.dumps(
            [
                {"name": "A", "locale": None, "source_page": 1},
                {"name": "B", "locale": "X", "source_page": 1},
            ]
        ),
        "page2": None,  # -> simulated hard LLM failure
        "page3": '```json\n[{"name": "C", "locale": null, "source_page": 3}]\n```',
    }

    def responder(system: str, user: str) -> str:
        if "=== PAGE 2 ===" in user:
            raise RuntimeError("simulated 429")
        return responses["page3"] if "=== PAGE 3 ===" in user else responses["page1"]

    _patch_llm(monkeypatch, responder)
    texts = ["p1 text", "p2 text", "p3 text"]
    result = bae._run_windows(
        model="stub",
        item_schema=ITEM_SCHEMA,
        field_name="owners",
        page_texts=texts,
        first_page=1,
        total_pages=3,
        page_window=1,
        max_workers=1,
        verbose=False,
    )
    assert [i["name"] for i in result["items"]] == ["A", "B", "C"]
    assert result["pages_processed"] == 2
    assert result["pages_failed"] == 1
    st = result["stats"]
    assert (
        st["windows_total"] == 3 and st["windows_ok"] == 2 and st["windows_failed"] == 1
    )
    assert st["parse_modes"]["bare_array"] == 1
    assert st["parse_modes"]["fenced_array"] == 1
    assert st["parse_modes"]["error"] == 1
    assert st["items_extracted"] == 3
    details = st["window_details"]
    assert details[1]["status"] == "error" and "simulated 429" in details[1]["error"]


def test_run_windows_respects_max_workers_cap(monkeypatch):
    _patch_llm(monkeypatch, lambda s, u: "[]")
    result = bae._run_windows(
        model="stub",
        item_schema=ITEM_SCHEMA,
        field_name="owners",
        page_texts=["x"] * 5,
        first_page=1,
        total_pages=5,
        page_window=1,
        max_workers=99,
    )
    # Requesting more than the cap must be clamped to the cap, not honoured.
    assert result["stats"]["workers"] == bae.MAX_WORKERS_CAP
    assert bae.MAX_WORKERS_CAP < 99


def test_run_windows_empty_pages_yields_nothing(monkeypatch):
    _patch_llm(monkeypatch, lambda s, u: "[]")
    result = bae._run_windows(
        model="stub",
        item_schema=ITEM_SCHEMA,
        field_name="owners",
        page_texts=["x"],
        first_page=1,
        total_pages=1,
        page_window=1,
        max_workers=1,
    )
    assert result["items"] == []
    assert result["pages_processed"] == 1
    assert result["pages_failed"] == 0
    assert result["stats"]["parse_modes"]["empty"] == 1


def test_run_windows_window_of_two_pages_uses_range_hints(monkeypatch):
    """Multi-page windows must tell the model the possible page numbers."""
    captured: dict[str, str] = {}

    def responder(system: str, user: str) -> str:
        captured["user"] = user
        return json.dumps([{"name": "A", "locale": None, "source_page": 1}])

    _patch_llm(monkeypatch, responder)
    bae._run_windows(
        model="stub",
        item_schema=ITEM_SCHEMA,
        field_name="owners",
        page_texts=["p1", "p2"],
        first_page=1,
        total_pages=2,
        page_window=2,
        max_workers=1,
    )
    assert "pages 1 to 2 of 2" in captured["user"]
    assert "1..2" in captured["user"] or "1 through 2" in captured["user"]


# ---------------------------------------------------------------------------
# End-to-end through extract_array_pagewise with a REAL (tiny) PDF
# ---------------------------------------------------------------------------


def _make_pdf(path: Path, pages: list[list[tuple[float, str]]]) -> None:
    """pages[k] = list of (x0, text) drawn top to bottom."""
    import fitz

    doc = fitz.open()
    for lines in pages:
        page = doc.new_page()
        y = 40.0
        for x, text in lines:
            page.insert_text((x, y), text, fontname="cour", fontsize=8)
            y += 12.0
    doc.save(str(path))
    doc.close()


def test_extract_pagewise_end_to_end_with_stub(monkeypatch, tmp_path):
    pdf = tmp_path / "tiny.pdf"
    _make_pdf(
        pdf,
        [
            [(50.0, "SMITH JOHN"), (54.5, "12 OAK ST")],
            [(50.0, "JONES MARY")],
        ],
    )

    def responder(system: str, user: str) -> str:
        if "=== PAGE 1 ===" in user:
            return json.dumps(
                [
                    {"name": "SMITH JOHN", "locale": "12 OAK ST", "source_page": 1},
                ]
            )
        return json.dumps(
            [
                {"name": "JONES MARY", "locale": None, "source_page": 2},
            ]
        )

    _patch_llm(monkeypatch, responder)
    result = bae.extract_array_pagewise(
        str(pdf),
        ITEM_SCHEMA,
        "owners",
        max_workers=1,
    )
    assert result["pages_processed"] == 2 and result["pages_failed"] == 0
    assert result["items"] == [
        {"name": "SMITH JOHN", "locale": "12 OAK ST", "source_page": 1},
        {"name": "JONES MARY", "locale": None, "source_page": 2},
    ]


def test_ident_renderer_marks_indented_lines(tmp_path):
    pdf = tmp_path / "indent.pdf"
    _make_pdf(
        pdf,
        [
            [
                (50.0, "FLUSH NAME"),
                (54.5, "INDENT CONT"),  # +4.5pt -> continuation marker
                (50.0, "FLUSH2"),
            ],
        ],
    )
    from fitz import open as pdoc

    doc = pdoc(str(pdf))
    try:
        text = bae._page_text_with_indent(doc[0])
    finally:
        doc.close()
    lines = text.splitlines()
    flush_idx = [l for l in lines if l.strip() == "FLUSH NAME"]
    indent_idx = [l for l in lines if l.strip() == "INDENT CONT"]
    assert flush_idx == ["FLUSH NAME"], text  # no marker on flush line
    assert indent_idx == ["  INDENT CONT"], text  # two-space marker
    assert any(l.strip() == "FLUSH2" and not l.startswith(" ") for l in lines)


def test_extract_page_texts_range_and_total(monkeypatch, tmp_path):
    pdf = tmp_path / "three.pdf"
    _make_pdf(
        pdf,
        [[(50.0, f"P{i}")] for i in (1, 2, 3)],
    )
    texts, total = bae.extract_page_texts(str(pdf), start_page=2, end_page=3)
    assert total == 3
    assert len(texts) == 2
    assert "P2" in texts[0] and "P3" in texts[1]


def test_extract_array_pagewise_no_model_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(bae, "_build_model", lambda: None)
    pdf = tmp_path / "x.pdf"
    _make_pdf(pdf, [[(50.0, "A")]])
    with pytest.raises(RuntimeError):
        bae.extract_array_pagewise(str(pdf), ITEM_SCHEMA, "owners")


# ---------------------------------------------------------------------------
# Prediction assembly
# ---------------------------------------------------------------------------


def test_build_prediction_null_skeleton_with_field_items():
    data_schema = {
        "properties": {
            "report_title": {"type": "string"},
            "publication": {"type": "string"},
            "owners": {"type": "array", "items": ITEM_SCHEMA},
        }
    }
    pred = bae.build_prediction(
        data_schema,
        "owners",
        [{"name": "A", "locale": None, "source_page": 1}],
    )
    assert pred["report_title"] is None and pred["publication"] is None
    assert pred["owners"] == [{"name": "A", "locale": None, "source_page": 1}]


def test_build_prediction_other_arrays_stay_empty():
    data_schema = {
        "properties": {
            "owners": {"type": "array", "items": ITEM_SCHEMA},
            "checks": {"type": "array"},
        }
    }
    pred = bae.build_prediction(
        data_schema, "owners", [{"name": "x", "locale": None, "source_page": 1}]
    )
    assert pred["checks"] == []
