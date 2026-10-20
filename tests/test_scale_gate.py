"""Tests for the scale gate: enumerate by DOCUMENT scale, not agent output.

The 2026-10-19 clean run exposed the flaw in the original gate. It fired only
when the agent's array was EMPTY, but ``real_ofac_ssi_full`` (192 pages,
1238 ground-truth entities) came back with 11 entities -- not empty -- so the
page-wise fallback never ran and the document scored 0.0141 while the same
document scores ~0.89 when the enumeration does run.

The gate now also fires when a sampled extraction says the document holds many
more rows than the agent produced. ``expected_records`` values below are the
row counts the probe is expected to estimate for each document (for the
documents the agent already handles, that estimate equals the ground-truth
count, which is why they must NOT be routed).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from autonomous_document_harness.scripts import bulk_array_extract as B  # noqa: E402
from autonomous_document_harness.scripts import run_tui_benchmark_v2 as R  # noqa: E402


def _schema() -> dict:
    return {
        "properties": {
            "entities": {
                "type": "array",
                "items": {"properties": {"name": {"type": "string"}}},
            },
            "title": {"type": "string"},
        }
    }


def _rows(n: int) -> list[dict]:
    return [{"name": f"entity-{i}"} for i in range(n)]


# --- the pure decision rule -------------------------------------------------


def test_routes_the_ofac_shape():
    route, reason = R.scale_gate_decision(
        pages=192, agent_items=11, expected_records=1150
    )
    assert route, reason


def test_routes_the_oklahoma_shape():
    route, reason = R.scale_gate_decision(
        pages=59, agent_items=0, expected_records=26725
    )
    assert route, reason


def test_does_not_route_documents_that_genuinely_hold_few_rows():
    """These are real D3 documents the agent handles; the probe estimate for
    them is their true row count, and none may be routed."""
    cases = [
        ("gov_clin_schedule_0005", 66, 13, 13),
        ("dd1155_schedule_continuation_0012", 68, 5, 5),
        ("sf1449_supplies_services_0042", 77, 5, 5),
        ("jackson_college_2024_federal_awards_audit", 16, 11, 11),
        ("gsa_mas_it_labor_0008", 33, 48, 48),
        ("erie_county_2017_single_audit", 40, 190, 189),
        ("real_penn_hills_pa_2023", 34, 354, 207),
        ("gov_clin_schedule_0031", 62, 23, 23),
    ]
    for name, pages, agent, expected in cases:
        route, reason = R.scale_gate_decision(
            pages=pages, agent_items=agent, expected_records=expected
        )
        assert not route, f"{name} must not be routed: {reason}"


def test_does_not_route_a_short_document():
    route, reason = R.scale_gate_decision(
        pages=3, agent_items=1, expected_records=10_000
    )
    assert not route and "pages" in reason


def test_does_not_route_when_the_agent_already_found_a_good_share():
    """RC-10c: 60% of the expected rows is enough -- no enumeration needed."""
    route, reason = R.scale_gate_decision(
        pages=192, agent_items=700, expected_records=1150
    )
    assert not route and "60%" in reason


def test_routes_when_the_agent_materially_under_filled():
    """RC-10c: the old rule needed a 4x shortfall; 60% is the new bar.

    `13f__leonteq_securities_2025q4` (agent 346 of 1223) and
    `real_credit_strategies_full` (agent 68 of 588) were never enumerated
    because the old test demanded a 4x gap.
    """
    for agent, expected in ((346, 1223), (68, 588), (400, 1150)):
        route, reason = R.scale_gate_decision(
            pages=100, agent_items=agent, expected_records=expected
        )
        assert route, f"agent={agent} expected={expected} must be routed: {reason}"


# --- wiring: fill_massive_arrays -------------------------------------------


def _patch_probe(monkeypatch, *, expected: int, record_calls: list | None = None):
    def fake_probe(pdf_path, item_schema, field_name, **kwargs):
        if record_calls is not None:
            record_calls.append(field_name)
        return {
            "pages": 100,
            "sample_pages": [33, 66],
            "records": max(1, expected // 50),
            "per_page": expected / 100,
            "expected_records": expected,
            "failed": 0,
            "elapsed_seconds": 4.2,
        }

    monkeypatch.setattr(B, "probe_array_scale", fake_probe)


def _patch_extractor(monkeypatch, items: list, record_calls: list | None = None):
    def fake_extract(pdf_path, item_schema, field_name, **kwargs):
        if record_calls is not None:
            record_calls.append(field_name)
        return {
            "items": items,
            "pages_processed": 100,
            "pages_failed": 0,
            "stats": {},
        }

    monkeypatch.setattr(B, "extract_array_pagewise", fake_extract)


def _enable(monkeypatch):
    monkeypatch.setenv("ADH_BULK_ARRAYS", "1")
    monkeypatch.delenv("ADH_SCALE_GATE", raising=False)
    monkeypatch.setattr(R, "_count_pdf_pages", lambda _p: 100)


def test_routes_a_small_non_empty_array_when_the_document_is_huge(monkeypatch):
    """The ofac case: 11 rows returned, ~1150 expected."""
    _enable(monkeypatch)
    _patch_probe(monkeypatch, expected=1150)
    _patch_extractor(monkeypatch, _rows(1160))

    prediction = {"entities": _rows(11), "title": "OFAC"}
    stats = R.fill_massive_arrays(prediction, _schema(), "doc.pdf")

    assert len(prediction["entities"]) == 1160
    field = stats["fields"]["entities"]
    assert field["routed"] is True and field["replaced"] is True
    assert field["agent_items"] == 11 and field["expected_records"] == 1150


def test_does_not_replace_a_good_agent_array_with_a_smaller_enumeration(monkeypatch):
    _enable(monkeypatch)
    _patch_probe(monkeypatch, expected=1150)
    _patch_extractor(monkeypatch, _rows(4))

    prediction = {"entities": _rows(11), "title": "OFAC"}
    stats = R.fill_massive_arrays(prediction, _schema(), "doc.pdf")

    assert len(prediction["entities"]) == 11
    assert stats["fields"]["entities"]["replaced"] is False


def test_empty_array_still_fires_without_paying_for_the_probe(monkeypatch):
    """The original trigger must keep working, and must not probe."""
    _enable(monkeypatch)
    probe_calls: list = []
    _patch_probe(monkeypatch, expected=26_000, record_calls=probe_calls)
    _patch_extractor(monkeypatch, _rows(26_000))

    prediction = {"entities": [], "title": "OK"}
    stats = R.fill_massive_arrays(prediction, _schema(), "doc.pdf")

    assert len(prediction["entities"]) == 26_000
    assert probe_calls == [], "empty arrays must not pay for a probe"
    assert stats["fields"]["entities"]["reason"] == "array empty"


def test_documents_with_a_large_agent_array_are_never_probed(monkeypatch):
    """A dense array is not suspicious, so it must not pay for a probe.

    RC-13: the bar is now fifteen rows per page, so 2 000 rows on a 100-page
    document is dense enough to be left alone.
    """
    _enable(monkeypatch)
    probe_calls: list = []
    _patch_probe(monkeypatch, expected=1150, record_calls=probe_calls)
    _patch_extractor(monkeypatch, _rows(5000))

    prediction = {"entities": _rows(2000), "title": "doc"}
    stats = R.fill_massive_arrays(prediction, _schema(), "doc.pdf")

    assert probe_calls == []
    assert stats == {} or "entities" not in stats.get("fields", {})
    assert len(prediction["entities"]) == 2000


def test_the_leonteq_shape_now_pays_for_the_probe(monkeypatch):
    """RC-13: the case the density cut was written for, at the real numbers.

    ``13f__leonteq_securities_2025q4`` holds ~40 holdings per page over 31
    pages (1 223 ground-truth rows).  The agent returned 346 of them, which is
    11.2 rows per page -- just above the old ten-row cut, so the probe never
    ran, the gate never fired and the document stayed at 0.4364.  The probe
    measures that document at 1 224 expected records, so the gate routes it.
    """
    _enable(monkeypatch)
    monkeypatch.setattr(R, "_count_pdf_pages", lambda _p: 31)
    probe_calls: list = []
    _patch_probe(monkeypatch, expected=1224, record_calls=probe_calls)
    _patch_extractor(monkeypatch, _rows(1223))

    prediction = {"entities": _rows(346), "title": "doc"}
    R.fill_massive_arrays(prediction, _schema(), "doc.pdf")

    assert probe_calls == ["entities"], "a 12-rows-per-page array must be probed"


def test_the_density_cut_is_fifteen_rows_per_page():
    """The calibration is pinned, with the corpus evidence that set it."""
    assert R.SCALE_DENSITY_ROWS_PER_PAGE == 15


def test_a_sparse_array_now_pays_for_the_probe(monkeypatch):
    """RC-10a: 300 rows on a 100-page document IS suspicious (3 rows/page).

    The old pre-filter ("fewer rows than pages") let this through untouched and
    `real_credit_strategies_full` (50 pages, agent 68 of 588 rows) kept only
    12% of the ground truth because the gate never even looked at it.
    """
    _enable(monkeypatch)
    probe_calls: list = []
    _patch_probe(monkeypatch, expected=1150, record_calls=probe_calls)
    _patch_extractor(monkeypatch, _rows(1160))

    prediction = {"entities": _rows(300), "title": "doc"}
    R.fill_massive_arrays(prediction, _schema(), "doc.pdf")

    assert probe_calls == ["entities"], "a sparse array must be probed"


def test_the_enumeration_must_corroborate_the_probe(monkeypatch):
    """RC-10b: a probe that over-estimated keeps the agent's answer.

    On `hpe_earnings_deck_q4fy25` the probe estimated 308 records for a 5-row
    array (an earnings deck: every slide carries a segment table) and the
    enumeration replaced the agent's PERFECT five rows with 88 invented ones,
    dropping the document to 0.169. When the enumeration returns far fewer
    records than the probe predicted, the probe was wrong.
    """
    _enable(monkeypatch)
    _patch_probe(monkeypatch, expected=308)
    _patch_extractor(monkeypatch, _rows(88))

    prediction = {"entities": _rows(5), "title": "deck"}
    stats = R.fill_massive_arrays(prediction, _schema(), "doc.pdf")

    assert len(prediction["entities"]) == 5, "the agent's rows must survive"
    field = stats["fields"]["entities"]
    assert field["replaced"] is False
    assert field["probe_not_corroborated"]["ratio"] == round(88 / 308, 3)


def test_a_corroborated_enumeration_is_used(monkeypatch):
    """The normal case: the enumeration matches the probe, so it is used."""
    _enable(monkeypatch)
    _patch_probe(monkeypatch, expected=1150)
    _patch_extractor(monkeypatch, _rows(1160))

    prediction = {"entities": _rows(11), "title": "OFAC"}
    stats = R.fill_massive_arrays(prediction, _schema(), "doc.pdf")

    assert len(prediction["entities"]) == 1160
    assert stats["fields"]["entities"]["replaced"] is True
    assert "probe_not_corroborated" not in stats["fields"]["entities"]


def test_scale_gate_kill_switch_restores_the_old_behaviour(monkeypatch):
    """With the gate off, a non-empty array is left completely alone -- the
    original empty-array-only behaviour."""
    _enable(monkeypatch)
    monkeypatch.setenv("ADH_SCALE_GATE", "0")
    probe_calls: list = []
    _patch_probe(monkeypatch, expected=1150, record_calls=probe_calls)
    _patch_extractor(monkeypatch, _rows(1160))

    prediction = {"entities": _rows(11), "title": "OFAC"}
    stats = R.fill_massive_arrays(prediction, _schema(), "doc.pdf")

    assert probe_calls == []
    assert len(prediction["entities"]) == 11
    assert stats == {}, "a non-empty array must not even be examined"


def test_bulk_stays_off_by_default(monkeypatch):
    monkeypatch.delenv("ADH_BULK_ARRAYS", raising=False)
    _patch_probe(monkeypatch, expected=1150)
    _patch_extractor(monkeypatch, _rows(1160))

    prediction = {"entities": _rows(11), "title": "OFAC"}

    assert R.fill_massive_arrays(prediction, _schema(), "doc.pdf") == {}
    assert len(prediction["entities"]) == 11


def test_a_probe_failure_does_not_break_the_document(monkeypatch):
    _enable(monkeypatch)

    def boom(*_a, **_k):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(B, "probe_array_scale", boom)
    _patch_extractor(monkeypatch, _rows(5000))

    prediction = {"entities": _rows(11), "title": "OFAC"}
    stats = R.fill_massive_arrays(prediction, _schema(), "doc.pdf")

    assert stats["fields"]["entities"]["routed"] is False
    assert "probe failed" in stats["fields"]["entities"]["reason"]
    assert len(prediction["entities"]) == 11


# --- the probe must count RECORDS, not empty objects ------------------------


def _corpus_pdf(name: str):
    return next(R.DATA_DIR.glob(f"**/{name}.pdf"), None)


def test_probe_counts_no_records_when_the_pages_hold_no_real_rows(monkeypatch):
    """dd1155_schedule_continuation_0011: prose pages yield empty objects.

    Measured 2026-10-19: pages 18 (a glossary) and 37 (accounting data) of that
    56-page document made the model return empty objects. Counting them as rows
    extrapolated ~1584 rows, the gate replaced a correct 20-row array with 57
    empty ones, and the document fell from 0.9680 to 0.1864.
    """
    pdf = _corpus_pdf("dd1155_schedule_continuation_0011")
    if pdf is None:
        import pytest

        pytest.skip("corpus document not present in this checkout")

    monkeypatch.setattr(B, "_build_model", lambda: object())
    monkeypatch.setattr(B, "_invoke_llm", lambda model, system, user: "[{}, {}, {}]")

    out = B.probe_array_scale(
        str(pdf), {"properties": {"name": {"type": "string"}}}, "line_items"
    )

    assert out["records"] == 0
    assert out["expected_records"] == 0
    route, _reason = R.scale_gate_decision(
        pages=out["pages"], agent_items=20, expected_records=out["expected_records"]
    )
    assert route is False, "a document with no sampled rows must not be routed"


def test_probe_still_counts_rows_that_carry_content(monkeypatch):
    """The empty-row filter must not blind the probe to real records."""
    pdf = _corpus_pdf("dd1155_schedule_continuation_0011")
    if pdf is None:
        import pytest

        pytest.skip("corpus document not present in this checkout")

    monkeypatch.setattr(B, "_build_model", lambda: object())
    monkeypatch.setattr(
        B, "_invoke_llm", lambda model, system, user: '[{"name": "alpha"}]'
    )

    out = B.probe_array_scale(
        str(pdf), {"properties": {"name": {"type": "string"}}}, "line_items"
    )

    assert out["records"] == out["sample_pages"].__len__() * 1
    assert out["expected_records"] > 0
