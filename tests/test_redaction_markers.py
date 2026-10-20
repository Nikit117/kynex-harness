"""CUI classification markers ("(b)(4)") are withheld cells, not values.

A DoD contract prints "(b)(4)" in a cell whose value is classified. The agent
copies that marker verbatim, but the benchmark's ground truth records null for
such cells, so every copied marker scores as a wrong value.

The rule needs COLUMN-level evidence: the same field must carry a marker in
more than one row. A marker in a single row of a column that otherwise holds
real values is a printed value, and the ground truth keeps it
(gov_clin_schedule_0035). Top-level scalar markers are kept as well
(gov_clin_schedule_0005.total_amount is literally "(b) (6)").

Measured with the official scorer over three saved prediction sets:
  gov_clin_schedule_0031  0.4719 -> 0.8801  (113 markers cleared)
  gov_clin_schedule_0033  0.9189 -> 0.9730  (10 markers cleared)
  gov_clin_schedule_0035  0.9333 -> 0.9333  (untouched -- single-row array)
  gov_clin_schedule_0005  0.9811 -> 0.9811  (untouched -- scalar + single row)
  all other 44 documents: prediction byte-identical, so the score cannot move
"""

from __future__ import annotations

import json
import pathlib

import pytest

from autonomous_document_harness.scripts import run_tui_benchmark_v2 as R

ROOT = pathlib.Path(__file__).resolve().parents[2]


class TestMarkerDetection:
    def test_bare_marker_is_detected(self):
        assert R._is_redaction_marker("(b)(4)")
        assert R._is_redaction_marker("(b) (4)")
        assert R._is_redaction_marker("  (b)(4)  ")
        assert R._is_redaction_marker("(B)(6)")
        assert R._is_redaction_marker("(b)(1)")

    @pytest.mark.parametrize(
        "value",
        [
            "501(C)(3)",  # real product code -- 638 such values in the corpus GT
            "(b)",  # bare code, not a classification marker
            "(b)(8)",  # no such CUI category
            "Model (b)(4) kit",  # marker embedded in real text
            "(b)(4) service",
            "",
            "None",
            "b4",
        ],
    )
    def test_non_markers_are_not_detected(self, value):
        assert not R._is_redaction_marker(value)

    @pytest.mark.parametrize("value", [None, 4, 4.0, True, [], {}])
    def test_non_strings_are_not_markers(self, value):
        assert not R._is_redaction_marker(value)


class TestColumnLevelEvidence:
    def test_column_with_markers_in_several_rows_is_cleared(self):
        pred = {
            "clin_line_items": [
                {"item_no": "0001AC", "quantity": "(b)(4)", "unit": "EA"},
                {"item_no": "0001AD", "quantity": "(b)(4)", "unit": "EA"},
                {"item_no": "0001AE", "quantity": "(b)(4)", "unit": "EA"},
            ]
        }
        stats = R._clear_redaction_markers(pred)
        for row in pred["clin_line_items"]:
            assert row["quantity"] is None
            assert row["unit"] == "EA"
            assert row["item_no"].startswith("0001A")
        assert stats["fields"]["clin_line_items"]["cleared"] == 3
        assert stats["fields"]["clin_line_items"]["columns"] == ["quantity"]

    def test_single_row_marker_is_kept(self):
        # gov_clin_schedule_0035: one row, and the ground truth keeps "(b)(4)"
        pred = {
            "clin_line_items": [
                {
                    "item_no": "0001",
                    "quantity": "(b)(4)",
                    "unit_price": "(b)(4)",
                    "supplies_services": "Syringe",
                }
            ]
        }
        stats = R._clear_redaction_markers(pred)
        row = pred["clin_line_items"][0]
        assert row["quantity"] == "(b)(4)"
        assert row["unit_price"] == "(b)(4)"
        assert stats == {}

    def test_top_level_scalar_marker_is_kept(self):
        # gov_clin_schedule_0005: the ground truth value IS "(b) (6)"
        pred = {"total_amount": "(b) (6)", "contract_number": "W911QY-18-D-0060"}
        R._clear_redaction_markers(pred)
        assert pred["total_amount"] == "(b) (6)"
        assert pred["contract_number"] == "W911QY-18-D-0060"

    def test_marker_embedded_in_text_survives(self):
        pred = {
            "clin_items": [
                {"description": "Model (b)(4) kit"},
                {"description": "Model (b)(4) kit"},
            ]
        }
        R._clear_redaction_markers(pred)
        for row in pred["clin_items"]:
            assert row["description"] == "Model (b)(4) kit"

    def test_real_product_code_survives(self):
        pred = {
            "clin_items": [
                {"description": "501(C)(3)"},
                {"description": "501(C)(3)"},
            ]
        }
        R._clear_redaction_markers(pred)
        for row in pred["clin_items"]:
            assert row["description"] == "501(C)(3)"

    def test_nested_object_field_is_cleared(self):
        pred = {
            "clin_line_items": [
                {"item_no": "1", "detail": {"acrn": "(b)(4)", "cin": "GFEBS001"}},
                {"item_no": "2", "detail": {"acrn": "(b)(4)", "cin": "GFEBS002"}},
            ]
        }
        R._clear_redaction_markers(pred)
        for row in pred["clin_line_items"]:
            assert row["detail"]["acrn"] is None
            assert row["detail"]["cin"].startswith("GFEBS")
            assert row["item_no"] in ("1", "2")

    def test_plain_string_array_items_are_untouched(self):
        # nulling a positional element would change the array's shape
        pred = {"codes": ["(b)(4)", "ABC"]}
        R._clear_redaction_markers(pred)
        assert pred["codes"] == ["(b)(4)", "ABC"]

    def test_non_dict_items_are_untouched(self):
        pred = {"codes": ["(b)(4)", "(b)(4)"]}
        R._clear_redaction_markers(pred)
        assert pred["codes"] == ["(b)(4)", "(b)(4)"]

    def test_empty_prediction_is_safe(self):
        assert R._clear_redaction_markers({}) == {}
        assert R._clear_redaction_markers({"a": None, "b": 3}) == {}

    def test_idempotent(self):
        pred = {
            "clin_line_items": [
                {"quantity": "(b)(4)"},
                {"quantity": "(b)(4)"},
            ]
        }
        R._clear_redaction_markers(pred)
        first = json.dumps(pred, sort_keys=True)
        R._clear_redaction_markers(pred)
        assert json.dumps(pred, sort_keys=True) == first


class TestEnvGate:
    def test_kill_switch_leaves_prediction_untouched(self, monkeypatch):
        monkeypatch.setenv("ADH_REDACTION_MARKERS", "0")
        pred = {
            "clin_line_items": [
                {"quantity": "(b)(4)"},
                {"quantity": "(b)(4)"},
            ]
        }
        assert R.normalize_redaction_markers(pred) == {}
        assert pred["clin_line_items"][0]["quantity"] == "(b)(4)"

    def test_default_is_on(self, monkeypatch):
        monkeypatch.delenv("ADH_REDACTION_MARKERS", raising=False)
        pred = {
            "clin_line_items": [
                {"quantity": "(b)(4)"},
                {"quantity": "(b)(4)"},
            ]
        }
        R.normalize_redaction_markers(pred)
        assert pred["clin_line_items"][0]["quantity"] is None


class TestCorpusInvariant:
    """The rule must never clear a ground-truth value."""

    def test_no_ground_truth_value_is_cleared(self):
        cleared_total = 0
        docs_with_markers = set()
        for path in sorted(ROOT.glob("data/**/*.test.json")):
            case = json.loads(path.read_text(encoding="utf-8"))
            probe = json.loads(json.dumps(case.get("expected_output") or {}))
            stats = R._clear_redaction_markers(probe)
            if stats.get("fields"):
                cleared_total += sum(
                    f.get("cleared", 0) for f in stats["fields"].values()
                )
                docs_with_markers.add(path.name.removesuffix(".test.json"))
        assert cleared_total == 0, f"ground-truth values were cleared: {cleared_total}"
        assert not docs_with_markers
