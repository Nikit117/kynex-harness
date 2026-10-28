"""Regression tests for benchmark schema serialization and JSON salvage.

Context: the v2 ExtractBench run scored 0.7601 mean F1. Two defects were
identified by evidence:

1. ``schema_to_json_prompt`` cut the serialized schema at 15,000 characters
   mid-string, producing INVALID JSON. 33 of 49 D3 schemas exceeded that
   limit, so the agent never saw the top-level property names (for CBP 7501
   all 40 keys were absent) and scored near zero on every field.

2. ``_extract_json_block`` discarded an entire agent answer when the answer
   was cut off by the output token limit.  The erie_county run had extracted
   148 awards but the truncated JSON was thrown away, yielding 48.

These tests pin the fixes so they cannot silently regress.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from autonomous_document_harness.scripts.run_tui_benchmark_v2 import (  # noqa: E402
    _count_array_items,
    _salvage_truncated_json,
    compact_schema_for_prompt,
    non_record_reason,
    prune_non_record_items,
    schema_to_json_prompt,
)


def _big_schema(n_props: int = 40, desc_chars: int = 900) -> dict:
    """Build a schema large enough to trip the old 15,000 char truncation."""
    long_desc = "Detailed field semantics. " * (desc_chars // 26)
    return {
        "$defs": {
            "Address": {
                "type": "object",
                "properties": {
                    "line_one": {"type": "string", "description": long_desc},
                    "city": {"type": "string", "description": long_desc},
                },
            }
        },
        "type": "object",
        "properties": {
            f"field_{i:02d}": {"type": "string", "description": long_desc}
            for i in range(n_props)
        },
    }


class TestCompactSchemaForPrompt:
    def test_emits_valid_json_for_oversized_schema(self):
        schema = _big_schema()
        assert len(json.dumps(schema)) > 15000, "fixture must exceed old limit"
        _, text = compact_schema_for_prompt(schema)
        json.loads(text)  # must not raise

    def test_preserves_every_top_level_property_name(self):
        schema = _big_schema()
        _, text = compact_schema_for_prompt(schema)
        for key in schema["properties"]:
            assert f'"{key}"' in text, f"{key} missing from compacted schema"

    def test_preserves_nested_structure(self):
        schema = _big_schema()
        compact, text = compact_schema_for_prompt(schema)
        assert "$defs" in compact
        assert "Address" in compact["$defs"]
        assert "line_one" in text

    def test_respects_budget_when_achievable(self):
        schema = _big_schema()
        _, text = compact_schema_for_prompt(schema, budget=20000)
        assert len(text) <= 20000

    def test_small_schema_is_untouched(self):
        schema = {
            "type": "object",
            "properties": {"a": {"type": "string", "description": "short"}},
        }
        compact, _ = compact_schema_for_prompt(schema)
        assert compact["properties"]["a"]["description"] == "short"

    def test_never_cuts_mid_string(self):
        # Even an impossible budget must return parseable structure.
        schema = _big_schema()
        _, text = compact_schema_for_prompt(schema, budget=10)
        json.loads(text)


class TestSchemaToJsonPrompt:
    def test_prompt_schema_block_is_valid_json(self):
        prompt = schema_to_json_prompt(_big_schema())
        block = prompt.split("```json")[1].split("```")[0].strip()
        json.loads(block)

    def test_prompt_lists_all_property_names(self):
        prompt = schema_to_json_prompt(_big_schema())
        for i in range(40):
            assert f"field_{i:02d}" in prompt


class TestSalvageTruncatedJson:
    def test_recovers_complete_array_prefix(self):
        rows = ",\n".join('{"i": %d, "v": "row%d"}' % (i, i) for i in range(50))
        truncated = '{\n "name": "X",\n "rows": [\n' + rows + ',\n {"i": 50, "v": "cu'
        out = _salvage_truncated_json(truncated)
        assert out is not None
        assert len(out["rows"]) == 50
        assert out["name"] == "X"

    def test_valid_json_round_trips(self):
        payload = {"a": 1, "b": [1, 2, 3], "c": {"d": None}}
        assert _salvage_truncated_json(json.dumps(payload)) == payload

    def test_nested_truncation(self):
        out = _salvage_truncated_json(
            '{"o": {"rows": [{"x": 1}, {"x": 2}], "label": "ab'
        )
        assert out is not None
        assert out["o"]["rows"] == [{"x": 1}, {"x": 2}]

    def test_truncation_inside_string_is_handled(self):
        out = _salvage_truncated_json(
            '{"a": "com',
        )
        # No complete value -> nothing recoverable, but must not raise.
        assert out is None or isinstance(out, dict)

    def test_returns_none_when_no_json_present(self):
        assert _salvage_truncated_json("no json at all") is None

    def test_markdown_fenced_truncated_json(self):
        out = _salvage_truncated_json(
            '```json\n{"n": 1, "items": [{"k": "a"}, {"k": "b"}]'
        )
        assert out is not None
        assert out["n"] == 1
        assert len(out["items"]) == 2


class TestCountArrayItems:
    def test_counts_nested_lists(self):
        assert _count_array_items({"a": [1, 2, 3], "b": {"c": [4, 5]}}) == 5

    def test_scalars_count_zero(self):
        assert _count_array_items({"a": 1, "b": "x", "c": None}) == 0


class TestDevPoolSourceResolution:
    """The pool must resolve each item to ITS OWN file, not a prefix sibling.

    Regression: ``census_statab`` resolved to
    ``census_statab_income_by_state.pdf`` because the old loop used
    ``sib.name.startswith(item)`` and took the last match. The prediction was
    then scored against a different document's ground truth.
    """

    def test_every_pool_item_resolves_to_its_exact_stem(self):
        from autonomous_document_harness.scripts.run_tui_benchmark_v2 import (
            dev_pool,
        )

        pool = dev_pool("D3")
        if not pool:
            pytest.skip("no corpus in this checkout")
        assert pool, "expected a non-empty D3 pool"
        mismatched = []
        for item_id, info in pool.items():
            stem = Path(info["source"]).stem
            if stem != item_id:
                mismatched.append((item_id, stem))
        assert not mismatched, (
            "source resolution must use the exact stem; offending entries: "
            f"{mismatched[:10]}"
        )

    def test_prefix_sibling_does_not_win(self):
        from autonomous_document_harness.scripts.run_tui_benchmark_v2 import (
            dev_pool,
        )

        pool = dev_pool("D3")
        entry = pool.get("census_statab")
        if entry is None:
            pytest.skip("census_statab not present in this checkout")
        assert Path(entry["source"]).stem == "census_statab"


class TestAntiFabricationClause:
    """ADH_STRICT_VERBATIM must ADD the clause and must not touch the default.

    The default prompt is what produced the current benchmark measurements, so
    it must stay byte-identical unless the operator opts in.
    """

    _SCHEMA = {
        "type": "object",
        "properties": {
            "contract_number": {"type": "string"},
            "clin_items": {"type": "array", "items": {"type": "object"}},
        },
    }

    def test_default_prompt_has_no_clause(self, monkeypatch):
        from autonomous_document_harness.scripts.run_tui_benchmark_v2 import (
            schema_to_json_prompt,
        )

        monkeypatch.delenv("ADH_STRICT_VERBATIM", raising=False)
        prompt = schema_to_json_prompt(self._SCHEMA)
        assert "NEVER invent" not in prompt
        # the exhaustiveness pressure that produced the big gains stays put
        assert "You MUST extract ALL items" in prompt

    def test_enabled_prompt_adds_clause_and_keeps_pressure(self, monkeypatch):
        from autonomous_document_harness.scripts.run_tui_benchmark_v2 import (
            schema_to_json_prompt,
        )

        monkeypatch.setenv("ADH_STRICT_VERBATIM", "1")
        prompt = schema_to_json_prompt(self._SCHEMA)
        assert "NEVER invent" in prompt
        assert "empty array []" in prompt
        assert "You MUST extract ALL items" in prompt

    def test_clause_only_delta_is_the_clause(self, monkeypatch):
        from autonomous_document_harness.scripts.run_tui_benchmark_v2 import (
            schema_to_json_prompt,
        )

        monkeypatch.delenv("ADH_STRICT_VERBATIM", raising=False)
        default = schema_to_json_prompt(self._SCHEMA)
        monkeypatch.setenv("ADH_STRICT_VERBATIM", "yes")
        enabled = schema_to_json_prompt(self._SCHEMA)

        assert enabled != default
        # Everything before the inserted clause must be identical.
        head = default.split("You MUST extract ALL items")[0]
        assert enabled.startswith(head)
        assert "You MUST extract ALL items" in enabled

    def test_off_values_leave_prompt_unchanged(self, monkeypatch):
        from autonomous_document_harness.scripts.run_tui_benchmark_v2 import (
            schema_to_json_prompt,
        )

        monkeypatch.delenv("ADH_STRICT_VERBATIM", raising=False)
        default = schema_to_json_prompt(self._SCHEMA)
        for value in ("", "0", "false", "no"):
            monkeypatch.setenv("ADH_STRICT_VERBATIM", value)
            assert schema_to_json_prompt(self._SCHEMA) == default, value


class TestFillMassiveArrays:
    """RC-4 fallback: fill only arrays the agent left EMPTY.

    Must never touch documents that already work, and must degrade to a no-op
    if the extractor module is absent rather than crashing the doc.
    """

    _SCHEMA = {
        "type": "object",
        "properties": {
            "report_title": {"type": "string"},
            "owners": {"type": "array", "items": {"type": "object"}},
        },
    }
    _MOD = "autonomous_document_harness.scripts.bulk_array_extract"

    def _stub(self, monkeypatch, items):
        import sys as _sys
        import types

        calls = []
        mod = types.ModuleType(self._MOD)

        def extract_array_pagewise(pdf_path, item_schema, field_name, **kw):
            calls.append((pdf_path, field_name))
            return {
                "items": items,
                "pages_processed": 3,
                "pages_failed": 0,
                "stats": {},
            }

        mod.extract_array_pagewise = extract_array_pagewise
        monkeypatch.setitem(_sys.modules, self._MOD, mod)
        return calls

    def _enable(self, monkeypatch, pages=59):
        from autonomous_document_harness.scripts import run_tui_benchmark_v2 as m

        monkeypatch.setenv("ADH_BULK_ARRAYS", "1")
        monkeypatch.setattr(m, "_count_pdf_pages", lambda p: pages)
        return m

    def test_disabled_by_default(self, monkeypatch):
        from autonomous_document_harness.scripts.run_tui_benchmark_v2 import (
            fill_massive_arrays,
        )

        monkeypatch.delenv("ADH_BULK_ARRAYS", raising=False)
        pred = {"owners": None}
        assert fill_massive_arrays(pred, self._SCHEMA, "x.pdf") == {}
        assert pred == {"owners": None}

    def test_populated_array_is_untouched(self, monkeypatch):
        m = self._enable(monkeypatch)
        calls = self._stub(monkeypatch, [{"name": "SHOULD_NOT_APPEAR"}])
        pred = {"owners": [{"name": "REAL"}]}
        assert m.fill_massive_arrays(pred, self._SCHEMA, "x.pdf") == {}
        assert pred == {"owners": [{"name": "REAL"}]}
        assert calls == []

    def test_short_document_is_skipped(self, monkeypatch):
        m = self._enable(monkeypatch, pages=2)
        calls = self._stub(monkeypatch, [{"name": "X"}])
        pred = {"owners": None}
        out = m.fill_massive_arrays(pred, self._SCHEMA, "x.pdf")
        assert "skipped" in out
        assert pred == {"owners": None}
        assert calls == []

    def test_missing_extractor_degrades_to_noop(self, monkeypatch):
        import sys as _sys

        m = self._enable(monkeypatch)
        monkeypatch.setitem(_sys.modules, self._MOD, None)
        pred = {"owners": None}
        out = m.fill_massive_arrays(pred, self._SCHEMA, "x.pdf")
        assert "skipped" in out and "unavailable" in out["skipped"]
        assert pred == {"owners": None}

    def test_empty_array_is_filled(self, monkeypatch):
        m = self._enable(monkeypatch)
        items = [{"name": "COLLINS RICHARD"}, {"name": "EATON BETHINA"}]
        calls = self._stub(monkeypatch, items)
        pred = {"owners": None}
        out = m.fill_massive_arrays(pred, self._SCHEMA, "x.pdf")
        assert pred["owners"] == items
        assert out["fields"]["owners"]["items"] == 2
        assert calls == [("x.pdf", "owners")]

    def test_empty_list_also_filled(self, monkeypatch):
        m = self._enable(monkeypatch)
        self._stub(monkeypatch, [{"name": "A"}])
        pred = {"owners": []}
        m.fill_massive_arrays(pred, self._SCHEMA, "x.pdf")
        assert pred["owners"] == [{"name": "A"}]

    def test_no_array_fields_means_no_work(self, monkeypatch):
        m = self._enable(monkeypatch)
        calls = self._stub(monkeypatch, [{"name": "A"}])
        scalar_only = {
            "type": "object",
            "properties": {"contract_number": {"type": "string"}},
        }
        assert (
            m.fill_massive_arrays({"contract_number": "X"}, scalar_only, "x.pdf") == {}
        )
        assert calls == []

    def test_extractor_exception_is_recorded_not_raised(self, monkeypatch):
        import sys as _sys
        import types

        m = self._enable(monkeypatch)
        mod = types.ModuleType(self._MOD)

        def boom(*a, **kw):
            raise RuntimeError("model exploded")

        mod.extract_array_pagewise = boom
        monkeypatch.setitem(_sys.modules, self._MOD, mod)
        pred = {"owners": None}
        out = m.fill_massive_arrays(pred, self._SCHEMA, "x.pdf")
        assert "error" in out["fields"]["owners"]
        assert pred == {"owners": None}

    def test_real_pdf_page_count_and_gate(self, monkeypatch):
        """Exercise the page gate against a REAL 59-page document.

        The other tests stub _count_pdf_pages, so without this the 5-page gate
        and the real page counter would never be checked on actual data.
        """
        from autonomous_document_harness.scripts import run_tui_benchmark_v2 as m

        pdfs = sorted(m.DATA_DIR.glob("**/real_oklahoma_unclaimed_2024.pdf"))
        if not pdfs:
            pytest.skip("oklahoma PDF not present in this checkout")
        assert m._count_pdf_pages(str(pdfs[0])) == 59

        calls = self._stub(monkeypatch, [{"name": "COLLINS RICHARD"}])
        monkeypatch.setenv("ADH_BULK_ARRAYS", "1")
        pred = {"owners": None}
        out = m.fill_massive_arrays(pred, self._SCHEMA, str(pdfs[0]))
        assert pred["owners"] == [{"name": "COLLINS RICHARD"}]
        assert out["pages"] == 59
        assert out["fields"]["owners"]["items"] == 1
        assert calls and calls[0][1] == "owners"

    def test_real_short_pdf_is_gated_out(self, monkeypatch):
        """A real 1-page doc must NOT trigger bulk extraction."""
        from autonomous_document_harness.scripts import run_tui_benchmark_v2 as m

        pdfs = sorted(m.DATA_DIR.glob("**/census_statab.pdf"))
        if not pdfs:
            pytest.skip("census_statab PDF not present in this checkout")
        assert m._count_pdf_pages(str(pdfs[0])) == 1
        calls = self._stub(monkeypatch, [{"name": "NOPE"}])
        monkeypatch.setenv("ADH_BULK_ARRAYS", "1")
        pred = {"owners": None}
        out = m.fill_massive_arrays(pred, self._SCHEMA, str(pdfs[0]))
        assert "skipped" in out
        assert pred == {"owners": None}
        assert calls == []


class TestPruneNonRecordItems:
    """RC-8: drop array items that are not structured records.

    The clauses are deliberately narrow because the obvious broad version was
    measured as catastrophic: phrased as a bare "identity and numeric fields are
    all blank", clause (b) is vacuously true for any schema that does not use
    those names, and dropped 190/190 items of erie_county and 169/169 of
    penn_hills (i.e. ~40 docs).
    """

    FULL = {
        "item_no": None,
        "parent_item_no": None,
        "supplies_services": None,
        "description": None,
        "quantity": None,
        "unit": None,
        "unit_price": None,
        "amount": None,
    }

    def _item(self, **kw):
        it = dict(self.FULL)
        it.update(kw)
        return it

    # -- RC-8e: clause (c), "no identifier, no numbers, one header-like text" --

    def test_clause_c_header_line_is_dropped(self):
        """The 0046 fabrication of the iteration-8 draw.

        The agent emitted rows that carry no identifier and no numbers, only a
        document header / programme line. The ground truth array is empty, so
        each of them costs precision (the document fell from 0.8333 to 0.1923).
        """
        for text in (
            "807372839",
            "INTEGRATED FEDERAL SOLUTIONS INC.",
            "Troubled Asset Relief Program (TARP)",
            "Acquisition Support",
            "Blanket Purchase Agreement",
        ):
            pred = {"clin_items": [self._item(supplies_services=text)]}
            stats = prune_non_record_items(pred, {})
            assert pred["clin_items"] == [], text
            reasons = stats["fields"]["clin_items"]["reasons"]
            assert (
                "c: no identifier, no numbers, one header-like text field" in reasons
            ), text

    def test_clause_c_keeps_prose_rows(self):
        """A description-only row can be a real line item.

        These are the adversarial guards from RC-8's hardening: a table
        continuation, a prose description, a no-identifier line and a
        no-additional-cost line all have no identifier and no numbers, and all
        of them must survive. Shape alone cannot separate them from a header
        line -- the text itself has to.
        """
        for text in (
            "see attached no 12345 of 9",
            "Line item with no identifier",
            "Furnish and install fiber optic cabling per Section C.3 (continued)",
            "Training for 10 users, included at no additional cost",
        ):
            pred = {"clin_items": [self._item(description=text)]}
            prune_non_record_items(pred, {})
            assert len(pred["clin_items"]) == 1, text

    def test_clause_c_keeps_rows_that_carry_an_identifier_or_numbers(self):
        """The rule only fires when the row cannot be located at all."""
        pred = {
            "clin_items": [
                self._item(item_no="0001", supplies_services="Acquisition Support"),
                self._item(quantity="2", supplies_services="Acquisition Support"),
            ]
        }
        prune_non_record_items(pred, {})
        assert len(pred["clin_items"]) == 2

    def test_clause_c_needs_both_field_kinds_in_the_schema(self):
        """A schema without identity/numeric names is never judged by clause (c).

        The same guard that saved ~40 documents in clause (b) applies here.
        """
        pred = {"rows": [{"label": "Acquisition Support"}]}
        prune_non_record_items(pred, {})
        assert len(pred["rows"]) == 1

    def test_clause_c_can_be_switched_off(self):
        """The rule is replayable, which is how it was measured."""
        from autonomous_document_harness.scripts import run_tui_benchmark_v2 as R

        original = R._PRUNE_HEADER_ONLY_ENABLED
        try:
            R._PRUNE_HEADER_ONLY_ENABLED = False
            pred = {"clin_items": [self._item(supplies_services="Acquisition Support")]}
            prune_non_record_items(pred, {})
            assert len(pred["clin_items"]) == 1
        finally:
            R._PRUNE_HEADER_ONLY_ENABLED = original

    def test_clause_a_identity_label_is_dropped(self):
        """The real 0046 item: item_no holds the envelope marker 'PRIME:'."""
        pred = {"clin_items": [self._item(item_no="PRIME:", supplies_services="SBA")]}
        stats = prune_non_record_items(pred, {})
        assert pred["clin_items"] == []
        assert stats["fields"]["clin_items"]["reasons"] == {
            "a: identity is a bare section label": 1
        }

    def test_clause_a_parent_identity_label_is_dropped(self):
        """The real 0046 item 1: parent_item_no='PRIME:', supplies_services='SUB:'."""
        pred = {
            "clin_items": [
                self._item(parent_item_no="PRIME:", supplies_services="SUB:")
            ]
        }
        prune_non_record_items(pred, {})
        assert pred["clin_items"] == []

    def test_clause_b_blank_identity_and_numeric_with_free_text_is_dropped(self):
        """The real 0046 item 2 / 0059 item 72: only prose populated."""
        pred = {"clin_items": [self._item(supplies_services="Base Year - 3/9/2012")]}
        prune_non_record_items(pred, {})
        assert pred["clin_items"] == []

    def test_real_0046_shape_is_fully_pruned(self):
        """The exact measured 0046 payload -> all 3 items were non-records."""
        pred = {
            "clin_items": [
                self._item(
                    item_no="PRIME:", supplies_services="Small Business Administration"
                ),
                self._item(
                    parent_item_no="PRIME:",
                    supplies_services="SUB:",
                    description="addr",
                ),
                self._item(supplies_services="Base Year - 3/9/2012 through 3/8/2013"),
            ]
        }
        stats = prune_non_record_items(pred, {})
        assert pred["clin_items"] == []
        assert stats["fields"]["clin_items"]["before"] == 3
        assert stats["fields"]["clin_items"]["after"] == 0

    def test_real_0046_contractor_address_block_is_dropped(self):
        """The 2026-10-19 0046 item that clause (b) used to keep.

        Identity and numbers are blank, so clause (a) cannot fire; the text is
        the PRIME/SUB mailing block from the form's own Block 19-20 grid and it
        matches no period/subtotal phrase. The document's ground-truth schedule
        is EMPTY, so keeping it cost 0.2667 of score (0.4000 vs a 0.6667 floor).
        """
        pred = {
            "clin_items": [
                self._item(
                    supplies_services="PRIME: Small Business Administration",
                    description=(
                        "SUB:\n807372839\nINTEGRATED FEDERAL SOLUTIONS INC.\n"
                        "1818 LIBRARY ST STE 500\nRESTON VA 20190-6274\n\n"
                        "Troubled Asset Relief Program (TARP)\n"
                        "Acquisition Support\nBlanket Purchase Agreement"
                    ),
                )
            ]
        }
        stats = prune_non_record_items(pred, {})
        assert pred["clin_items"] == []
        assert stats["fields"]["clin_items"]["reasons"] == {
            "b: label, period or contact text with no identifier and no numbers": 1
        }

    @pytest.mark.parametrize(
        "text",
        [
            "1818 LIBRARY ST STE 500",
            "123 main street",
            "RESTON VA 20190-6274",
            "P.O. Box 44221",
            "PO Box 1188, Springfield",
            "5000 Corporate Center Drive, Suite 300",
        ],
    )
    def test_contact_block_text_is_a_signal(self, text):
        pred = {"clin_items": [self._item(description=text)]}
        prune_non_record_items(pred, {})
        assert pred["clin_items"] == [], text

    def test_address_text_survives_when_the_item_carries_structure(self):
        """The signal is only consulted when identity AND numbers are blank."""
        pred = {
            "clin_items": [
                self._item(item_no="0001", description="Deliver to 1818 LIBRARY ST"),
                self._item(
                    description="Site at RESTON VA 20190-6274", amount="1250.00"
                ),
            ]
        }
        prune_non_record_items(pred, {})
        assert len(pred["clin_items"]) == 2

    def test_lowercase_word_plus_five_digits_is_not_an_address(self):
        """The ZIP alternative is case-sensitive on purpose.

        "no 12345" in prose must not read as the state-and-ZIP shape.
        """
        pred = {"clin_items": [self._item(description="see attached no 12345 of 9")]}
        prune_non_record_items(pred, {})
        assert len(pred["clin_items"]) == 1

    def test_real_0059_subtotal_line_is_dropped(self):
        """0059 item 72: 'Option Period 1 Subtotal (Without Optional Tasks)'."""
        pred = {
            "clin_items": [
                self._item(
                    item_no="0001", supplies_services="X", quantity="1", amount="10.00"
                ),
                self._item(
                    supplies_services="Option Period 1 Subtotal (Without Optional Tasks)"
                ),
            ]
        }
        stats = prune_non_record_items(pred, {})
        assert len(pred["clin_items"]) == 1
        assert stats["fields"]["clin_items"]["dropped"] == 1

    def test_legit_records_survive(self):
        pred = {
            "clin_items": [
                self._item(
                    item_no="0001", quantity="5", unit_price="10.00", amount="50.00"
                ),
                self._item(item_no="0002", supplies_services="Widgets", amount="12.00"),
            ]
        }
        prune_non_record_items(pred, {})
        assert len(pred["clin_items"]) == 2

    # -- catastrophe guards ---------------------------------------------------
    def test_clause_b_does_not_fire_when_items_lack_identity_fields(self):
        """oklahoma/ofac shape: no identity or numeric field names at all.

        This is the erie_county / penn_hills / gsa_0003 guard. Those items have
        no item_no and no quantity/unit_price/amount, so a bare
        "all blanks are blank" test condemns 100% of their legitimate rows.
        """
        pred = {
            "owners": [
                {"name": "COLLINS RICHARD", "locale": "RT 1 BOX 220", "source_page": 2},
                {"name": "EATON BETHINA", "locale": "RT 1 BOX 220", "source_page": 2},
            ]
        }
        before = json.dumps(pred, sort_keys=True)
        stats = prune_non_record_items(pred, {})
        assert json.dumps(pred, sort_keys=True) == before
        assert stats == {}

    def test_clause_b_does_not_fire_when_items_lack_numeric_fields(self):
        """Identity-but-no-numerics shape must be left alone."""
        pred = {"rows": [{"item_no": None, "note": "free text only"}]}
        prune_non_record_items(pred, {})
        assert len(pred["rows"]) == 1

    def test_clause_a_does_not_fire_without_identity_keys(self):
        """A free-text value ending in ':' is not judged if the schema has no
        identity field to judge it against."""
        pred = {"items": [{"label": "PRIME:"}]}
        prune_non_record_items(pred, {})
        assert len(pred["items"]) == 1

    def test_description_only_row_without_structure_is_kept(self):
        """Regression guard for the defect adversarial testing found in cut #1.

        The first shipped version deleted any item with no identifier and no
        number. That is wrong: such a row is a real line item unless its text is
        a label/period/subtotal fragment.
        """
        pred = {"clin_items": [self._item(description="Line item with no identifier")]}
        prune_non_record_items(pred, {})
        assert len(pred["clin_items"]) == 1

    def test_table_continuation_row_survives(self):
        """A table spanning a page break repeats no numbers on the next page."""
        pred = {
            "clin_items": [
                self._item(
                    description="Furnish and install fiber optic cabling per Section C.3 (continued)"
                )
            ]
        }
        prune_non_record_items(pred, {})
        assert len(pred["clin_items"]) == 1

    def test_no_additional_cost_line_survives(self):
        pred = {
            "clin_items": [
                self._item(
                    description="Training for 10 users, included at no additional cost"
                )
            ]
        }
        prune_non_record_items(pred, {})
        assert len(pred["clin_items"]) == 1

    def test_colon_captured_identifier_with_numbers_survives(self):
        """'CLIN 0001:' is a real identifier wearing the layout's colon.

        Cut #1 matched it as a section marker and deleted a fully priced line item.
        """
        pred = {
            "clin_items": [
                self._item(
                    item_no="CLIN 0001:",
                    quantity="5",
                    unit_price="10.00",
                    amount="50.00",
                )
            ]
        }
        prune_non_record_items(pred, {})
        assert len(pred["clin_items"]) == 1

    def test_genuine_section_markers_are_still_dropped(self):
        """The guards above must not weaken the fix the rule exists for."""
        pred = {
            "clin_items": [
                self._item(item_no="PRIME:"),
                self._item(parent_item_no="SUB:"),
                self._item(
                    supplies_services="Option Period 1 Subtotal (Without Optional Tasks)"
                ),
            ]
        }
        prune_non_record_items(pred, {})
        assert pred["clin_items"] == []

    # -- plumbing -------------------------------------------------------------
    def test_kill_switch_disables_pruning(self, monkeypatch):
        monkeypatch.setenv("ADH_PRUNE_NON_RECORDS", "0")
        pred = {"clin_items": [self._item(item_no="PRIME:")]}
        stats = prune_non_record_items(pred, {})
        assert stats == {}
        assert len(pred["clin_items"]) == 1

    def test_enabled_by_default_without_env_var(self, monkeypatch):
        monkeypatch.delenv("ADH_PRUNE_NON_RECORDS", raising=False)
        pred = {"clin_items": [self._item(item_no="PRIME:")]}
        prune_non_record_items(pred, {})
        assert pred["clin_items"] == []

    def test_empty_and_non_list_arrays_untouched(self):
        pred = {
            "a": [],
            "b": None,
            "c": "text",
            "d": [{"item_no": "0001", "amount": "1"}],
        }
        stats = prune_non_record_items(pred, {})
        assert pred["a"] == [] and pred["b"] is None and pred["c"] == "text"
        assert len(pred["d"]) == 1
        assert stats == {}

    def test_mixed_arrays_are_skipped_conservatively(self):
        """An array that is not uniformly dicts is left entirely alone.

        Deliberate: if the array's shape is unexpected, do nothing rather than
        partially mangle it.
        """
        pred = {"x": ["plain string item", {"item_no": "PRIME:"}]}
        stats = prune_non_record_items(pred, {})
        assert pred["x"] == ["plain string item", {"item_no": "PRIME:"}]
        assert stats == {}

    def test_non_record_reason_is_pure(self):
        keys = set(self.FULL)
        assert non_record_reason(self._item(item_no="PRIME:"), keys).startswith("a:")
        assert (
            non_record_reason(self._item(item_no="0001", amount="1.00"), keys) is None
        )
        assert non_record_reason(self._item(), keys) is None


class TestPruneGeneralizesAcrossCorpus:
    """The rule must not delete legitimate records anywhere in the corpus.

    Ground truth is a corpus of records known to be correct, so any deletion from a
    ground-truth array is a false positive *by definition*. This guards the release
    property directly (does it break real documents?) instead of inferring it from
    49 documents of one domain, and it covers every domain in the checkout rather
    than only ``domain:D3``.
    """

    def test_no_ground_truth_item_is_ever_removed(self):
        from autonomous_document_harness.scripts import run_tui_benchmark_v2 as m

        tests = sorted(m.DATA_DIR.glob("**/*.test.json"))
        if not tests:
            pytest.skip("no test items present in this checkout")

        total = 0
        removed = []
        for p in tests:
            doc = json.loads(p.read_text(encoding="utf-8"))
            for field, arr in (doc.get("expected_output") or {}).items():
                if not isinstance(arr, list) or not arr:
                    continue
                if not all(isinstance(x, dict) for x in arr):
                    continue
                keys: set = set()
                for x in arr:
                    keys |= set(x)
                for x in arr:
                    total += 1
                    reason = m.non_record_reason(x, keys)
                    if reason:
                        removed.append((p.name, field, reason))

        assert total > 1000, (
            f"corpus unexpectedly small ({total} items) -- check DATA_DIR"
        )
        assert removed == [], (
            f"rule deleted {len(removed)} legitimate ground-truth records, "
            f"first few: {removed[:5]}"
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
