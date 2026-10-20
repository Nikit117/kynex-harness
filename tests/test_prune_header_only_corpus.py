"""Corpus-wide invariant for the RC-8e rule (clause c).

A rule that deletes a ground-truth record is wrong by definition, no matter
how well it scores. The rule ("no identifier, no numbers, exactly one
populated field whose text is a name rather than prose") is applied to every
ground-truth array item of every document in the corpus (all 8 domains) and
must delete none.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from autonomous_document_harness.scripts import run_tui_benchmark_v2 as R

ROOT = pathlib.Path(__file__).resolve().parents[2]


def test_rule_deletes_no_ground_truth_record():
    if not list(ROOT.glob("data/**/*.test.json")):
        pytest.skip("no corpus in this checkout")
    scanned = 0
    deleted = 0
    offenders = []
    for path in sorted(ROOT.glob("data/**/*.test.json")):
        case = json.loads(path.read_text(encoding="utf-8"))
        for name, value in (case.get("expected_output") or {}).items():
            if not isinstance(value, list):
                continue
            dict_items = [it for it in value if isinstance(it, dict)]
            if not dict_items:
                continue
            field_names: set = set()
            for it in dict_items:
                field_names.update(it.keys())
            for it in dict_items:
                scanned += 1
                if R.non_record_reason(it, field_names) is not None:
                    deleted += 1
                    if len(offenders) < 5:
                        offenders.append(
                            f"{path.name}::{name} {json.dumps(it, ensure_ascii=False)[:140]}"
                        )
    assert scanned > 90_000, f"corpus too small: {scanned}"
    assert deleted == 0, f"ground-truth records would be deleted: {offenders}"
