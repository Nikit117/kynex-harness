"""Extract one document with the KYNEX and emit schema-shaped JSON.

This is the single-document entry point used by the ExtractBench provider
(``autonomous_document_harness/eval/extractbench/kynex_provider.py``), which
runs it as a subprocess the same way the leaderboard's coding-agent pipelines
run a local CLI.

It uses the same code path as the benchmark runner, so the submitted system is
the system that was measured:

    stage the document (with the OCR fallback for image-only pages)
      -> agent loop with read_page / extract_table / DI / vision tools
      -> answer_to_json_v2
      -> bulk enumeration of long record lists (fill_massive_arrays)
      -> non-record pruning (prune_non_record_items)
      -> redaction-marker rule (normalize_redaction_markers)
      -> structured table rows (_apply_structured_arrays)

    The order of the five post-processing steps is the benchmark runner's
    order; ``tests/test_extract_one_pipeline.py`` pins it.

Usage:
    python scripts/extract_one.py --source <file.pdf> --schema <schema.json> [--out <out.json>]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="path to the document (PDF, ...)")
    ap.add_argument("--schema", required=True, help="path to the JSON schema")
    ap.add_argument(
        "--out", default="", help="write the extraction here (default: stdout)"
    )
    args = ap.parse_args()

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

    from autonomous_document_harness.deep.model import ensure_azure_env
    from autonomous_document_harness.scripts.run_tui_benchmark_v2 import (
        _apply_structured_arrays,
        answer_to_json_v2,
        fill_massive_arrays,
        normalize_redaction_markers,
        prune_non_record_items,
        schema_to_json_prompt,
    )
    from autonomous_document_harness.tui.services.harness import HarnessService

    ensure_azure_env()

    source = pathlib.Path(args.source)
    schema = json.loads(pathlib.Path(args.schema).read_text(encoding="utf-8"))

    session: dict = {
        "id": f"extractone_{source.stem}",
        "documents": [],
        "turns": [],
    }
    svc = HarnessService(session)
    added = svc.add_documents([str(source)])
    if added.get("errors"):
        print(json.dumps({"error": f"staging failed: {added['errors']}"}), flush=True)
        return 2

    result = svc.ask(schema_to_json_prompt(schema))
    prediction = answer_to_json_v2(result.get("text", ""), schema)
    fill_massive_arrays(prediction, schema, str(source))
    prune_non_record_items(prediction, schema)
    normalize_redaction_markers(prediction)
    # Structured rows (built from the document's tables) replace the agent's
    # transcription when they are at least as complete.  This is the last step
    # of the benchmark runner's pipeline and it MUST be here too, otherwise the
    # submitted system is not the system that was measured.
    _apply_structured_arrays(prediction, schema, str(source))

    payload = json.dumps(prediction, indent=2, default=str)
    if args.out:
        pathlib.Path(args.out).write_text(payload, encoding="utf-8")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
