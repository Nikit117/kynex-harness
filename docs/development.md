# Development

## Layout

| path | what it is |
|---|---|
| `autonomous_document_harness/tui/` | the Textual TUI and the harness service |
| `autonomous_document_harness/deep/` | the agent loop, the model bridge, token accounting |
| `autonomous_document_harness/tools/` | the tools the agent calls |
| `autonomous_document_harness/parsers/` | document staging |
| `autonomous_document_harness/workspace/` | the staged corpus and the world model |
| `autonomous_document_harness/models/` | schema, contract and routing models |
| `autonomous_document_harness/verification/` | entailment and completeness checks |
| `autonomous_document_harness/core/` | runtime configuration |
| `autonomous_document_harness/scripts/` | the entry points, the pipeline, the extractors |
| `autonomous_document_harness/eval/extractbench/` | the ExtractBench provider |
| `tests/` | the test suite |
| `docs/` | the documentation |

## The loop

```bash
pip install -r requirements.txt
pip install pytest ruff

ruff format .
ruff check .
pytest -q
```

`ruff` is configured in `pyproject.toml`. The vendored harness is held to
the rules that catch real problems (undefined names, syntax errors); the
unused-name rules are relaxed for the vendored tree, because it is kept as
it was measured. New code outside `autonomous_document_harness/` is held to
the full standard.

## Where the pipeline lives

The five rules are in `autonomous_document_harness/scripts/run_tui_benchmark_v2.py`:

| function | what it does |
|---|---|
| `answer_to_json_v2` | the agent's text becomes schema-shaped JSON |
| `fill_massive_arrays` | enumerate long record lists page by page |
| `prune_non_record_items` | drop contacts, blanks and header-like rows |
| `normalize_redaction_markers` | a classification marker is a withheld cell |
| `_apply_structured_arrays` | rebuild the rows from the document's tables |

`scripts/extract_one.py` applies exactly these five, in this order, and so
does the ExtractBench provider. The provider and the benchmark runner must
never drift apart -- there is a test that pins the order.

## The rule switches

Every rule is gated by an environment variable, so the agent can be measured
on its own:

`ADH_BULK_ARRAYS`, `ADH_PRUNE_NON_RECORDS`, `ADH_REDACTION_MARKERS`,
`ADH_OCR_FALLBACK`, `ADH_BULK_OCR`, `ADH_STRUCTURED_ARRAYS`, `ADH_SCALE_GATE`.

## Reproducing one document

```bash
python autonomous_document_harness/scripts/extract_one.py \
    --source <document.pdf> --schema <schema.json> --out /tmp/one.json
```

## Tests

`pytest -q`. The suite covers the pure functions -- the pruning rules, the
marker rules, the header matcher, the scale gate -- and the ones that need a
document skip themselves when no corpus is present.

## Releasing

1. bump the version in `pyproject.toml`
2. add the entry to `CHANGELOG.md`
3. tag the release and let the workflow publish it
