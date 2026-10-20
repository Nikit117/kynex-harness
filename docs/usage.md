# Usage

## The interactive TUI

```bash
python -m autonomous_document_harness.tui
```

Load documents with `/add <path>`, ask questions in plain language, and read
the evidence the agent used. `/help` lists the commands.

## The scriptable entry points

```bash
# stage a document and ask one question
python -m autonomous_document_harness.tui.app run paper.pdf --prompt "extract the totals"

# ask a question about the session that is already loaded
python -m autonomous_document_harness.tui.app ask "which line items are on page 12?" --session my-session
```

The `run` subcommand prints the staged documents, the answer and the evidence
references the agent collected.

## The extraction pipeline

`scripts/extract_one.py` is the same path the benchmark measures: the agent
plus the five post-processing rules, and it writes schema-shaped JSON.

```bash
python autonomous_document_harness/scripts/extract_one.py \
    --source paper.pdf --schema schema.json --out extraction.json
```

The schema is a JSON Schema; the output validates against it.

## The rule switches

Each rule can be turned off, which is how the agent is measured on its own:

| variable | default | what it turns on |
|---|---|---|
| `ADH_BULK_ARRAYS` | off | page-wise enumeration of long record lists |
| `ADH_PRUNE_NON_RECORDS` | on | pruning of contacts, blanks, headers |
| `ADH_REDACTION_MARKERS` | on | a classification marker is a withheld cell |
| `ADH_OCR_FALLBACK` | on | reading pages that have no text layer |
| `ADH_BULK_OCR` | on | the OCR fallback inside the enumerator |
| `ADH_STRUCTURED_ARRAYS` | on | rows rebuilt from the document's own tables |
| `ADH_SCALE_GATE` | on | the sampled probe that decides when to enumerate |

To reproduce the measured behaviour:

```bash
export ADH_BULK_ARRAYS=1 ADH_PRUNE_NON_RECORDS=1 ADH_REDACTION_MARKERS=1
export ADH_OCR_FALLBACK=1 ADH_BULK_OCR=1 ADH_STRUCTURED_ARRAYS=1
```

## Cost and latency

`deep/usage.py` counts the tokens of every model call. The per-document
summary is written next to the extraction, and the measured cost of a full
ExtractBench run is in [benchmarks.md](benchmarks.md).
