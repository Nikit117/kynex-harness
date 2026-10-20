# Nikit Harness (TUI v3)

A local, tool-using document-extraction agent. It stages a document, lets the
model call tools (`read_page`, `extract_table`, Document Intelligence layout
analysis, vision) and returns schema-shaped JSON.

This is the code that was measured on the
[ExtractBench](https://github.com/run-llama/ExtractBench) leaderboard with the
**unmodified official scorer**:

| metric | value |
|---|---|
| **unified value F1 (overall, 370 documents)** | **92.65** |
| short / medium / long | 94.12 / 91.52 / 79.73 |
| precision / recall (short) | 93.42 / 95.07 |
| precision / recall (medium) | 92.65 / 90.84 |
| precision / recall (long) | 84.81 / 77.34 |
| cost per page | 14.29 ¢ (gpt-5.4 list price + Document Intelligence layout) |
| latency per document (short / medium / long) | 101.9 s / 271.0 s / 867.2 s |
| documents below 0.50 | 1 (`real_ftx_full_corrupted`, 0.4099) |

That is **rank 12 of 50** on the published board. The grounding metrics are
0.00 by construction: the harness emits no per-field citation boxes, so
`field_citations` is empty — the same situation as the other value-only entries
on the board.

## What is in the box

```
autonomous_document_harness/
  tui/            the interactive TUI (Textual) and the harness service
  deep/           the agent loop, the model bridge, token accounting
  tools/          read_page / extract_table / DI layout / vision / search
  parsers/        PDF, DOCX, XLSX, image staging (+ the OCR fallback)
  workspace/      the staged-document corpus and the world model
  models/         the schema and contract models
  verification/   the entailment checks the agent uses to ground its answers
  core/           runtime configuration
  scripts/
    extract_one.py             single-document entry point (the provider calls this)
    run_tui_benchmark_v2.py    the runner and the post-processing pipeline
    bulk_array_extract.py      page-wise enumeration of long record lists
    structured_array.py        table rows read structurally from the document
  eval/extractbench/
    nikit_harness_provider.py  the ExtractBench provider
```

### The post-processing pipeline

The measured score is not "one prompt". It is the agent plus five
post-processing rules applied to its output, in this order:

1. `answer_to_json_v2` — the model's answer becomes schema-shaped JSON;
2. `fill_massive_arrays` — arrays with thousands of rows are enumerated page
   by page (a 26,725-row `owners` array otherwise comes back as `null`);
3. `prune_non_record_items` — contact blocks, blank items, table headers and
   redaction markers are not records;
4. `normalize_redaction_markers` — a CUI classification marker such as
   `(b)(4)` is a withheld cell, not a value;
5. `_apply_structured_arrays` — rows are rebuilt from the document's own
   tables, because the agent transcribes currency symbols, thousands
   separators and column alignment by eye.

## Install

```bash
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.10+.

## Configure

The harness talks to an Azure OpenAI deployment of **gpt-5.4** and to Azure
Document Intelligence. Put the credentials in the environment (or in a `.env`
file that is **never committed** — see `.gitignore`):

| variable | meaning |
|---|---|
| `AZURE_OPENAI_ENDPOINT` | the Azure OpenAI endpoint |
| `AZURE_OPENAI_API_KEY` | the key |
| `AZURE_OPENAI_CHAT_DEPLOYMENT` | the gpt-5.4 deployment name |
| `AZURE_OPENAI_API_VERSION` | the API version |
| `AZURE_DI_ENDPOINT` | Document Intelligence endpoint (optional) |
| `AZURE_DI_API_KEY` | Document Intelligence key (optional) |

`ADH_LEGACY_ENV` can point at a `.env` file to load instead.

## Run

### Interactive TUI

```bash
python -m autonomous_document_harness.tui
```

### Scriptable

```bash
# one document, one schema prompt
python -m autonomous_document_harness.tui.app run <file.pdf> --prompt "<schema prompt>"

# the same pipeline the benchmark measures: agent + the five rules
python autonomous_document_harness/scripts/extract_one.py \
    --source <file.pdf> --schema <schema.json> --out extraction.json
```

The rule switches the measured run used are opt-in in the harness, so set them
when you want the benchmark behaviour:

```bash
export ADH_BULK_ARRAYS=1 ADH_PRUNE_NON_RECORDS=1 ADH_REDACTION_MARKERS=1
export ADH_OCR_FALLBACK=1 ADH_BULK_OCR=1 ADH_STRUCTURED_ARRAYS=1
```

### As an ExtractBench provider

The provider runs `extract_one.py` as a subprocess, like the other
coding-agent pipelines, so the benchmark environment needs no extra packages.

```bash
export NIKIT_HARNESS_ROOT=/path/to/nikit-harness
export NIKIT_HARNESS_PYTHON=/path/to/nikit-harness/.venv/bin/python
uv run extract-bench run nikit_harness_gpt_5_4
```

## Reproducibility notes

* The run is **one draw of the harness**, not a deterministic property of it:
  two runs of the same code differ by a mean of ~0.02 overall and by up to 0.3
  on individual documents.
* No rule de-duplicates rows: the corpus ground truth legitimately contains
  exact duplicate rows, so de-duplication would be wrong.
* The official scorer, the schemas and the ground truth were not modified.

## What is deliberately not here

* the RFQ / CPQ workbook feature and its prompts,
* the benchmark corpus, the ground truth and the run artifacts,
* any credential, `.env` file or endpoint.

## License

Apache-2.0. See `LICENSE`.
