<p align="center">
  <img src="docs/images/kynex-logo.svg" alt="KYNEX" width="96" height="96">
</p>

<h1 align="center">KYNEX</h1>

<p align="center">
  <strong>Knowledge Yield Nexus for EXtraction</strong><br>
  A local, tool-using agent that reads documents the way a careful analyst does.
</p>

<p align="center">
  <a href="https://github.com/Nikit117/kynex-harness/actions/workflows/ci.yml"><img src="https://github.com/Nikit117/kynex-harness/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-3b82f6.svg" alt="License"></a>
  <img src="https://img.shields.io/badge/python-3.11%2B-3b82f6.svg" alt="Python">
  <img src="https://img.shields.io/badge/ExtractBench-92.65%20unified%20value%20F1-38bdf8.svg" alt="ExtractBench 92.65">
  <a href="https://github.com/run-llama/ExtractBench/pull/84"><img src="https://img.shields.io/badge/leaderboard%20entry-PR%20%2384%20pending-64748b.svg" alt="leaderboard entry pending"></a>
  <img src="https://img.shields.io/badge/docs-docs%2F-3b82f6.svg" alt="Docs">
</p>

<p align="center">
  <a href="#why">Why</a> •
  <a href="#results">Results</a> •
  <a href="#quick-start">Quick start</a> •
  <a href="#how-it-works">How it works</a> •
  <a href="#documentation">Documentation</a> •
  <a href="#contributing">Contributing</a>
</p>

---

KYNEX is an **agentic document-extraction harness**. It stages a document, lets
the model call tools against it — page text, tables, layout analysis, rendered
pages — and returns JSON that validates against the requested schema. It is
not a one-shot prompt, and that is the point: the tools are what let it read a
26,725-row schedule or a scanned page that has no text layer at all.

On the [ExtractBench](https://github.com/run-llama/ExtractBench) leaderboard
(370 documents, scored with the unmodified official scorer) it reaches
**92.65 unified value F1 — rank 12 of 50**, at 14.17 cents per page.

| | Short | Medium | Long | **Overall** |
|---|---:|---:|---:|---:|
| Unified value F1 | 94.12 | 91.52 | 79.73 | **92.65** |
| Precision | 93.42 | 92.65 | 84.81 | |
| Recall | 95.07 | 90.84 | 77.34 | |
| Documents | 252 | 98 | 20 | 370 |
| Latency / document | 101.9 s | 271.0 s | 867.2 s | |

The grounding metrics are **0.00 by construction**: the harness does not emit
per-field citation boxes, so `field_citations` is empty — the same situation as
the other value-only entries on the board.

---

## Why

Extraction benchmarks punish three things that a model does by default:

1. **It answers once.** A 26,725-row `owners` array does not fit in an answer,
   so the agent returns `null` and the document scores ~0.
2. **It reads the table by eye.** It gets the right number of rows and then
   loses the currency symbol, the thousands separator or the column alignment —
   and the scorer compares values exactly. Across the corpus, **133 of the 370
   documents have the right row count and not one row that matches exactly**.
3. **It counts things that are not records.** A contact block, an empty item, a
   table header, a classification marker: all of them look like rows.

KYNEX fixes all three with tools and post-processing, and every rule is
measured on the corpus before it ships. See
[docs/design-notes.md](docs/design-notes.md) for the reasoning behind each
one and the numbers behind them.

---

## Results

Full tables, the cost derivation and the honest caveats:
**[docs/benchmarks.md](docs/benchmarks.md)**.

- 92.65 unified value F1 over the published 370-document set, rank 12 of 50
- the 49-document D3 slice scores 0.8950 and 0.9104 in two independent
  measurements, with **zero** documents below 0.50 in both
- one document of the 370 is still below 0.50 (`real_ftx_full_corrupted`, a
  7,554-row scanned list where the row count is right and the per-row values
  drift) — reported, not hidden

---

## Quick start

```bash
git clone https://github.com/Nikit117/kynex-harness
cd kynex-harness
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Put the credentials in the environment (never in the repository):

```bash
export AZURE_OPENAI_ENDPOINT=...        # the gpt-5.4 deployment
export AZURE_OPENAI_API_KEY=...
export AZURE_OPENAI_CHAT_DEPLOYMENT=...
export AZURE_OPENAI_API_VERSION=...
export AZURE_DI_ENDPOINT=...            # optional: table rows + the OCR fallback
export AZURE_DI_API_KEY=...
```

Then drive it:

```bash
# interactive TUI
python -m autonomous_document_harness.tui

# one document, one schema prompt
python -m autonomous_document_harness.tui.app run paper.pdf --prompt "$(cat prompt.txt)"

# the same pipeline the benchmark measures: agent + the five rules
python autonomous_document_harness/scripts/extract_one.py \
    --source paper.pdf --schema schema.json --out extraction.json
```

Full instructions: [docs/installation.md](docs/installation.md) ·
[docs/usage.md](docs/usage.md) · [docs/configuration.md](docs/configuration.md)

---

## How it works

```
   document
       │
       ▼
  ┌──────────────┐   staged: text, tables, page images
  │   parsers/   │   (+ the OCR fallback for pages with no text layer)
  └──────┬───────┘
         ▼
  ┌──────────────┐   read_page · extract_table · layout analysis ·
  │  the agent  │   vision · search — the model drives the tools
  └──────┬───────┘
         ▼
  ┌──────────────┐   1. answer_to_json_v2
  │ post-process │   2. fill_massive_arrays        (enumerate long lists page by page)
  │              │   3. prune_non_record_items     (contacts, blanks, headers)
  │              │   4. normalize_redaction_markers (a marker is a withheld cell)
  │              │   5. _apply_structured_arrays   (rebuild rows from the tables)
  └──────┬───────┘
         ▼
    schema-shaped JSON
```

| rule | what it fixes | measured on |
|---|---|---|
| page-wise enumeration | a 26,725-row array returned as `null` | `real_oklahoma_unclaimed_2024` 0.00 → 0.97 |
| non-record pruning | contact blocks, empty items, table headers | `sf1449_supplies_services_0046` 0.23 → 0.83 |
| redaction markers | `(b)(4)` is a withheld cell, not a value | `gov_clin_schedule_0031` 0.47 → 0.88 |
| structured table rows | values read structurally, not by eye | `real_franklin_nov_2025` 0.57 → 1.00 |
| OCR fallback | pages with no text layer | `real_imedia_full_corrupted` 0.10 → 0.97 |

Every rule has a kill switch (`ADH_BULK_ARRAYS`, `ADH_PRUNE_NON_RECORDS`,
`ADH_REDACTION_MARKERS`, `ADH_OCR_FALLBACK`, `ADH_BULK_OCR`,
`ADH_STRUCTURED_ARRAYS`), so you can measure the agent on its own.

---

## Showcase

| | |
|---|---|
| **ExtractBench leaderboard** | **92.65** unified value F1, rank **12 of 50** — [pull request #84](https://github.com/run-llama/ExtractBench/pull/84) |
| **The D3 slice** | 0.8950 and 0.9104 in two independent runs, zero documents below 0.50 |

The leaderboard entry is a local subprocess provider, so the official
environment needs no extra packages: point it at a checkout and it runs the
published code.

---

## Limitations, and how these numbers were produced

Being precise about this, because the number is only worth what its method is
worth.

* **The leaderboard entry is still pending.** The measurement is ours; the
  ExtractBench entry is pull request
  [#84](https://github.com/run-llama/ExtractBench/pull/84) and is open. Until
  the maintainers merge it and re-run the pipeline, the 92.65 is a
  self-reported figure with the per-document scores published alongside it.
* **The rules were developed against the same 370 documents the score comes
  from.** That is what iterating on a public benchmark means, and it means the
  number is optimistic about unseen documents. Two things limit the damage: the
  rules are general statements about documents (a contact block is not a
  record; a table read structurally is worth more than one read by eye), not
  document-specific patches, and the per-document scores are published so the
  distribution is visible rather than just the mean.
* **One run is one draw.** Two runs of the same code differ by a mean of about
  0.02 overall and by up to 0.3 on a single document. The number is a draw,
  not a deterministic property of the harness.
* **The grounding metrics are 0.00 by construction**, because the harness does
  not export per-field citation boxes.
* **Document Intelligence is optional** and used for the structured table rows
  and the OCR fallback of image-only pages. The 49-document D3 slice uses it on
  no document at all, so that result is reproducible with the model endpoint
  alone.

---

## Transports

The harness talks to the model through one of two transports, selected with
`ADH_MODEL_BACKEND`:

| backend | endpoint | auth |
|---|---|---|
| `openai-compatible` (default for Azure) | `<base>/openai/deployments/<deployment>/chat/completions?api-version=...` | API key |
| `openai` | `<base>/chat/completions` | Bearer token |

`openai` is opt-in and works with any standard Chat Completions server, so the
harness can run against OpenAI itself or a compatible server you host. The
Azure path is unchanged and stays the default when the Azure variables are
present.

| variable | meaning |
|---|---|
| `ADH_MODEL_BACKEND` | `openai` for the standard Chat Completions transport |
| `ADH_MODEL_BASE_URL` | the endpoint base URL |
| `ADH_MODEL_API_KEY` | the key (sent as a Bearer token in `openai` mode) |
| `ADH_MODEL_DEPLOYMENT` | the model or deployment name |

The transport choice is covered by tests for the agent's tool-calling path and
for the vision path, in both the Azure-compatible and the standard mode.

---

## Documentation

| | |
|---|---|
| [docs/flow.md](docs/flow.md) | a document from upload to answer, step by step |
| [docs/architecture.md](docs/architecture.md) | how the system works, module by module |
| [docs/design-notes.md](docs/design-notes.md) | the engineering decisions and their measured effect |
| [docs/benchmarks.md](docs/benchmarks.md) | the measured results, in full |
| [docs/extractbench.md](docs/extractbench.md) | reproducing the leaderboard run |
| [docs/installation.md](docs/installation.md) | install and configure |
| [docs/usage.md](docs/usage.md) | the TUI, the scriptable entry points, the provider |
| [docs/configuration.md](docs/configuration.md) | every environment variable |
| [docs/development.md](docs/development.md) | working on the code |
| [docs/faq.md](docs/faq.md) | the questions people actually ask |
| [docs/roadmap.md](docs/roadmap.md) | what is next |

---

## Repository layout

```
autonomous_document_harness/
  tui/              the Textual TUI and the harness service
  deep/             the agent loop, the model bridge, token accounting
  tools/            read_page, extract_table, layout analysis, vision, search
  parsers/          PDF, DOCX, XLSX, image staging, the OCR fallback
  workspace/        the staged corpus and the world model
  models/           schema, contract and routing models
  verification/     the entailment and completeness checks
  core/             runtime configuration
  scripts/          extract_one.py, the pipeline, the extractors
  eval/extractbench/ the ExtractBench provider
docs/               the documentation you are reading
tests/              the test suite
```

---

## Contributing

Bug reports, ideas and pull requests are welcome — see
[CONTRIBUTING.md](CONTRIBUTING.md). If you want to add an extraction rule,
[docs/design-notes.md](docs/design-notes.md) explains the bar it has to clear:
it must be measured on the corpus, must not fire where it would hurt, and must
come with a kill switch.

---

## License

Apache-2.0 — see [LICENSE](LICENSE).

The ExtractBench leaderboard entry for this system is
[PR #84](https://github.com/run-llama/ExtractBench/pull/84).

---

## A note on this release

The public tree is the code that was measured, with three things changed for
publication:

1. **The RFQ/CPQ workbook feature is not here.** Its prompts contain
   company-specific material, so the feature, its prompt module and the
   engine it used were removed from the published tree.
2. **The internal project name is not here.** The harness was built for a
   project whose name does not appear in this repository.
3. **A few domain examples were replaced with neutral placeholders of the
   same shape.** The agent's instruction text and a handful of documentation
   strings used the vocabulary of the documents the harness was first written
   for. They are illustrations of the *format* an answer must take, not
   instructions, and they were replaced so the published code says nothing
   about that domain. The measured run used the original text.

None of these changes touch the extraction pipeline, the rules or their
thresholds. The published entry point reproduces the stored prediction of the
measured run -- 178 of 178 rows identical on the document where the
structured-table rule fires -- so the code in this repository is the code that
produced 92.65.

No credential, endpoint, document or ground truth is in this repository.
