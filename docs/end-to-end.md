# End to end: from installation to a leaderboard number

This is the whole path in one page — what you install, what happens when a
document arrives, how the answer is produced, how it is scored, and how the
number ends up on the board. The deeper dives are in
[architecture.md](architecture.md) (the modules) and
[design-notes.md](design-notes.md) (the decisions).

---

## 0. Install

```bash
git clone https://github.com/Nikit117/kynex-harness
cd kynex-harness
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The dependencies are the agent runtime, the document libraries (PyMuPDF,
pdfplumber, pypdfium2, python-docx, openpyxl) and the TUI (textual, rich).
Nothing is fetched at run time except the model calls and the Document
Intelligence calls you configure.

## 1. Point it at a model

```bash
export AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com
export AZURE_OPENAI_API_KEY=<key>
export AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-5.4
export AZURE_OPENAI_API_VERSION=2024-10-21
```

Document Intelligence is optional but recommended — it is what makes the table
rows and the scanned pages work:

```bash
export AZURE_DI_ENDPOINT=https://<resource>.cognitiveservices.azure.com
export AZURE_DI_API_KEY=<key>
```

The harness never stores a credential: it reads the environment, or a `.env`
file you point it at with `ADH_LEGACY_ENV`.

## 2. A document arrives

**In the TUI** you `/add paper.pdf`. **In a script** you pass `--source`.

Either way the document is *staged*: it is copied into the session's workspace
and read once, into three representations the tools can work against —

| representation | produced by | used for |
|---|---|---|
| page text | `parsers/` (PyMuPDF, pdfplumber, pypdfium2, python-docx, openpyxl) | `read_page`, `search_document` |
| tables | the layout model in `tools/azure_di.py` | `find_tables`, `extract_table` |
| page images | a renderer in `tools/` | `inspect_page_visually` and the cropped reads |

A page that has **no text layer** (a scan) would be blank to the text reader,
so the parser falls back to OCR for it. That single fallback is what moves a
scanned 114-page document from 0.10 to 0.97.

## 3. The agent drives

The model is given the schema and a set of tools, and it *chooses* what to
call: read this page, find the tables, extract this one, render that region
because the columns are ambiguous. The loop lives in `deep/agent.py`; the tool
set is registered in `tools/registry.py`.

This is what makes the harness different from a one-shot prompt: a 30-page
13F filing with 1,223 holdings is not something a model can transcribe from
context, but it is something it can *walk*, page by page, if the tools let it.

## 4. One answer, many rows

When the agent is done it returns a single answer. That is fine for a summary
and fatal for a table with thousands of rows: the answer would be truncated, or
the model would answer `null` because the array does not fit.

So the harness measures the *document* rather than the answer. A sampled probe
reads two evenly spaced pages, counts the records there, and extrapolates. If
the document really does hold thousands of records and the agent produced a
small fraction of them, the enumeration runs: every page is read on its own
and the rows are collected.

Two guards keep this honest:

- the enumeration is only used when it finds **more** rows than the agent
  did, so it can never reduce the answer;
- if it finds far fewer rows than the probe predicted, the probe was wrong and
  the agent's answer stands.

## 5. The answer becomes data

The agent's text is parsed into the requested schema
(`answer_to_json_v2`), and then five rules clean it up. Each exists because a
measurement said so:

| # | rule | the failure it removes |
|---|---|---|
| 1 | `answer_to_json_v2` | the answer is prose; the benchmark wants JSON |
| 2 | `fill_massive_arrays` | a 26,725-row array came back as `null` |
| 3 | `prune_non_record_items` | contact blocks, empty items and table headers were counted as records |
| 4 | `normalize_redaction_markers` | `(b)(4)` is a withheld cell, but it was copied as a value |
| 5 | `_apply_structured_arrays` | the row count was right and the *values* were not |

Rule 5 is the one that decides a lot of documents. Over the corpus, **133 of
370 documents have the right number of rows and not one row that matches the
ground truth exactly** — the model read the table but transcribed the currency
symbol, the thousands separator or the column alignment by eye. The layout
model returns the same table structurally, so the rows are rebuilt from the
cells. The rule is deliberately conservative: it only replaces the agent's rows
when the table carries every field the agent filled, and when it has at least
as many rows.

## 6. The result

```json
{
  "line_items": [ { "item_number": "8000", "description": "...", "amount": "NSP" } ]
}
```

Schema-shaped, and — for the rules that need it — verifiable against the
source: the agent's evidence (page, region) travels with the answer, which is
how the harness can tell a real value from an invented one.

## 7. Scoring it

The official scorer is used unmodified. It compares the extracted values with
the ground truth and reports the **unified value F1**: an unweighted mean over
documents, so a one-page document counts exactly as much as a 192-page one.

```bash
# one document
python autonomous_document_harness/scripts/extract_one.py \
    --source paper.pdf --schema schema.json --out extraction.json
```

The ExtractBench provider does the same thing for a whole run, as a subprocess,
which is why the official environment needs no extra packages:

```bash
export KYNEX_HARNESS_ROOT=$PWD
export KYNEX_HARNESS_PYTHON=$PWD/.venv/bin/python
uv run extract-bench run kynex_gpt_5_4
```

See [extractbench.md](extractbench.md) for the registration and the full recipe.

## 8. The number

The measured run over the published 370-document set:

| | value |
|---|---:|
| unified value F1 | **92.65** |
| rank on the board | **12 of 50** |
| documents below 0.50 | 1 |
| cost per page | 14.29 cent |

Full tables, the cost derivation and the caveats are in
[benchmarks.md](benchmarks.md).

---

## The shape of it, in one paragraph

A document is staged into text, tables and images. A model with tools walks
it. The answer is parsed into the schema, then five measured rules repair the
four ways an agent reliably gets a table wrong. The official scorer compares
the result with the ground truth, and the mean over the corpus is the number
on the board.
