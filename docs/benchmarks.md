# Benchmarks

## ExtractBench

The [ExtractBench](https://github.com/run-llama/ExtractBench) set is 370
documents (4,869 pages): 252 short, 98 medium and 20 long. The numbers below
come from one clean end-to-end run scored with the **unmodified** official
scorer (`extract_bench.evaluation.score_case.score_extract_prediction`).

| metric | value |
|---|---:|
| **unified value F1 (overall)** | **92.65** |
| short | 94.12 |
| medium | 91.52 |
| long | 79.73 |
| precision, short / medium / long | 93.42 / 92.65 / 84.81 |
| recall, short / medium / long | 95.07 / 90.84 / 77.34 |
| documents below 0.50 | 1 |
| cost per page | 14.17 cent |
| latency per document, short / medium / long | 101.9 s / 271.0 s / 867.2 s |

**92.65 puts it 12th of the 50 systems on the published board** - above OpenAI
GPT-6 Astra (91.91) and Codex (GPT-5.6 Terra Evidence, 92.26), and below the
top-10 cut of 93.57. The entry itself is in pull request #84, open with the
maintainers.

The metric is an unweighted mean over documents: a one-page document counts
exactly as much as a 192-page one, which is why the long split is the hard
one -- a long document has many more chances to be wrong.

### The D3 slice

The 49-document D3 slice was measured twice, independently: **0.8950** and
**0.9104**, with **zero documents below 0.50** in both.

The agent read tables through Document Intelligence on **49 of those 49**
documents (404 calls), while the structured-row pass and the OCR fallback
fired on none of them - so on that subset the layout model contributed
through the table reads only. A separate run of the same slice with **no**
Document Intelligence credentials at all is published below, with the
per-document delta.

### The same slice without Document Intelligence

The D3 slice was run again with **no Document Intelligence credentials at all**
(`ADH_DI_ENDPOINT` and `ADH_DI_API_KEY` empty): the agent falls back to the
native table reader, the structured-row pass has no tables to read and the OCR
fallback is unavailable.

| | with ADI | without ADI |
|---|---:|---:|
| mean unified value F1 (49 documents) | 0.8950 | **0.8967** (+0.0016) |
| documents below 0.50 | 0 | 0 |

The structured-row pass and the OCR fallback never fire on this subset, so the
layout model reaches it only through the agent's table reads - and the native
table reader reaches the same values.

### What each rule is worth

| rule | document | before | after |
|---|---|---:|---:|
| page-wise enumeration | `real_oklahoma_unclaimed_2024` | 0.00 | 0.97 |
| non-record pruning | `sf1449_supplies_services_0046` | 0.23 | 0.83 |
| redaction markers | `gov_clin_schedule_0031` | 0.47 | 0.88 |
| structured table rows | `real_franklin_nov_2025` | 0.57 | 1.00 |
| OCR fallback | `real_imedia_full_corrupted` | 0.10 | 0.97 |
| the density pre-filter | `13f__leonteq_securities_2025q4` | 0.44 | 0.99 |

## Cost

| | |
|---|---:|
| input tokens | 183,567,244 |
| output tokens | 10,206,069 |
| model calls | 11,186 |
| documents | 370 |
| pages | 4,869 |

Priced at the public list price of gpt-5.4 ($2.50 / 1M input, $15.00 / 1M
output) plus Azure Document Intelligence Layout analysis ($10.00 / 1,000
pages) for the pages the harness analyses: **USD 690** for the whole run, or
**14.17 cents per page**. The Azure resource's own idle cost is not included,
and a re-run may differ.

## The caveats

They matter, so they are stated rather than buried.

- **The run is one draw.** Two runs of the same code differ by a mean of about
  0.02 overall and by up to 0.3 on a single document. The number above is one
  draw of the harness, not a deterministic property of it.
- **One document is still below 0.50**: `real_ftx_full_corrupted` at 0.4099.
  It holds 7,554 `creditors` rows; the enumeration finds 7,326 of them (97% of
  the rows) and the per-row values drift, and the scorer compares rows
  exactly. The row count is right and the content is not -- the same class of
  problem as the structured-table rule, but at a scale where a per-row check
  is expensive. Every other document is at or above 0.5411.
- **The grounding metrics are 0.00 by construction.** The harness does not
  emit per-field citation boxes, so `field_citations` is empty. This is the
  same situation as the other value-only entries on the board.
- **Nothing was modified to make the number.** The official scorer, the
  schemas and the ground truth are untouched, and the harness' own prompt is
  byte-identical to the default.

The per-document scores of the submitted run are available on request, so any
number here can be checked.
