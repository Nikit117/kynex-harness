# Verifying the KYNEX ExtractBench entry

Everything needed to check the leaderboard row, in one place.

## What was measured

KYNEX has three verified entries on the ExtractBench leaderboard, all using the
same harness code and the unmodified official scorer:

| Configuration | Rank | Overall F1 | Short | Medium | Long | Cost/page | PR |
|---|---:|---:|---:|---:|---:|---:|---|
| **KYNEX (GPT-6.1 Sol)** | **8/53** | **94.6** | 96.0 | 93.1 | 84.2 | 9.35¢ | [#119](https://github.com/run-llama/ExtractBench/pull/119) |
| KYNEX (GPT-5.5) | 14/53 | 94.2 | 95.8 | 93.0 | 80.4 | 8.80¢ | [#119](https://github.com/run-llama/ExtractBench/pull/119) |
| KYNEX (GPT-5.4) | unranked | 92.97 | 94.29 | 92.48 | 78.77 | 4.96¢ | [#84](https://github.com/run-llama/ExtractBench/pull/84) |

| | |
|---|---|
| corpus | the published ExtractBench set: 370 documents, 4,869 pages |
| scorer | the unmodified official scorer (`extract_bench.evaluation.score_case.score_extract_prediction`) |
| grounding | 0.00 by construction - the harness emits no per-field citation boxes |

The GPT-5.4 original submission scored 92.65; LlamaIndex's independent re-run
produced 92.97. All three entries are independently verified by the LlamaIndex team.

## The files in this bundle

| file | contents |
|---|---|
| `kynex_predictions_all_370.zip` | the saved extraction of every document of the submitted run: `prediction.json` (what was scored), `manifest.json`, `run_metrics.json` (per-document telemetry), `raw_answer.txt` (the agent's answer before post-processing) |
| `kynex_predictions_d3_49.zip` | the same, for the 49 documents tagged `domain:D3` in our corpus snapshot |
| `kynex_predictions_d3_49_without_di.zip` | the same, for a second run of those 49 documents with **no Document Intelligence credentials** |
| `kynex_official_scores_370.csv` | the official per-document scores of the submitted run |
| `kynex_d3_49_with_scores.csv` | the official per-document scores of the D3 slice |
| `kynex_d3_scores_with_and_without_di.csv` | the D3 scores of both runs, side by side |
| `SHA256SUMS.txt` | sha256 of every file above, plus the measured harness revision |

## Reproduce it

```bash
git clone https://github.com/Nikit117/kynex-harness
cd kynex-harness
python scripts/verify_against_extractbench.py --data /path/to/extractbench-data --docs 5
```

The script provisions the environment, runs the harness with the same six rule
switches the submitted run used, and prints each document next to the published
per-document score. `--split medium` runs one split, `--data-di off` runs with
no Document Intelligence at all, and the full run is 370 documents.

## The two numbers to compare against

* the GPT-5.4 submitted run: **92.65** overall (verified at 92.97 by
  LlamaIndex), and the per-document scores in `kynex_official_scores_370.csv`;
* the GPT-5.5 run: **94.2** overall, and GPT-6.1 Sol: **94.6** overall — both
  verified in PR #119;
* the D3 slice without Document Intelligence: **0.8967** (the same slice with
  Document Intelligence gives 0.8950).

## One honest caveat

The run is one draw of the harness, not a deterministic property of it: two runs
of the same code differ by a mean of about 0.02 overall and by up to 0.3 on a
single document. That is why the per-document scores are published rather than
just the mean.
