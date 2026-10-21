# Verifying the KYNEX ExtractBench entry

Everything needed to check the leaderboard row, in one place.

## What was measured

| | |
|---|---|
| system | KYNEX (gpt-5.4) - the harness in this repository |
| corpus | the published ExtractBench set: 370 documents, 4,869 pages |
| scorer | the unmodified official scorer (`extract_bench.evaluation.score_case.score_extract_prediction`) |
| result | **92.65** unified value F1 (94.12 short / 91.52 medium / 79.73 long), rank 12 of 50 |
| cost | USD 689.72 for the run = 14.17 cents per page (public list prices) |
| grounding | 0.00 by construction - the harness emits no per-field citation boxes |

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

* the submitted run: **92.65** overall, and the per-document scores in
  `kynex_official_scores_370.csv`;
* the D3 slice without Document Intelligence: **0.8967** (the same slice with
  Document Intelligence gives 0.8950).

## One honest caveat

The run is one draw of the harness, not a deterministic property of it: two runs
of the same code differ by a mean of about 0.02 overall and by up to 0.3 on a
single document. That is why the per-document scores are published rather than
just the mean.
