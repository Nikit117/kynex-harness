# Running the ExtractBench benchmark

How to reproduce the leaderboard run with this harness.

## 1. Get the harness and a model

```bash
git clone https://github.com/Nikit117/kynex-harness
cd kynex-harness
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

```bash
export AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com
export AZURE_OPENAI_API_KEY=<key>
export AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-5.4
export AZURE_OPENAI_API_VERSION=2024-10-21
export AZURE_DI_ENDPOINT=https://<resource>.cognitiveservices.azure.com
export AZURE_DI_API_KEY=<key>
```

## 2. Check one document first

```bash
python autonomous_document_harness/scripts/extract_one.py \
    --source <document.pdf> --schema <schema.json> --out /tmp/one.json
```

If that produces schema-shaped JSON with the right number of rows, the
environment is right. A full run is 370 documents and costs a few hundred
dollars, so check one first.

## 3. Register the provider

The provider is a **local subprocess** provider, like the coding-agent
pipelines already on the board: it shells out to the harness' own entry point,
so the benchmark environment needs no extra Python packages.

In `src/extract_bench/inference/providers/extract/__init__.py`, add the module
to the list:

```python
_PROVIDER_MODULES = [
    ...
    "kynex",
]
```

In `src/extract_bench/inference/pipelines/extract.py`, register the pipeline:

```python
register_fn(
    _pipeline_spec(
        pipeline_name="kynex_gpt_5_4",
        provider_name="kynex_extract",
        config={"timeout": 3600},
        # The harness enumerates long record lists page by page, so a
        # 192-page document needs more than the 600s run default.
        per_file_timeout=3600.0,
    )
)
```

The provider file itself is in this repository at
`autonomous_document_harness/eval/extractbench/kynex_provider.py`.

## 4. Tell the provider where the harness is

```bash
export KYNEX_HARNESS_ROOT=/path/to/kynex-harness
export KYNEX_HARNESS_PYTHON=/path/to/kynex-harness/.venv/bin/python
```

## 5. Run it

```bash
uv run extract-bench run kynex_gpt_5_4
```

A quick smoke run first is a good idea:

```bash
uv run extract-bench run kynex_gpt_5_4 --test
```

## 6. The rule switches

The measured run had all the rules enabled. Most of them are on by default;
the enumeration is opt-in, so set it explicitly:

```bash
export ADH_BULK_ARRAYS=1 ADH_PRUNE_NON_RECORDS=1 ADH_REDACTION_MARKERS=1
export ADH_OCR_FALLBACK=1 ADH_BULK_OCR=1 ADH_STRUCTURED_ARRAYS=1
```

The provider passes these to the subprocess itself, so the run is reproducible
without any of them being set in the benchmark environment.

## 7. What a good run looks like

- 370 documents scored, no document failing to produce an extraction
- an overall unified value F1 in the low nineties
- the long split (20 documents) around 0.80
- roughly 190,000 input tokens and 10,000 output tokens per document on
  average, and a run cost near USD 700

If the long split comes back much lower, the enumeration is probably not
running -- check `ADH_BULK_ARRAYS`.
