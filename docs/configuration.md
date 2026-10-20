# Configuration

Everything is configured through the environment. There is no configuration
file, and no secret is ever stored in the repository.

## The model

| variable | meaning |
|---|---|
| `AZURE_OPENAI_ENDPOINT` | the Azure OpenAI endpoint |
| `AZURE_OPENAI_API_KEY` | the key |
| `AZURE_OPENAI_CHAT_DEPLOYMENT` | the deployment name (for example `gpt-5.4`) |
| `AZURE_OPENAI_API_VERSION` | the API version |

## Document Intelligence

Optional. When it is configured, the harness reads the document's tables
structurally and falls back to OCR for pages that have no text layer.

| variable | meaning |
|---|---|
| `AZURE_DI_ENDPOINT` | the Document Intelligence endpoint |
| `AZURE_DI_API_KEY` | the key |

## The extraction rules

| variable | default | effect |
|---|---|---|
| `ADH_BULK_ARRAYS` | off | enumerate long record lists page by page |
| `ADH_PRUNE_NON_RECORDS` | on | drop contacts, blank items, header-like rows |
| `ADH_REDACTION_MARKERS` | on | treat `(b)(4)` as a withheld cell, not a value |
| `ADH_OCR_FALLBACK` | on | read pages with no text layer |
| `ADH_BULK_OCR` | on | the OCR fallback inside the enumerator |
| `ADH_STRUCTURED_ARRAYS` | on | rebuild rows from the document's own tables |
| `ADH_SCALE_GATE` | on | the sampled probe that decides when to enumerate |
| `ADH_FULL_SWEEP_HINT` | off | an extra hint in the agent's instructions |

## Credentials from a file

`ADH_LEGACY_ENV` points at a `.env` file to load instead of exporting the
variables in the shell:

```bash
export ADH_LEGACY_ENV=/path/to/config/.env
```

## The ExtractBench provider

| variable | meaning |
|---|---|
| `KYNEX_HARNESS_ROOT` | the harness checkout the provider calls |
| `KYNEX_HARNESS_PYTHON` | the interpreter with the harness' dependencies |
