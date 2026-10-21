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

### Standard OpenAI-compatible endpoints

Set `ADH_MODEL_BACKEND=openai` to use standard Chat Completions for the agent,
JSON conversion, bulk extraction, and vision calls:

```bash
export ADH_MODEL_BACKEND=openai
export ADH_MODEL_BASE_URL=https://api.example.com/v1
export ADH_MODEL_API_KEY=your-key
export ADH_MODEL_DEPLOYMENT=your-model
```

The base URL includes the endpoint's API prefix (such as `/v1`). Requests use
Bearer authentication and send `ADH_MODEL_DEPLOYMENT` as the model name.
`ADH_MODEL_API_VERSION` is not used in this mode. The default backend and the
existing `openai-compatible` mode continue to use Azure's deployment API.
Document Intelligence is configured separately below.

## Document Intelligence

Configured, the harness reads the document's tables through the layout
model and falls back to OCR for pages that have no text layer. Over the
370-document benchmark run it read tables on **359 of the 370** documents
(2,902 calls) and rewrote the agent's rows from those tables on 8 of them,
so a faithful reproduction needs a Document Intelligence resource.

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
