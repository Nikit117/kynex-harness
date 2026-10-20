# Installation

## Requirements

- Python 3.11 or newer
- an Azure OpenAI deployment of **gpt-5.4**
- optionally an Azure Document Intelligence resource (for the table rows and
  the OCR fallback)

## From source

```bash
git clone https://github.com/Nikit117/kynex-harness
cd kynex-harness
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## The credentials

The harness reads its configuration from the environment. Put it in a `.env`
file in the working directory (the repository ignores `.env*`) or export it in
your shell:

```bash
export AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com
export AZURE_OPENAI_API_KEY=<your-key>
export AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-5.4
export AZURE_OPENAI_API_VERSION=2024-10-21
```

Document Intelligence is optional; without it the table rows and the OCR
fallback are skipped and the rest of the harness works:

```bash
export AZURE_DI_ENDPOINT=https://<your-resource>.cognitiveservices.azure.com
export AZURE_DI_API_KEY=<your-key>
```

`ADH_LEGACY_ENV` points the harness at a `.env` file elsewhere, which is handy
when you keep several configurations side by side.

## Check the installation

```bash
python -m autonomous_document_harness.tui --help
pytest -q
```

## Installing for the ExtractBench provider

The provider runs the harness as a subprocess, so the benchmark environment
needs no extra packages; it needs a checkout and an interpreter:

```bash
export KYNEX_HARNESS_ROOT=/path/to/kynex-harness
export KYNEX_HARNESS_PYTHON=/path/to/kynex-harness/.venv/bin/python
```

See [extractbench.md](extractbench.md) for the full recipe.
