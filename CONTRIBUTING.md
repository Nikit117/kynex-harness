# Contributing to KYNEX

Thanks for taking the time. This document is short on purpose.

## Getting set up

```bash
git clone https://github.com/Nikit117/kynex-harness
cd kynex-harness
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install pytest ruff
```

## The loop

```bash
ruff format .
ruff check .
pytest -q
```

Keep a pull request small: one change, one measurement, one piece of
documentation.

## Adding an extraction rule

This is the part that matters, because the pipeline is where the score comes
from. A new rule has to clear three bars:

1. **It is measured on a corpus.** Which documents change, and by how much,
   with the official scorer. "It looks better" is not a measurement.
2. **It does not fire where it would hurt.** A rule that improves three
   documents and damages one is a bad trade, and the corpus will show it.
3. **It has a kill switch.** Every rule in `scripts/run_tui_benchmark_v2.py`
   is gated by an `ADH_*` environment variable, so the agent can be measured
   on its own.

The rules that are already there, and the measurements behind them, are in
[docs/design-notes.md](docs/design-notes.md).

## Commits and pull requests

- one logical change per pull request
- a clear title, and the measurement in the body
- do not commit credentials, `.env` files, documents or run artifacts

## Reporting a bug

Use the issue templates. Please do not attach documents that contain
confidential material; a redacted description of the failure is enough to fix
most of them.

## Code of conduct

By participating you agree to abide by the [Code of Conduct](CODE_OF_CONDUCT.md).
