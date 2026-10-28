## What this changes

<!-- One paragraph. -->

## How it was measured

<!--
If this touches the extraction pipeline, say which documents changed and by
how much, measured with the official scorer. A rule that cannot be measured
does not ship.
-->

## Checklist

- [ ] `ruff format --check` and `ruff check` pass
- [ ] `pytest -q` passes
- [ ] new behaviour has a kill switch (`ADH_*`)
- [ ] no credential, endpoint or corpus file is committed
- [ ] the documentation is updated
