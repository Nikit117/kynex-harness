# Frequently asked questions

### Is this a single prompt?

No. It is an agent loop: the model chooses which tool to call -- page text, a
table, the layout analysis, a rendered region -- and the tools are what let
it read a document that is too large, too ambiguous or too visual to hold in
one context.

### Why are the grounding metrics 0.00?

Because the harness does not export per-field citation boxes. It collects
evidence internally to decide whether a value is supported, but it does not
emit the boxes the grounding metric needs. The value metrics are what this
system is measured on.

### What happens with a scanned document?

The text reader sees a blank page, so the parser falls back to OCR for the
pages that have no text layer, and the enumerator has its own copy of that
fallback because it reads the document by itself. On a scanned 114-page
document this moved the score from 0.10 to 0.97.

### Why did a 26,725-row array come back as null?

Because the agent returns one answer and that array does not fit in it. The
harness measures the document with a sampled probe and, when the document
really does hold thousands of records, enumerates it page by page.

### How is the cost computed?

From the token counts the harness records per document, priced at the public
list price of the model, plus the layout-analysis pages. The full derivation
is in [benchmarks.md](benchmarks.md).

### Can it run without Azure?

Not usefully. The model endpoint is the whole point of the harness. Document
Intelligence is optional: without it the table rows and the OCR fallback are
skipped and the rest still works.

### Can I use a different model deployment?

Yes. The deployment name comes from `AZURE_OPENAI_CHAT_DEPLOYMENT`. The
harness is not tied to one model, although the numbers in
[benchmarks.md](benchmarks.md) were measured with gpt-5.4.

### What do the rule switches do, and can I turn them off?

Every rule in the pipeline is gated by an `ADH_*` variable, and turning them
off is how the agent is measured on its own. The table is in
[configuration.md](configuration.md).

### Why are duplicate rows not removed?

Because the ground truth contains thousands of exact duplicate rows, and a
document that lists the same line item twenty times really does have twenty
line items. De-duplication would cost more than it gains.

### How reproducible is a single run?

Not exactly. Two runs of the same code differ by a mean of about 0.02 overall
and by up to 0.3 on one document -- the agent's own arithmetic over a large
table and the enumeration's per-page reads both carry variance. The
document-level scores are available so any claim can be checked.

### What is the known weak spot?

Long documents with very large tables. The row count comes out right and the
per-row values drift, because the official scorer compares rows exactly. One
document of the 370 is still below 0.50 for that reason.

### How do I contribute a rule?

See [design-notes.md](design-notes.md) for the bar: measured on a corpus, it
must not fire where it would hurt, and it needs a kill switch.

### Can I use it on my own documents?

Yes -- that is what the TUI is for. `/add` a document and ask it questions, or
call `scripts/extract_one.py` with your own schema.
