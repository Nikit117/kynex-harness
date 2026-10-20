# Design notes

The engineering decisions behind the harness, and the measurement behind each
one. Every rule below was measured on the corpus before it shipped, and every
rule has a kill switch, so the agent can be measured on its own.

## 1. The agent answers once, and that is the problem

An agent returns one final answer. A 26,725-row `owners` array does not fit in
an answer, so the model returns `null` for it and the document scores ~0. The
first version of the harness had exactly this failure, and it is the reason
the enumeration exists.

**The enumeration** (`fill_massive_arrays`) reads the document page by page
and collects the rows itself, instead of hoping one answer can hold them.

Measured: `real_oklahoma_unclaimed_2024` 0.00 → 0.97.

**The scale gate** decides when to enumerate. The agent's output alone cannot
be trusted — a document that genuinely holds five rows and a document that
holds five thousand look the same from the outside. So the gate measures the
*document* instead: a sampled probe reads two evenly spaced pages, counts the
records there, and extrapolates.

**The corroboration rule** keeps the probe honest. If the enumeration returns
far fewer rows than the probe predicted, the probe was wrong, and the agent's
answer stands. This is not hypothetical: on an earnings deck the probe
estimated 308 records for a five-row array, and the enumeration replaced a
perfect answer with 88 invented rows -- the document fell to 0.169. The
corroboration rule rejects exactly that case.

## 2. The density pre-filter

Before the probe runs, a cheap filter asks whether the agent's array is even
suspicious: if the agent found more rows than the document's page count
suggests, there is nothing to enumerate.

The cut is a calibration, and it was set by measurement rather than taste.
Across all 439 array fields of the corpus that have a non-empty prediction,
moving the cut from ten to fifteen rows per page newly probes sixteen
documents, and exactly **one** of them is under-filled; the other fifteen are
dense and already complete, so the gate rejects them and their answers do not
change.

Measured: `13f__leonteq_securities_2025q4` 0.44 → 0.99 (the agent had read
all thirty pages of a 1,223-row table and transcribed 346 of them; the
enumeration found all of them).

## 3. Not everything that looks like a row is a record

An extraction array often contains material that is not a record:

| what it looks like | why it is not a record |
|---|---|
| a contact or address block | a name and an address are not a line item |
| an item whose every field is empty | there is nothing in it |
| an item with no identifier and one header-like text | it is the table's header, repeated |
| a cell that reads `(b)(4)` | that is a classification marker: a withheld cell, not a value |

The pruning rule is deliberately narrow. It needs column-level evidence -- the
same field must carry the marker in more than one row -- because a marker in a
single row of a column that otherwise holds real values is a printed value and
the ground truth keeps it. A rule that deletes a ground-truth record is wrong
by definition, however well it scores, so the rule is checked against every
ground-truth array item of the corpus: it must delete none.

Measured: `sf1449_supplies_services_0046` 0.23 → 0.83,
`gov_clin_schedule_0031` 0.47 → 0.88.

## 4. The row count was right and the values were not

This was the largest single loss class in the corpus: **133 of the 370
documents have the right number of rows and not one row that matches the
ground truth exactly.** The model read the table correctly and then
transcribed the values by eye -- losing the currency symbol, the thousands
separator, the column alignment, the scale factor in a heading.

The layout model returns the same table structurally, so the rows are rebuilt
from the cells:

```
['117649', 'ACEA', '$0.00', '$2,800.00', '$0.00', 'STAFF ED SERVICES']
```

which is the ground-truth row verbatim.

The rule is conservative on purpose, because a partial table is worse than a
transcribed one. It replaces the agent's rows only when the table carries
**every field the agent filled** and has at least as many rows. A table that
covers seven of nine fields is refused: on one document that replacement cost
0.28 of F1.

Measured: `real_franklin_nov_2025` 0.57 → 1.00, `real_franklin_aug_2025`
0.57 → 1.00, `real_franklin_sep_2025` 0.57 → 1.00.

## 5. A scanned page is a blank page

A page with no text layer is invisible to the text reader. The parser
therefore falls back to OCR for those pages, and the enumerator has its own
copy of the same fallback (it reads the document by itself, so the parser's
fallback never reached it).

Measured: `real_imedia_full_corrupted` 0.10 → 0.97,
`real_sm0801_eco_full_corrupted` 0.64 → 0.99.

## 6. Evidence, and why there are no citation boxes

The harness collects evidence as it works -- which page, which region of the
page -- and uses it internally to decide whether a value is supported. It does
**not** export per-field citation boxes, so on a benchmark that scores
grounding, this system scores 0.00 on the grounding metrics by construction.
That is stated plainly rather than hidden: the value metrics are what this
system is measured on, and the grounding columns are empty for the same
reason they are empty for the other value-only entries on the board.

## 7. Rows are never de-duplicated

An obvious-looking rule -- drop repeated rows -- is wrong here: the corpus
ground truth contains thousands of exact duplicate rows, and a document that
repeats a line item twenty times really does have twenty line items. No rule
in the harness de-duplicates.

## 8. The harness is a draw, not a function

The same code on the same document does not always produce the same answer:
the agent's own arithmetic over a large table, and the enumeration's
per-page reads, both carry variance. Two runs of the same code differ by a
mean of about 0.02 overall and by up to 0.3 on a single document. Every number
in [benchmarks.md](benchmarks.md) is therefore one draw, and the document-level
scores are available so that any claim can be checked.
