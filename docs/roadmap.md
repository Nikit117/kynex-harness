# Roadmap

What is being worked on, and what is deliberately not planned.

## Next

- **Value-level verification for very large tables.** The remaining
  sub-0.50 document has 97% of its rows and drifts on the values. The fix is
  to check the enumerated rows against the source rather than trusting the
  enumeration, and the cost of that check at 7,554 rows is the open question.
- **A better header matcher for the structured-table rule.** A table whose
  headings are abbreviations rather than field names is currently refused,
  because matching them loosely risks mapping a column to the wrong field.
  The rule needs its own corpus-wide gate before it can ship.

## Being measured

Every change is measured on the corpus before it ships, and the measurement
is published with it. A rule that improves three documents and damages one is
a bad trade, and the corpus shows it.

## Not planned

- **A hosted service.** The harness is local by design: the documents stay on
  your machine and only the pages the agent needs go to the model endpoint.
- **A plugin system for extraction rules.** The pipeline is five rules with
  kill switches, and that is enough. A general rule language would make it
  impossible to say what a run did.
- **Multi-model ensembles.** One deployment, one agent, and the post
  processing that the measurements justify.
