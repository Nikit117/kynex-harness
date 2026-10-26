# Announcement kit

Everything below is ready to paste. The demo image is
`docs/images/kynex-demo.svg` (download it from the repo, or use the rendered
README on GitHub).

**The wording is deliberate.** The score is measured with the unmodified
official scorer and the per-document results are published, so it can be
checked. The rank is a computed position among the systems on the published
board, not an official listing: the entry is in pull request #84, open with the
maintainers. The cost is the list price of the run, with no free-tier
assumption. Nothing here needs anyone else's approval to be true.

---

## 1. X / Twitter (the one that usually travels)

> I built an open-source agent that reads documents properly.
>
> KYNEX is a local, tool-using extraction agent: it stages a PDF, calls tools
> against it (page text, tables, layout analysis, rendered pages) and returns
> JSON that matches your schema.
>
> It is not a one-shot prompt. That is the point - it recovers a 1,223-row
> filing, enumerates a 26,725-row schedule, and OCRs a 114-page scan.
>
> It scores **92.65 unified value F1 on ExtractBench** (370 documents,
> unmodified official scorer) - **12th of the 50 systems on the board by
> score**, above OpenAI GPT-6 Astra, at 14.17 cents/page. The leaderboard entry
> is in pull request #84 with the maintainers.
>
> Open source: github.com/Nikit117/kynex-harness
>
> [demo image]

---

## 2. LinkedIn (the one that gets a recruiter's attention)

> Most "document AI" is a prompt with a PDF attached. It works on the demo and
> falls over on the real thing: a 30-page filing with 1,223 rows comes back as
> a summary, and a scan comes back blank.
>
> So I built **KYNEX** - a local, tool-using extraction agent. The model
> decides which tools to call (page text, tables, layout analysis, rendered
> pages), the harness measures the document rather than the answer, and five
> measured rules repair what agents reliably get wrong about tables.
>
> The part I am happiest with is the measurement, not the model. The rules were
> not guessed: over the 370-document ExtractBench corpus, **133 documents have
> the right number of rows and not one row that matches the ground truth
> exactly** - the agent reads the table and then loses the currency symbol, the
> thousands separator or the column alignment. Rebuilding the rows from the
> document's own tables is worth more than any prompt change.
>
> **92.65 unified value F1 on ExtractBench** - 12th of the 50 systems on the
> board by score, above OpenAI GPT-6 Astra, at 14.17 cents per page. The
> official scorer, the schemas and the ground truth are untouched; the
> per-document scores, the predictions and a sha256 manifest are published, so
> every number can be checked rather than trusted. The leaderboard entry is in
> pull request #84 with the maintainers.
>
> Open source: github.com/Nikit117/kynex-harness
>
> [demo image]

---

## 3. Hacker News (Show HN)

**Title**

> Show HN: KYNEX - an open-source agent that extracts structured data from documents (92.65 F1 on ExtractBench)

**First comment**

> Hi HN - I am the author of KYNEX, a local, tool-using extraction agent for
> PDFs and other documents. It stages a document, lets the model call tools
> against it (page text, tables, layout analysis, rendered pages) and returns
> JSON that validates against a JSON Schema.
>
> Two things make it different from "a prompt with a PDF attached":
>
> 1. **It measures the document, not the answer.** An agent returns one final
>    answer, so a 26,725-row array does not fit in it and comes back as null. A
>    sampled probe reads two pages, counts the records and extrapolates, and
>    only then does the harness enumerate the document page by page.
> 2. **The rules were measured, not guessed.** On the 370-document ExtractBench
>    corpus, 133 documents have the right row count and not one row that
>    matches exactly - the agent reads the table and then loses the currency
>    symbol, the thousands separator or the column alignment. Rebuilding the
>    rows from the document's own tables is worth more than any prompt change.
>
> It scores 92.65 unified value F1 on ExtractBench with the unmodified
> official scorer - 12th of the 50 systems on the board by score - at 14.17
> cents per page. The per-document scores are in the repo, so the claim can be
> checked rather than trusted. The leaderboard entry is in pull request #84
> with the maintainers.
>
> It also runs without Azure Document Intelligence at all: on the 49-document
> D3 slice, 0.8950 with the layout model and 0.8967 without it, with no
> document below 0.50 either way.
>
> Repo: github.com/Nikit117/kynex-harness
> Docs: docs/end-to-end.md
> The leaderboard entry: github.com/run-llama/ExtractBench/pull/84
>
> Feedback welcome - in particular on the two things I know are weak: long
> documents with very large tables (the row count comes out right, the per-row
> values drift), and the fact that one run of the harness is one draw, not a
> deterministic function.

---

## 4. Subreddits

**r/LocalLLaMA**

> Open-sourced an agentic document-extraction harness. It scores 92.65 unified
> value F1 on ExtractBench (370 documents, official scorer) - 12th of the 50
> systems on the board by score. The interesting part was finding that 133 of
> the 370 documents have the right number of rows and not one row that matches
> exactly, so the rows get rebuilt from the document's own tables.
> github.com/Nikit117/kynex-harness

**r/AI_Agents**

> KYNEX - a local agent that extracts schema-shaped JSON from documents using
> tools (page text, tables, layout analysis, rendered pages), with page-wise
> enumeration for very long record lists. 92.65 F1 on ExtractBench (370
> documents, official scorer).
> github.com/Nikit117/kynex-harness

**r/Python**

> Built an open-source PDF/document extraction agent in Python: local, tool
> using, schema-guided, with a measured pipeline instead of a prompt. 92.65
> unified value F1 on ExtractBench (370 documents, official scorer), 14.17
> cents per page. The docs walk through the whole path from install to
> benchmark. github.com/Nikit117/kynex-harness

---

## 5. The one that matters most: the maintainer

The ExtractBench board is run by the LlamaIndex team. When pull request #84 is
merged, ask them to mention the entry - a single post from the benchmark's own
maintainers reaches more people than any of the above.

---

## Where the image lives

- `docs/images/kynex-demo.svg` - the demo (repo)
- `docs/images/kynex-logo.svg` - the logo (repo)

To turn the demo into a GIF for a video post, record a real session:

```bash
python -m autonomous_document_harness.tui.app run <doc.pdf> --prompt "$(cat prompt.txt)"
```

and convert it with `asciinema rec demo.cast && agg -o demo.gif demo.cast`
(or `svg-term-cli`).
