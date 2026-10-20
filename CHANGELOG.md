# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-10-20

The first public release: the harness that was measured on ExtractBench.

### Added

- the interactive TUI (`python -m autonomous_document_harness.tui`) and the
  scriptable entry points (`run`, `ask`)
- the agent loop with the tool set: page text, tables, layout analysis, vision
  and search
- document staging for PDF, DOCX, XLSX and images, with an OCR fallback for
  pages that have no text layer
- the extraction pipeline and its five rules: schema-shaped answers, page-wise
  enumeration of long record lists, non-record pruning, redaction-marker
  normalisation and structured rows read from the document's own tables
- the ExtractBench provider (`kynex_gpt_5_4`), a local subprocess provider
- token accounting per document
