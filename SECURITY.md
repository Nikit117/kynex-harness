# Security policy

## Supported versions

| version | supported |
|---|---|
| 0.1.x | yes |

## Reporting a vulnerability

Please report security issues privately through the repository's security
advisory page rather than in a public issue. Include a description, the
affected version and, if possible, a reproduction.

You can expect an acknowledgement within a few days and an assessment within
two weeks.

## What the harness does with your data

- **Documents stay on your machine.** The harness stages files locally and
  sends page text, tables and images to the model endpoint you configure.
- **Credentials come from the environment.** They are read from
  `AZURE_OPENAI_*` and `AZURE_DI_*` (or a `.env` file you keep out of git).
  The repository ignores `.env*`, and no credential is committed.
- **The model endpoint receives document content.** Use it with documents you
  are allowed to send to that endpoint.
