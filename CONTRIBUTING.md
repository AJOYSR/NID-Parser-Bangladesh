# Contributing to NID Parser Bangladesh

First off — thank you for taking the time to contribute. This project is a small,
focused OCR service, and it gets better every time someone reports a bad parse,
adds a document pattern, or tightens up the code.

This document explains how to get set up, what we expect in a pull request, and
how decisions get made.

---

## Table of contents

- [Code of conduct](#code-of-conduct)
- [Ways to contribute](#ways-to-contribute)
- [Working with identity documents](#working-with-identity-documents)
- [Development setup](#development-setup)
- [Running the service](#running-the-service)
- [Running the tests](#running-the-tests)
- [Coding style](#coding-style)
- [Commit messages](#commit-messages)
- [Pull request process](#pull-request-process)
- [Adding a new document type](#adding-a-new-document-type)
- [Reporting a bad parse](#reporting-a-bad-parse)
- [Getting help](#getting-help)

---

## Code of conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). By
participating, you agree to uphold it. Report unacceptable behaviour to
**ajoysr.official@gmail.com**.

---

## Ways to contribute

You do not need to write code to help:

| Contribution | Where it goes |
| --- | --- |
| A document the parser reads incorrectly | [Bad parse report](https://github.com/AJOYSR/NID-Parser-Bangladesh/issues/new?template=bad_parse.yml) |
| A bug in the API or setup scripts | [Bug report](https://github.com/AJOYSR/NID-Parser-Bangladesh/issues/new?template=bug_report.yml) |
| An idea for a new feature or document type | [Feature request](https://github.com/AJOYSR/NID-Parser-Bangladesh/issues/new?template=feature_request.yml) |
| A question about usage | [Discussions](https://github.com/AJOYSR/NID-Parser-Bangladesh/discussions) |
| A security problem | **Do not open an issue** — see [SECURITY.md](SECURITY.md) |
| Docs, typos, clearer examples | A pull request, straight away |

Small fixes (typos, broken links, an obviously wrong regex) do not need an issue
first. Anything that changes the API shape, adds a dependency, or reworks how
extraction happens is worth discussing in an issue before you write the code.

---

## Working with identity documents

This is the one rule that matters more than any style guide:

> **Never commit, attach, or paste a real identity document — yours or anyone
> else's.** No real NID cards, BO account statements, TIN certificates, scans,
> photos, or the text OCR produced from them.

Real NID numbers, names, and dates of birth are personal data. Once something is
pushed to a public repository it is in the git history and in every fork, and it
cannot be meaningfully deleted.

When you need to show a document to explain a problem:

- **Redact it.** Black out every digit of the NID/BO/TIN number, the photo, the
  signature, and the address. Leave only the layout and the field labels visible.
- **Or synthesise it.** Replace the values with obvious fakes — `MD: JOHN DOE`,
  `01/01/1990`, `1234567890123`. The parser cares about shape, not truth.
- **Or generate one.** `scripts/generate_sample_document.py` draws a synthetic
  NID, BO, or TIN image with obviously fake values:

  ```bash
  python scripts/generate_sample_document.py --type NID --output sample_nid.png
  ```

- **Or describe it.** "A smart NID where the name wraps onto two lines" is often
  enough for a maintainer to reproduce the problem.

Test fixtures in this repository must be synthetic. A pull request that adds a
real document will be closed and the branch asked to be force-pushed clean.

---

## Development setup

**Requirements:** Python 3.9 or newer (3.11 recommended), `pip`, and roughly
2 GB of free disk space — PyTorch and the EasyOCR models are large.

```bash
git clone https://github.com/AJOYSR/NID-Parser-Bangladesh.git
cd NID-Parser-Bangladesh
```

Then either run the setup script:

```bash
./setup.sh          # macOS / Linux
```

```bat
setup.bat           :: Windows
```

Or do it by hand:

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

PDF support additionally needs [Poppler](https://poppler.freedesktop.org/) on
your `PATH` (`brew install poppler`, `apt install poppler-utils`). Without it the
API still handles images and returns a clear error for PDFs.

The first run downloads the EasyOCR English models (~100 MB) into `models/`.
Both `models/` and `venv/` are git-ignored.

---

## Running the service

```bash
uvicorn main:app --reload --port 8000
```

Interactive docs are at <http://localhost:8000/docs>.

A manual smoke test against a running server lives in `scripts/smoke_test.py`:

```bash
python scripts/smoke_test.py
```

It exercises every endpoint and reports on each one. It is a script, not part of
the automated suite.

---

## Running the tests

```bash
pytest
```

The suite in `tests/` is pure and offline — it tests the extraction functions
directly and never starts a server, downloads a model, or touches the network.
Keep it that way: a test that needs OCR inference belongs behind a marker, not in
the default run.

Linting matches what CI enforces:

```bash
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
```

That selection catches syntax errors and undefined names — the things that break
a build. A second, non-blocking pass reports style warnings at a 127-character
line limit.

---

## Coding style

- Follow [PEP 8](https://peps.python.org/pep-0008/); 127-character lines are fine.
- Type-hint new functions, and give each one a docstring saying what it returns
  when it finds nothing.
- Extraction helpers return `Optional[str]` — `None` for "not found". Let the
  endpoint layer decide how that is presented to the caller.
- Keep regexes readable. A commented pattern with an example of the text it
  matches is worth far more than a clever one-liner:

  ```python
  # Matches "BO Account Number: 1234 5678 9012 3456" (16 digits, any spacing)
  bo_pattern = r'BO\s*Account\s*Number\s*[:\-]?\s*((?:\d\s*){16})'
  ```

- Log at `INFO` for lifecycle events and `ERROR` for failures. **Never log a full
  NID number, name, or date of birth** — truncate, as the existing OCR logging does.
- New dependencies need a reason in the pull request description. This project
  already pulls in PyTorch; we are careful about adding more weight.

---

## Commit messages

Write a short imperative subject line under ~72 characters:

```
Add TIN certificate number extraction
Fix BO account regex dropping the last digit
Document Poppler requirement for PDF input
```

[Conventional Commits](https://www.conventionalcommits.org/) prefixes (`feat:`,
`fix:`, `docs:`) are welcome but not required. What matters is that the subject
says what changed, and the body says why if it is not obvious.

---

## Pull request process

1. **Fork** the repository and create a branch off `main`:
   `git checkout -b fix/bo-account-regex`
2. **Make your change.** Keep it focused — one logical change per pull request.
3. **Add a test** for any behaviour change in the extraction logic. A bad-parse
   fix should come with a case in `tests/` using synthetic text that failed before.
4. **Run `pytest` and `flake8` locally.** CI runs both on every pull request.
5. **Update the docs** — the README, docstrings, and `CHANGELOG.md` under
   `## [Unreleased]`.
6. **Open the pull request** against `main`, fill in the template, and link the
   issue it closes (`Closes #12`).
7. **Respond to review.** Push follow-up commits rather than force-pushing, so
   reviewers can read the diff since their last look. Squashing happens on merge.

A pull request is ready to merge when CI is green, a maintainer has approved it,
and any discussion threads are resolved.

---

## Adding a new document type

The parser is organised so that a new Bangladeshi document type is a small,
contained change. To add one:

1. Add the value to the `DocumentType` enum in `main.py`.
2. Write an `extract_<field>()` helper that takes the OCR text and returns
   `Optional[str]`.
3. Wire it into `extract_fields_by_type()` under a new branch.
4. Shape the response in `extract_nid_info()` so callers get only the fields that
   apply to that type.
5. Add tests in `tests/` covering a clean match, a messy-OCR match, and a miss.
6. Document the type and its response shape in the README.

Bring a few examples of how the document's text actually comes out of OCR —
redacted or synthetic — in the pull request description. OCR output is rarely as
tidy as the printed document, and that detail is what makes a pattern hold up.

---

## Reporting a bad parse

This is the most useful kind of issue for this project. A good report has:

- The document type (`NID`, `BO`, `TIN`) and whether it is a smart card, an old
  laminated card, a photo, a scan, or a PDF.
- What you got back from the API, and what you expected.
- **The OCR text, redacted** — the `extracted_text` field, or the log line, with
  real identifiers replaced by `X`s. Keep the layout and labels intact.
- Image conditions, if relevant: glare, angle, resolution, cropping.

Use the [bad parse template](https://github.com/AJOYSR/NID-Parser-Bangladesh/issues/new?template=bad_parse.yml),
which asks for exactly this.

---

## Getting help

- **Questions and ideas:** [GitHub Discussions](https://github.com/AJOYSR/NID-Parser-Bangladesh/discussions)
- **Bugs and bad parses:** [Issues](https://github.com/AJOYSR/NID-Parser-Bangladesh/issues)
- **Security:** [SECURITY.md](SECURITY.md) — privately, never in an issue
- **Maintainer:** Ajoy Sarker — [@AJOYSR](https://github.com/AJOYSR)

By contributing, you agree that your contributions will be licensed under the
[MIT License](LICENSE) that covers this project.
