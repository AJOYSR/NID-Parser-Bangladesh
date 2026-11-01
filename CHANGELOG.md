# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `LICENSE` — the MIT license the README and the OpenAPI metadata already claimed.
- `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, and `SECURITY.md`.
- Issue forms for bug reports, bad parses, and feature requests, plus a pull
  request template, an issue-template chooser, and `CODEOWNERS`.
- Dependabot configuration for pip and GitHub Actions, with `torch`,
  `torchvision`, and `torchaudio` grouped so they update together.
- An offline unit test suite in `tests/` — 56 cases covering the extraction
  helpers and the per-type dispatch.
- `scripts/generate_sample_document.py`, which draws synthetic NID, BO, and TIN
  images so contributors can test and file issues without touching a real
  identity document.
- `requirements-dev.txt`, a `pyproject.toml` holding the pytest configuration,
  and a `.flake8` file holding the lint configuration.

### Changed

- `test_api.py` moved to `scripts/smoke_test.py` and was rewritten against the
  current API: it now sends the required `type` field, reads the
  `{"type": ..., "details": {...}}` response shape, takes `--url`, `--image`,
  and `--type` arguments, and checks all four validation paths.
- CI now runs lint and tests as separate jobs, covers Python 3.9 through 3.11,
  caches pip downloads, and cancels superseded runs.
- The README documents the actual request and response shapes, including the
  required `type` form field.

### Fixed

- CI test collection, which had failed on every run since the workflow was
  added: pytest collected `test_api.py` as a test module and read its
  `image_path` parameter as a missing fixture.
- `bo_id` was returned with a trailing space, because the trailing `\s*` in the
  BO account pattern absorbed the whitespace after the last digit.

## [1.0.0] - 2025-07-10

### Added

- FastAPI service exposing `POST /extract-nid-info/`, `GET /health`, and `GET /`.
- EasyOCR-backed text extraction with grayscale preprocessing and downscaling to
  a maximum dimension of 800px for speed.
- `DocumentType` enum supporting `NID`, `BO`, and `TIN` documents, with a
  response shaped to the fields that apply to each type.
- Extraction helpers for name (including the `MD:` honorific), date of birth
  across four formats, NID number, BO account number, and TIN number.
- PDF input via `pdf2image`, processed page by page.
- Swagger UI at `/docs` and ReDoc at `/redoc`, with per-endpoint examples.
- CORS middleware, a 10 MB upload limit, and empty-file rejection.
- `setup.sh` and `setup.bat` for environment setup.
- Workarounds for macOS SSL certificate verification and the removal of
  `PIL.Image.ANTIALIAS` in Pillow 10.

[Unreleased]: https://github.com/AJOYSR/NID-Parser-Bangladesh/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/AJOYSR/NID-Parser-Bangladesh/releases/tag/v1.0.0
