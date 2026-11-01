# NID Parser Bangladesh

[![CI](https://github.com/AJOYSR/NID-Parser-Bangladesh/actions/workflows/python-app.yml/badge.svg)](https://github.com/AJOYSR/NID-Parser-Bangladesh/actions/workflows/python-app.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688.svg)](https://fastapi.tiangolo.com/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![Code of Conduct](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)

A FastAPI service that reads Bangladeshi identity documents with OCR and returns
the fields you actually want — name, date of birth, and the document number —
as JSON.

It handles three document types today: **NID** (National ID card), **BO**
(Beneficiary Owner account), and **TIN** (Taxpayer's Identification Number).
Input can be an image or a PDF.

> [!IMPORTANT]
> This project processes national identity data. The default configuration has
> no authentication, no rate limiting, and permissive CORS — it is built for
> local development. Read [SECURITY.md](SECURITY.md) before deploying it
> anywhere, and never commit a real identity document to this repository or to
> an issue.

---

## Table of contents

- [How it works](#how-it-works)
- [Quick start](#quick-start)
- [API reference](#api-reference)
- [Usage examples](#usage-examples)
- [Testing](#testing)
- [Project layout](#project-layout)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [Security](#security)
- [License](#license)

---

## How it works

```
  upload ──► validate ──► preprocess ──► EasyOCR ──► regex extraction ──► JSON
            (type, MIME,   (grayscale,    (English     (per document
             size ≤ 10MB)   max 800px)     model)       type)
```

1. The upload is checked for type and size (10 MB limit).
2. PDFs are split into page images with `pdf2image`; images go straight through.
3. Each image is converted to grayscale and downscaled to a maximum dimension of
   800 px — smaller images are both faster and, in practice, no less accurate
   for card-sized documents.
4. EasyOCR's English model reads the text.
5. Document-type-specific patterns pull out the fields.

The OCR reader is initialised once at startup and reused, so the first request
after boot is not penalised by model loading.

---

## Quick start

**Requirements:** Python 3.9+ (3.11 recommended), ~2 GB free disk space for
PyTorch and the OCR models.

### Automated setup

```bash
git clone https://github.com/AJOYSR/NID-Parser-Bangladesh.git
cd NID-Parser-Bangladesh
./setup.sh
```

On Windows, run `setup.bat` instead.

### Manual setup

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Run it

```bash
uvicorn main:app --reload --port 8000
```

Open <http://localhost:8000/docs> for interactive documentation.

The first startup downloads the EasyOCR English models (~100 MB) into `models/`.
That directory is git-ignored and reused on later runs.

### Try it without a real document

```bash
python scripts/generate_sample_document.py --type NID --output sample_nid.png

curl -X POST http://localhost:8000/extract-nid-info/ \
     -F "type=NID" -F "file=@sample_nid.png"
```

The generator draws a synthetic card with obviously fake values — use it for
testing, bug reports, and demos instead of a real ID.

---

## API reference

### `POST /extract-nid-info/`

Extract structured fields from a document.

**Request** — `multipart/form-data`:

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `type` | string | ✅ | One of `NID`, `BO`, `TIN` |
| `file` | file | ✅ | Image (JPEG, PNG, BMP, TIFF) or PDF, max 10 MB |

**Response** — the `details` object is shaped by `type`:

<details open>
<summary><code>type=NID</code></summary>

```json
{
  "type": "NID",
  "details": {
    "name": "MD: JOHN DOE",
    "date_of_birth": "15/03/1990",
    "nid_number": "1234567890123"
  }
}
```
</details>

<details>
<summary><code>type=BO</code></summary>

```json
{
  "type": "BO",
  "details": {
    "bo_id": "1234 5678 9012 3456"
  }
}
```
</details>

<details>
<summary><code>type=TIN</code></summary>

```json
{
  "type": "TIN",
  "details": {
    "tin_number": "123456789012"
  }
}
```
</details>

A field that could not be read comes back as `null`. The response shape is
stable regardless of what was found, so callers can rely on it.

**Errors:**

| Status | Meaning |
| --- | --- |
| `400` | Unsupported file type, empty file, or file over 10 MB |
| `422` | Missing `type`, or a `type` outside `NID` / `BO` / `TIN` |
| `500` | OCR processing failed |

### `GET /health`

```json
{ "status": "healthy", "timestamp": "2025-07-10T10:30:00.123456" }
```

### `GET /`

Returns the API name, version, and the list of endpoints.

### `GET /docs` and `GET /redoc`

Swagger UI and ReDoc. The OpenAPI schema is at `GET /openapi.json`.

---

## Usage examples

### curl

```bash
curl -X POST "http://localhost:8000/extract-nid-info/" \
     -H "accept: application/json" \
     -F "type=NID" \
     -F "file=@path/to/document.jpg"
```

### Python

```python
import requests

with open("document.jpg", "rb") as handle:
    response = requests.post(
        "http://localhost:8000/extract-nid-info/",
        data={"type": "NID"},
        files={"file": handle},
        timeout=120,
    )

response.raise_for_status()
details = response.json()["details"]

print(details["name"])
print(details["date_of_birth"])
print(details["nid_number"])
```

### JavaScript

```javascript
const form = new FormData();
form.append("type", "NID");
form.append("file", fileInput.files[0]);

const response = await fetch("http://localhost:8000/extract-nid-info/", {
  method: "POST",
  body: form,
});

const { details } = await response.json();
console.log(details.name, details.date_of_birth, details.nid_number);
```

---

## Testing

The automated suite is offline — it exercises the extraction functions directly
and never starts a server or downloads a model:

```bash
pip install -r requirements-dev.txt
pytest
```

For an end-to-end check against a running server:

```bash
uvicorn main:app --port 8000          # terminal 1
python scripts/smoke_test.py          # terminal 2

# ...or with a document
python scripts/smoke_test.py --image sample_nid.png --type NID
```

Linting matches CI:

```bash
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
```

---

## Project layout

```
.
├── main.py                              FastAPI app, OCR pipeline, extraction patterns
├── requirements.txt                     Runtime dependencies
├── requirements-dev.txt                 Test and lint dependencies
├── pyproject.toml                       pytest and flake8 configuration
├── setup.sh / setup.bat                 One-shot environment setup
├── scripts/
│   ├── smoke_test.py                    Manual end-to-end check against a running server
│   └── generate_sample_document.py      Draws synthetic NID / BO / TIN images
├── tests/                               Offline unit tests for the extraction helpers
├── models/                              Downloaded OCR models (git-ignored)
└── logs/                                Log output (git-ignored)
```

---

## Configuration

### Supported input

| Format | Notes |
| --- | --- |
| JPEG, PNG, BMP, TIFF | Read directly |
| PDF | Requires [Poppler](https://poppler.freedesktop.org/) on your `PATH` |

Install Poppler with `brew install poppler` (macOS) or
`apt install poppler-utils` (Debian/Ubuntu). Without it, images still work and
PDFs return a clear error.

### OCR settings

Edit `initialize_reader()` in `main.py`:

| Setting | Default | Notes |
| --- | --- | --- |
| Language | `en` | Bangla script is not currently extracted |
| GPU | `False` | Set `gpu=True` for a CUDA device |
| Model | `english_g2` | A smaller recognition network, chosen for speed |
| Model directory | `./models` | Where downloaded weights are cached |

### Extraction patterns

Each field has its own function in `main.py`, and each returns `None` when it
finds nothing:

| Function | Field |
| --- | --- |
| `extract_name()` | Name, including the `MD:` honorific |
| `extract_date_of_birth()` | Four date formats, numeric and month-name |
| `extract_nid_number()` | Plain and space-separated digit runs |
| `extract_fields_by_type()` | Dispatch per document type, plus BO and TIN numbers |

Adding a new document type is a contained change — see
[CONTRIBUTING.md](CONTRIBUTING.md#adding-a-new-document-type).

---

## Troubleshooting

<details>
<summary><strong>SSL certificate errors on macOS during model download</strong></summary>

`main.py` already falls back to an unverified HTTPS context for the EasyOCR
download. If it still fails, run
`/Applications/Python 3.x/Install Certificates.command`.
</details>

<details>
<summary><strong><code>PIL.Image.ANTIALIAS</code> AttributeError</strong></summary>

Pillow 10 removed `ANTIALIAS`. `main.py` aliases it to
`Image.Resampling.LANCZOS` at import time, so this should not surface — if it
does, check that you are running the current `main.py`.
</details>

<details>
<summary><strong>PDF upload returns "pdf2image is not installed"</strong></summary>

`pdf2image` is a thin wrapper around Poppler. Install the Poppler binaries, not
just the Python package.
</details>

<details>
<summary><strong>A field comes back <code>null</code></strong></summary>

Check the server log for the `Extracted text:` line to see what OCR actually
read. If the text is there but the field was not picked up, that is a parsing
bug worth reporting — open a
[bad parse report](https://github.com/AJOYSR/NID-Parser-Bangladesh/issues/new?template=bad_parse.yml)
with the OCR text **redacted**.

If the text is garbled, it is an image-quality problem: more resolution, less
glare, and a straight-on angle all help.
</details>

<details>
<summary><strong>Slow responses</strong></summary>

OCR on CPU takes a few seconds per document. The reader is cached after startup,
so only the first boot is slow. Set `gpu=True` in `initialize_reader()` if you
have CUDA.
</details>

<details>
<summary><strong>scipy <code>ImportError</code> on recent macOS</strong></summary>

Older scipy wheels fail to load on current macOS releases. Install a newer one
that still satisfies the pinned numpy:
`pip install "numpy==1.24.3" "scipy==1.13.1"`.
</details>

---

## Contributing

Contributions are very welcome — especially reports of documents the parser
reads incorrectly, which are what make the patterns more robust.

- [**Contributing guide**](CONTRIBUTING.md) — setup, style, pull request process
- [**Code of Conduct**](CODE_OF_CONDUCT.md) — Contributor Covenant 2.1
- [**Report a bad parse**](https://github.com/AJOYSR/NID-Parser-Bangladesh/issues/new?template=bad_parse.yml)
- [**Report a bug**](https://github.com/AJOYSR/NID-Parser-Bangladesh/issues/new?template=bug_report.yml)
- [**Request a feature**](https://github.com/AJOYSR/NID-Parser-Bangladesh/issues/new?template=feature_request.yml)
- [**Changelog**](CHANGELOG.md)

**One rule above all the others:** never commit, attach, or paste a real
identity document. Redact it, or generate a synthetic one with
`scripts/generate_sample_document.py`.

---

## Security

Found a vulnerability? Please report it privately — see [SECURITY.md](SECURITY.md).
Do not open a public issue.

That file also covers what to change before running this service anywhere other
than your own machine: CORS, authentication, rate limiting, bind address, and
log retention.

---

## License

[MIT](LICENSE) © Ajoy Sarker

---

## Author

**Ajoy Sarker**
[GitHub @AJOYSR](https://github.com/AJOYSR) · [LinkedIn @ajoysrju](https://www.linkedin.com/in/ajoysrju)

Built with [FastAPI](https://fastapi.tiangolo.com/) and
[EasyOCR](https://github.com/JaidedAI/EasyOCR).

If this project is useful to you, a ⭐ helps others find it.
