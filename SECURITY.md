# Security Policy

## Supported Versions

This project is maintained on a rolling basis. Security fixes land on `main` and
are released as a new patch version. Only the latest release is supported.

| Version | Supported |
| ------- | --------- |
| 1.0.x   | ✅ |
| < 1.0   | ❌ |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues,
discussions, or pull requests.**

Instead, report them one of two ways:

1. **GitHub Security Advisories** (preferred) — open a
   [private advisory](https://github.com/AJOYSR/NID-Parser-Bangladesh/security/advisories/new).
   This keeps the report private until a fix is published.
2. **Email** — **ajoysr.official@gmail.com** with `[SECURITY]` in the subject.

Please include as much of the following as you can:

- The type of issue (for example: path traversal, resource exhaustion,
  deserialisation, data leakage through logs).
- Full paths of the source files involved.
- The location of the affected code — a tag, branch, or commit SHA.
- Any special configuration needed to reproduce it.
- Step-by-step instructions to reproduce.
- Proof-of-concept or exploit code, if you have it.
- The impact as you see it, including how an attacker might use it.

**Do not include real identity documents or real personal data in a report.**
Use synthetic or redacted examples — see
[CONTRIBUTING.md](CONTRIBUTING.md#working-with-identity-documents).

### What to expect

| Stage | Target |
| --- | --- |
| Acknowledgement of your report | within 72 hours |
| Initial assessment and severity | within 7 days |
| Fix or mitigation plan | within 30 days for high severity |
| Public disclosure | after a fix ships, coordinated with you |

If you would like credit in the advisory and release notes, say so in your
report and tell us the name or handle to use. This project has no bug bounty.

## Scope

In scope:

- The FastAPI application in `main.py` and anything under `scripts/` and `tests/`.
- The setup scripts (`setup.sh`, `setup.bat`) and the CI workflows.
- Dependency vulnerabilities that are reachable through this project's code.

Out of scope:

- Vulnerabilities in EasyOCR, PyTorch, OpenCV, or other upstream dependencies
  that are not reachable from this code — report those upstream.
- Results of running the service exposed to the public internet without the
  hardening described below. The default configuration is for development.
- OCR inaccuracy. A wrong or missed field is a
  [bad parse](https://github.com/AJOYSR/NID-Parser-Bangladesh/issues/new?template=bad_parse.yml),
  not a vulnerability.

## Security considerations when deploying

This service processes national identity documents. If you run it anywhere other
than your own machine, please read this section.

**The default configuration is not production-ready.** Out of the box the
service:

- **Allows every origin.** `CORSMiddleware` is configured with
  `allow_origins=["*"]` and `allow_credentials=True`. Replace the wildcard with
  your own origins before deploying.
- **Has no authentication.** Any client that can reach the port can submit
  documents. Put it behind an authenticating gateway, or add auth yourself.
- **Has no rate limiting.** OCR is CPU-expensive; an unauthenticated caller can
  exhaust your capacity with large images. Rate-limit at the proxy.
- **Binds to `0.0.0.0`.** `python main.py` listens on every interface. Bind to
  `127.0.0.1` unless you intend otherwise.
- **Returns `extracted_text`** from the OCR helper, which contains the full raw
  text of the document. Do not expose that field to callers who should only see
  the parsed fields.

**Handling the documents themselves:**

- Uploads are written to temporary files for OCR and deleted afterwards. If the
  process crashes mid-request, a temporary file may survive — make sure your
  temp directory is on encrypted storage and is cleaned on boot.
- Nothing is persisted by design. The service keeps no database and writes no
  uploads to disk beyond those temporary files. Verify this still holds if you
  fork and add storage.
- Application logs include a 200-character prefix of the OCR text, which can
  contain names and partial numbers. Review your log retention and access
  controls, and consider turning that line down to `DEBUG` in production.
- Depending on where you operate, processing national identity data may carry
  legal obligations around consent, retention, and breach notification. That is
  your responsibility as the operator, not something this software handles for you.

## Known hardening gaps

These are understood and documented rather than hidden. Pull requests welcome:

- No maximum image dimension check — only a 10 MB file-size limit — so a
  decompression-heavy image can still consume significant memory.
- A multi-page PDF is processed page by page with no page-count cap.
- The MIME type is taken from the client's `Content-Type` header rather than
  from the file contents.
