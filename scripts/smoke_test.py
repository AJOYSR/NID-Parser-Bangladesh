#!/usr/bin/env python3
"""Manual smoke test for the NID Parser API.

Exercises every endpoint against a running server and prints what it found.
This is an interactive helper, not part of the automated suite — the offline
unit tests live in ``tests/`` and are run with ``pytest``.

Usage::

    uvicorn main:app --port 8000        # in one terminal
    python scripts/smoke_test.py        # in another

    python scripts/smoke_test.py --url http://localhost:9000
    python scripts/smoke_test.py --image sample_nid.jpg --type NID

Never point this at a real identity document you intend to share. Any output
you paste into an issue must be redacted first.
"""

import argparse
import os
import time
from typing import Any, Dict, Optional

import requests

DEFAULT_BASE_URL = "http://localhost:8000"

ENDPOINTS = {
    "root": "/",
    "health": "/health",
    "extract": "/extract-nid-info/",
}

DOCUMENT_TYPES = ("NID", "BO", "TIN")

# Fields the API returns for each document type.
EXPECTED_FIELDS = {
    "NID": ("name", "date_of_birth", "nid_number"),
    "BO": ("bo_id",),
    "TIN": ("tin_number",),
}


def print_separator(title: str) -> None:
    """Print a formatted separator with a title."""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def check_root(base_url: str) -> Dict[str, Any]:
    """Check the root endpoint."""
    print_separator("Root Endpoint")

    try:
        response = requests.get(f"{base_url}{ENDPOINTS['root']}", timeout=10)
        print(f"Status Code: {response.status_code}")

        if response.status_code != 200:
            print(f"❌ Root endpoint failed: {response.text}")
            return {}

        data = response.json()
        print("✅ Root endpoint working correctly")
        print(f"API Message: {data['message']}")
        print(f"API Version: {data['version']}")
        print("Available Endpoints:")
        for endpoint, path in data["endpoints"].items():
            print(f"  - {endpoint}: {path}")
        return data

    except requests.exceptions.RequestException as exc:
        print(f"❌ Request failed: {exc}")
        return {}


def check_health(base_url: str) -> Dict[str, Any]:
    """Check the health endpoint."""
    print_separator("Health Endpoint")

    try:
        response = requests.get(f"{base_url}{ENDPOINTS['health']}", timeout=10)
        print(f"Status Code: {response.status_code}")

        if response.status_code != 200:
            print(f"❌ Health endpoint failed: {response.text}")
            return {}

        data = response.json()
        print("✅ Health endpoint working correctly")
        print(f"Status: {data['status']}")
        print(f"Timestamp: {data['timestamp']}")
        return data

    except requests.exceptions.RequestException as exc:
        print(f"❌ Request failed: {exc}")
        return {}


def check_documentation(base_url: str) -> None:
    """Check that the generated documentation endpoints are served."""
    print_separator("API Documentation")

    for name, endpoint in (("Swagger UI", "/docs"), ("ReDoc", "/redoc"), ("OpenAPI JSON", "/openapi.json")):
        try:
            response = requests.get(f"{base_url}{endpoint}", timeout=10)
            status = "✅ Available" if response.status_code == 200 else "❌ Not available"
        except requests.exceptions.RequestException:
            status = "❌ Not available"
        print(f"{name}: {status}")


def check_input_validation(base_url: str) -> None:
    """Check that malformed uploads are rejected with the documented errors."""
    print_separator("Input Validation")
    url = f"{base_url}{ENDPOINTS['extract']}"

    cases = [
        (
            "Missing document type",
            {"files": {"file": ("test.jpg", b"not-really-an-image", "image/jpeg")}},
            422,
        ),
        (
            "Unsupported file type",
            {
                "data": {"type": "NID"},
                "files": {"file": ("notes.txt", b"This is not an image", "text/plain")},
            },
            400,
        ),
        (
            "Empty file",
            {
                "data": {"type": "NID"},
                "files": {"file": ("empty.jpg", b"", "image/jpeg")},
            },
            400,
        ),
        (
            "Unknown document type",
            {
                "data": {"type": "PASSPORT"},
                "files": {"file": ("test.jpg", b"not-really-an-image", "image/jpeg")},
            },
            422,
        ),
    ]

    for label, kwargs, expected_status in cases:
        print(f"\n--- {label} ---")
        try:
            response = requests.post(url, timeout=30, **kwargs)
            if response.status_code == expected_status:
                print(f"✅ Rejected with {response.status_code} as expected")
            else:
                print(f"❌ Expected {expected_status}, got {response.status_code}")
            print(f"   Response: {response.text[:200]}")
        except requests.exceptions.RequestException as exc:
            print(f"❌ Request failed: {exc}")


def check_extraction(base_url: str, file_path: str, doc_type: str) -> Dict[str, Any]:
    """Run a real extraction against a document file."""
    print_separator(f"Extraction ({doc_type})")
    print(f"File: {file_path}")

    content_type = "application/pdf" if file_path.lower().endswith(".pdf") else "image/jpeg"

    try:
        with open(file_path, "rb") as handle:
            response = requests.post(
                f"{base_url}{ENDPOINTS['extract']}",
                data={"type": doc_type},
                files={"file": (os.path.basename(file_path), handle, content_type)},
                timeout=120,
            )

        print(f"Status Code: {response.status_code}")

        if response.status_code != 200:
            print(f"❌ Extraction failed: {response.text}")
            return {}

        data = response.json()
        details = data.get("details", {})
        print("✅ Extraction succeeded")
        print(f"Document type: {data.get('type')}")
        for field in EXPECTED_FIELDS.get(doc_type, ()):
            value = details.get(field)
            print(f"  {field}: {value if value is not None else '— not detected —'}")

        missing = [f for f in EXPECTED_FIELDS.get(doc_type, ()) if details.get(f) is None]
        if missing:
            print(f"\n⚠️  Not detected: {', '.join(missing)}")
            print("   If these are present on the document, that is a bad parse worth reporting.")
            print("   Redact the values before opening an issue.")
        return data

    except OSError as exc:
        print(f"❌ Could not read file: {exc}")
        return {}
    except requests.exceptions.RequestException as exc:
        print(f"❌ Request failed: {exc}")
        return {}


def run_performance_check(base_url: str, file_path: str, doc_type: str, runs: int = 3) -> None:
    """Time repeated extractions to see the effect of the cached OCR reader."""
    print_separator("Performance")

    durations = []
    for index in range(1, runs + 1):
        print(f"Run {index}/{runs}...")
        start = time.time()
        try:
            with open(file_path, "rb") as handle:
                response = requests.post(
                    f"{base_url}{ENDPOINTS['extract']}",
                    data={"type": doc_type},
                    files={"file": (os.path.basename(file_path), handle, "image/jpeg")},
                    timeout=120,
                )
            if response.status_code == 200:
                duration = time.time() - start
                durations.append(duration)
                print(f"  ✅ Completed in {duration:.2f}s")
            else:
                print(f"  ❌ Failed with status {response.status_code}")
        except (OSError, requests.exceptions.RequestException) as exc:
            print(f"  ❌ {exc}")

    if durations:
        print("\nSummary:")
        print(f"  Average: {sum(durations) / len(durations):.2f}s")
        print(f"  Min:     {min(durations):.2f}s")
        print(f"  Max:     {max(durations):.2f}s")


def find_sample_image() -> Optional[str]:
    """Look for a sample document in the working directory."""
    for candidate in ("test_nid.jpg", "test_nid.png", "sample.jpg", "sample.png", "sample.pdf"):
        if os.path.exists(candidate):
            return candidate
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default=DEFAULT_BASE_URL, help=f"Base URL of the API (default: {DEFAULT_BASE_URL})")
    parser.add_argument("--image", help="Path to a document image or PDF to run a real extraction against")
    parser.add_argument("--type", choices=DOCUMENT_TYPES, default="NID", help="Document type to send (default: NID)")
    parser.add_argument("--skip-performance", action="store_true", help="Skip the repeated-request timing check")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_url = args.url.rstrip("/")

    print("🚀 NID Parser API Smoke Test")
    print(f"Target: {base_url}")

    try:
        response = requests.get(f"{base_url}{ENDPOINTS['health']}", timeout=5)
        if response.status_code != 200:
            print(f"❌ API server responded with {response.status_code} on /health")
            return 1
    except requests.exceptions.RequestException:
        print(f"❌ Cannot reach the API at {base_url}")
        print("   Start the server with: uvicorn main:app --port 8000")
        return 1

    check_root(base_url)
    check_health(base_url)
    check_documentation(base_url)
    check_input_validation(base_url)

    sample = args.image or find_sample_image()

    if sample and os.path.exists(sample):
        print(f"\n📸 Using document: {sample}")
        check_extraction(base_url, sample, args.type)
        if not args.skip_performance and not sample.lower().endswith(".pdf"):
            run_performance_check(base_url, sample, args.type)
    else:
        if args.image:
            print(f"\n❌ File not found: {args.image}")
        else:
            print("\n📸 No sample document found — validation checks only.")
            print("   Pass one with --image to exercise real extraction.")

    print_separator("Summary")
    print("✅ Endpoints reachable")
    print("✅ Input validation checked")
    print("✅ Documentation endpoints checked")
    if sample and os.path.exists(sample):
        print("✅ Extraction checked against a real document")
    else:
        print("⚠️  Extraction not checked — no document provided")

    print(f"\n📖 Interactive documentation: {base_url}/docs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
