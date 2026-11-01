"""Tests for name extraction.

Every name here is synthetic. Real names from real documents must never appear
in this repository.
"""

import pytest

from main import extract_name


def test_md_honorific_is_preserved():
    text = "MD: JOHN DOE Data of Birth 15/03/1990"
    assert extract_name(text) == "MD: JOHN DOE"


def test_md_honorific_stops_before_date_label():
    text = "MD: JOHN DOE Date of Birth 15/03/1990 ID NO 1234567890123"
    assert extract_name(text) == "MD: JOHN DOE"


def test_trailing_single_letter_artifact_is_dropped():
    # OCR often picks up a stray initial from the card border.
    text = "MD: JOHN DOE D Data of Birth 15/03/1990"
    assert extract_name(text) == "MD: JOHN DOE"


def test_name_label_form():
    assert extract_name("Name: JOHN DOE") == "JOHN DOE"


def test_name_label_keeps_md_prefix():
    assert extract_name("Name: MD. JOHN DOE") == "MD. JOHN DOE"


def test_name_inferred_from_date_label_context():
    # No explicit "Name:" label — the name sits directly before the DOB label.
    assert extract_name("JOHN DOE Data of Birth 15/03/1990") == "JOHN DOE"


@pytest.mark.parametrize("noise_word", ["SCREENSHOT", "CHROME", "EXTENSION", "RECORDER"])
def test_browser_chrome_words_are_not_returned_as_names(noise_word):
    # Screenshots of a document in a browser pull UI text into the OCR output.
    result = extract_name(noise_word)
    assert result is None or noise_word not in result


@pytest.mark.parametrize(
    "text",
    [
        "",
        "no uppercase words here at all",
        "1234567890123",
    ],
)
def test_returns_none_when_no_name_present(text):
    assert extract_name(text) is None
