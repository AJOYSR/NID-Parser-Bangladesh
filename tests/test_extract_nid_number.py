"""Tests for NID number extraction.

Every number here is synthetic and does not correspond to a real NID.
"""

import pytest

from main import extract_nid_number


def test_plain_thirteen_digit_number():
    assert extract_nid_number("ID NO: 1234567890123") == "1234567890123"


def test_plain_seventeen_digit_number():
    assert extract_nid_number("NID 19901234567890123") == "19901234567890123"


def test_space_separated_groups():
    assert extract_nid_number("ID NO 600 458 9963 0012") == "600 458 9963 0012"


def test_three_space_separated_groups():
    assert extract_nid_number("600 458 9963") == "600 458 9963"


def test_number_is_found_among_other_text():
    text = "NATIONAL ID CARD MD: JOHN DOE Date of Birth 15/03/1990 ID NO: 1234567890123"
    assert extract_nid_number(text) == "1234567890123"


@pytest.mark.parametrize(
    "text",
    [
        "",
        "No digits at all here",
        "Only 123 short 45 runs",
    ],
)
def test_returns_none_when_no_number_present(text):
    assert extract_nid_number(text) is None


def test_result_has_no_surrounding_whitespace():
    result = extract_nid_number("ID NO  600 458 9963  ")
    assert result is not None
    assert result == result.strip()
