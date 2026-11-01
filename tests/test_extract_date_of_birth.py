"""Tests for date-of-birth extraction.

Every sample here is synthetic. Real dates of birth from real documents must
never appear in this repository.
"""

import pytest

from main import extract_date_of_birth


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Date of Birth 15/03/1990", "15/03/1990"),
        ("Date of Birth 15-03-1990", "15-03-1990"),
        ("Date of Birth 15.03.1990", "15.03.1990"),
        ("Date of Birth 1 1 1990 01/01/1990", "01/01/1990"),
    ],
)
def test_numeric_day_first_formats(text, expected):
    assert extract_date_of_birth(text) == expected


def test_year_first_format():
    assert extract_date_of_birth("DOB 1990/03/15") == "1990/03/15"


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Data of Birth 15 Mar 1990", "15 Mar 1990"),
        ("Data of Birth 15 March 1990", "15 March 1990"),
        ("Data of Birth 15 JAN 1990", "15 JAN 1990"),
    ],
)
def test_day_month_name_year(text, expected):
    assert extract_date_of_birth(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Birth Mar 15, 1990", "Mar 15, 1990"),
        ("Birth March 15 1990", "March 15 1990"),
    ],
)
def test_month_name_day_year(text, expected):
    assert extract_date_of_birth(text) == expected


def test_surrounding_ocr_noise_is_ignored():
    noisy = "GovernmenI 0f the People's Republic MD: JOHN DOE Data of Birth 15/03/1990 ID NO 1234567890123"
    assert extract_date_of_birth(noisy) == "15/03/1990"


@pytest.mark.parametrize(
    "text",
    [
        "",
        "No date anywhere in this text",
        "ID NO 1234567890123",
    ],
)
def test_returns_none_when_no_date_present(text):
    assert extract_date_of_birth(text) is None
