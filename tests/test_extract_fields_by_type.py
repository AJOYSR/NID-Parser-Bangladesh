"""Tests for per-document-type field dispatch.

Every identifier here is synthetic.
"""

import pytest

from main import DocumentType, extract_fields_by_type

ALL_FIELDS = {"name", "date_of_birth", "nid_number", "bo_id", "tin_number"}


@pytest.mark.parametrize("doc_type", ["NID", "BO", "TIN"])
def test_every_type_returns_the_full_field_set(doc_type):
    # Fields that do not apply come back as None rather than being absent, so
    # callers can rely on the shape.
    assert set(extract_fields_by_type(doc_type, "")) == ALL_FIELDS


def test_document_type_enum_values_match_dispatch_keys():
    assert {t.value for t in DocumentType} == {"NID", "BO", "TIN"}


class TestNID:
    text = "NATIONAL ID CARD MD: JOHN DOE Data of Birth 15/03/1990 ID NO: 1234567890123"

    def test_extracts_name(self):
        assert extract_fields_by_type("NID", self.text)["name"] == "MD: JOHN DOE"

    def test_extracts_date_of_birth(self):
        assert extract_fields_by_type("NID", self.text)["date_of_birth"] == "15/03/1990"

    def test_extracts_nid_number(self):
        assert extract_fields_by_type("NID", self.text)["nid_number"] == "1234567890123"

    def test_leaves_other_types_fields_empty(self):
        details = extract_fields_by_type("NID", self.text)
        assert details["bo_id"] is None
        assert details["tin_number"] is None


class TestBO:
    def test_extracts_sixteen_digit_account_number(self):
        details = extract_fields_by_type("BO", "BO Account Number: 1234567890123456")
        assert details["bo_id"] is not None
        assert details["bo_id"].replace(" ", "") == "1234567890123456"

    def test_accepts_spaced_digits(self):
        details = extract_fields_by_type("BO", "BO Account Number 1234 5678 9012 3456")
        assert details["bo_id"] is not None
        assert details["bo_id"].replace(" ", "") == "1234567890123456"

    def test_label_match_is_case_insensitive(self):
        details = extract_fields_by_type("BO", "bo account number: 1234567890123456")
        assert details["bo_id"] is not None

    def test_returns_none_without_the_label(self):
        assert extract_fields_by_type("BO", "1234567890123456")["bo_id"] is None

    def test_does_not_extract_nid_fields(self):
        details = extract_fields_by_type("BO", "MD: JOHN DOE BO Account Number: 1234567890123456")
        assert details["name"] is None
        assert details["nid_number"] is None

    def test_result_has_no_surrounding_whitespace(self):
        # The pattern's trailing \s* used to absorb the space after the last digit.
        details = extract_fields_by_type("BO", "BO Account Number 1234 5678 9012 3456 Holder: JOHN DOE")
        assert details["bo_id"] == details["bo_id"].strip()


class TestTIN:
    @pytest.mark.parametrize("digits", ["123456789", "123456789012"])
    def test_extracts_nine_to_twelve_digit_numbers(self, digits):
        assert extract_fields_by_type("TIN", f"TIN: {digits}")["tin_number"] == digits

    def test_label_match_is_case_insensitive(self):
        assert extract_fields_by_type("TIN", "tin - 123456789012")["tin_number"] == "123456789012"

    def test_returns_none_without_the_label(self):
        assert extract_fields_by_type("TIN", "123456789012")["tin_number"] is None

    def test_does_not_extract_nid_fields(self):
        details = extract_fields_by_type("TIN", "MD: JOHN DOE TIN: 123456789012")
        assert details["name"] is None
        assert details["date_of_birth"] is None


def test_unknown_type_returns_empty_fields():
    details = extract_fields_by_type("PASSPORT", "MD: JOHN DOE 1234567890123")
    assert all(value is None for value in details.values())
