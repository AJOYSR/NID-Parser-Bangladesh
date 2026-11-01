"""Regressions found by running real documents through the parser.

Each case here is the OCR text that a real document produced, with the personal
data replaced by synthetic values. The layout, the stray characters, and the
misread labels are reproduced exactly, because those are what broke the
patterns.
"""

import pytest

from main import extract_date_of_birth, extract_name


class TestTrailingOCRDigit:
    """OCR appends a stray digit to the year on otherwise legible cards.

    The year group used to require a word boundary, so "23 Jan 19941" matched
    nothing at all and a readable date of birth came back as None.
    """

    def test_trailing_digit_does_not_discard_the_date(self):
        assert extract_date_of_birth("Date ol Birth: 23 Jan 19941") == "23 Jan 1994"

    def test_clean_year_is_unaffected(self):
        assert extract_date_of_birth("Date of Birth: 23 Jan 1994") == "23 Jan 1994"

    def test_trailing_digit_on_numeric_format(self):
        assert extract_date_of_birth("Date of Birth 15/03/19901") == "15/03/1990"


class TestYearPlausibility:
    """A truncated year must not be reported as a date."""

    def test_three_digit_year_is_rejected(self):
        # Reporting "23 Jan 199" as a date of birth is worse than reporting nothing.
        assert extract_date_of_birth("Date 0( Dlrih; 23 Jan 199 |") is None

    def test_future_year_is_rejected(self):
        # 02/01/2028 is an expiry date, not a date of birth.
        assert extract_date_of_birth("ELP 02/01/2028 SAMPLE CITIZEN") is None

    def test_month_above_twelve_is_rejected(self):
        # A US card prints MM/DD/YYYY; read day-first, month 14 is impossible.
        assert extract_date_of_birth("02/14/1978") is None

    def test_two_digit_year_is_still_accepted(self):
        assert extract_date_of_birth("Date of Birth 15/03/90") == "15/03/90"

    @pytest.mark.parametrize("year", ["1900", "1990", "2000"])
    def test_plausible_years_are_accepted(self, year):
        assert extract_date_of_birth(f"Date of Birth 15/03/{year}") == f"15/03/{year}"


class TestDateOfBirthLabelAnchoring:
    """Documents carry other dates; the first one in the text is not always right."""

    def test_issue_date_before_the_label_is_skipped(self):
        # A birth certificate prints its registration date above the date of birth.
        text = "Birth Registration Number Issuance 23/09/2023 1234567890 Date of Birth 28/03/1996 Sex"
        assert extract_date_of_birth(text) == "28/03/1996"

    @pytest.mark.parametrize(
        "label",
        ["Date of Birth", "Data of Birth", "Date ol Birth", "Date 0f Birth", "DOB", "D.O.B."],
    )
    def test_label_variants_are_recognised(self, label):
        text = f"Issued 01/01/2020 {label} 15/03/1990"
        assert extract_date_of_birth(text) == "15/03/1990"

    def test_falls_back_to_first_date_when_label_is_unreadable(self):
        # "Date 0* Bn" is too mangled to match, so reading order decides.
        assert extract_date_of_birth("Date 0* Bn 12 May 1975 On 247") == "12 May 1975"

    def test_distant_date_is_not_claimed_by_the_label(self):
        # Nothing plausible follows the label, and there is no other date.
        assert extract_date_of_birth("Data of Birth: Jmnh KO 2360511497") is None


class TestMDHonorific:
    """The honorific is printed "MD." but OCR often reads it as "MD:"."""

    def test_period_form_is_recognised(self):
        assert extract_name("Name MD. JOHN DOE Ret Grrrt") == "MD. JOHN DOE"

    def test_colon_form_is_recognised(self):
        assert extract_name("Eer 0p MD: JOHN DOE 4 G7 Data of Birth") == "MD: JOHN DOE"

    def test_separator_is_preserved_as_read(self):
        assert extract_name("MD. JOHN DOE Date of Birth").startswith("MD.")
        assert extract_name("MD: JOHN DOE Date of Birth").startswith("MD:")

    def test_preceding_uppercase_noise_does_not_swallow_the_honorific(self):
        # "@PN MD. JOHN DOE": the fallback used to match "PN MD" as one run,
        # consuming the honorific before the name pattern could see it.
        assert extract_name("Cm @f @PN MD. JOHN DOE 5: 37*77 Tel 0") == "MD. JOHN DOE"

    def test_name_survives_unreadable_following_label(self):
        # The name used to be anchored on a following "Date of Birth"; when OCR
        # mangled that label, the honorific was lost.
        assert extract_name("Erer QP MD. JOHN DOE Jtt 0n Date 0( Dlrih;") == "MD. JOHN DOE"

    def test_field_labels_are_not_absorbed_into_the_name(self):
        assert extract_name("MD. JOHN DOE NID NO 1234567890123") == "MD. JOHN DOE"


class TestScreenshotChrome:
    """A document captured in a browser pulls UI text into the OCR output."""

    def test_browser_chrome_is_not_returned_as_a_name(self):
        text = (
            "Awesome Screen Recorder Screenshot chrome extension Developinent "
            "Interview MMERN Company Your Repositorles Rosearch Translate "
            "AWESOME SCREENSHOT Feedback Options MD: JOHN DOE 4 G7 Data of Birth"
        )
        assert extract_name(text) == "MD: JOHN DOE"
