from datetime import datetime

import pytest

from macrotones.data.fetch_fomc import (
    _date_from_url,
    _scan_for_date,
    _valid_ymd,
    extract_date,
)


@pytest.mark.parametrize(
    ("y", "m", "d", "expected"),
    [
        (2024, 1, 31, True),
        (1899, 12, 1, False),
        (2024, 13, 1, False),
        (2024, 2, 30, False),
    ],
)
def test_valid_ymd_bounds(y: int, m: int, d: int, expected: bool) -> None:
    assert _valid_ymd(y, m, d) is expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Meeting held on 2024-01-31 in DC", datetime(2024, 1, 31)),
        ("Minutes from 01/30/2023 released", datetime(2023, 1, 30)),
        ("Statement June 14, 2017 discusses policy", datetime(2017, 6, 14)),
        ("Announcement on 14 June 2017", datetime(2017, 6, 14)),
        ("Budget 2024-13-01 is invalid", None),
    ],
)
def test_scan_for_date_iso_and_text(text: str, expected: datetime | None) -> None:
    assert _scan_for_date(text) == expected


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://fed.gov/releases/2017-06-14/minutes", datetime(2017, 6, 14)),
        ("https://fed.gov/releases/20170614/minutes", datetime(2017, 6, 14)),
        ("https://fed.gov/releases/2017-13-14/minutes", None),
    ],
)
def test_date_from_url_patterns(url: str, expected: datetime | None) -> None:
    assert _date_from_url(url) == expected


def test_extract_date_prefers_url_then_text() -> None:
    url = "https://fed.gov/releases/2017-06-14/minutes"
    title = "Federal Reserve Statement June 32, 2017"  # invalid day
    h1 = ""
    snippet = "Policy decision announced on June 14, 2017."
    assert extract_date(url, title, h1, snippet) == datetime(2017, 6, 14)


def test_extract_date_rejects_invalid_values() -> None:
    url = "https://fed.gov/minutes/latest"
    title = "Summary 2024-13-01"  # invalid month
    h1 = "Released on 02/32/2024"  # invalid day
    snippet = "Meeting recap without valid date."
    assert extract_date(url, title, h1, snippet) is None
