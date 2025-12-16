from __future__ import annotations

import calendar
import hashlib
import io
import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional
from urllib.parse import urljoin, urlparse

import pandas as pd

OUT = Path("data/raw/fomc")
OUT.mkdir(parents=True, exist_ok=True)

MIN_TEXT_CHARS = 1_000
YEAR_START = 1993
FED_BASE = "https://www.federalreserve.gov"
FRASER_BASE = "https://fraser.stlouisfed.org/title/federal-open-market-committee-meeting-minutes-transcripts-677"
PRESS_BASE = f"{FED_BASE}/monetarypolicy"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; MacroTone/1.0)"}
RATE_LIMIT_SECONDS = 1.0

MONTHS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}

ISO_PATTERNS = [
    re.compile(r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b"),
    re.compile(r"\b(\d{1,2})[-/](\d{1,2})[-/](\d{4})\b"),
]
TEXT_PATTERNS = [
    re.compile(r"\b(?P<m>[A-Za-z]{3,9})\s+(?P<d>\d{1,2}),?\s+(?P<y>\d{4})\b"),
    re.compile(r"\b(?P<d>\d{1,2})\s+(?P<m>[A-Za-z]{3,9}),?\s+(?P<y>\d{4})\b"),
]
URL_PATTERNS = [
    re.compile(r"/(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})/"),
    re.compile(r"/(?P<y>\d{4})(?P<m>\d{2})(?P<d>\d{2})/"),
    re.compile(r"(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})"),
    re.compile(r"(?P<y>\d{4})(?P<m>\d{2})(?P<d>\d{2})"),
]

WHITELIST = (
    "/monetarypolicy/fomc",
    "/monetarypolicy/policy",
    "/monetarypolicy/press",
    "/newsevents/pressreleases/monetary",
    "fraser.stlouisfed.org/title/federal-open-market-committee",
)

BLACKLIST = (
    "/aboutthefed/",
    "/education/",
    "/careers/",
    "/consumer-",
    "/supervisionreg/",
    "/newsevents/pressreleases.htm",
)


try:  # dependency guardrails
    import requests
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "fetch_fomc requires `requests`. Install via `uv pip install requests`."
    ) from exc

try:
    from bs4 import BeautifulSoup
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "fetch_fomc requires `beautifulsoup4`. Install via `uv pip install beautifulsoup4`."
    ) from exc

try:
    from pdfminer.high_level import extract_text as pdf_extract_text
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "fetch_fomc requires `pdfminer.six`. Install via `uv pip install pdfminer.six`."
    ) from exc

try:
    import urllib3
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "fetch_fomc requires `urllib3`. Install via `uv pip install urllib3`."
    ) from exc

try:
    from tenacity import (
        retry,
        retry_if_exception_type,
        stop_after_attempt,
        wait_random_exponential,
    )
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "fetch_fomc requires `tenacity`. Install via `uv pip install tenacity`."
    ) from exc

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Candidate:
    url: str
    label: str
    source: str


class RateLimitedSession:
    """Thin wrapper around requests with polite throttling and retries."""

    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self._last_request = 0.0

    def _wait(self) -> None:
        now = time.time()
        elapsed = now - self._last_request
        if elapsed < RATE_LIMIT_SECONDS:
            time.sleep(RATE_LIMIT_SECONDS - elapsed)
        self._last_request = time.time()

    @retry(
        reraise=True,
        stop=stop_after_attempt(5),
        wait=wait_random_exponential(multiplier=1, max=30),
        retry=retry_if_exception_type(requests.RequestException),
    )
    def get(self, url: str, *, stream: bool = False) -> requests.Response:
        self._wait()
        resp = self.session.get(url, timeout=60, stream=stream)
        if resp.status_code >= 500:
            resp.raise_for_status()
        return resp


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = value.strip("-")
    return value or "fomc"


def month_end(ts: datetime) -> datetime:
    last_day = calendar.monthrange(ts.year, ts.month)[1]
    return ts.replace(day=last_day, hour=0, minute=0, second=0, microsecond=0)


def _valid_ymd(y: int, m: int, d: int) -> bool:
    try:
        if not (1900 <= y <= 2100):
            return False
        if not (1 <= m <= 12):
            return False
        if not (1 <= d <= 31):
            return False
        datetime(y, m, d)
        return True
    except Exception:
        return False


def _scan_for_date(text: str) -> Optional[datetime]:
    for pat in ISO_PATTERNS:
        for groups in pat.findall(text):
            if len(groups) != 3:
                continue
            nums = [int(x) for x in groups]
            a, b, c = nums
            candidates = [(a, b, c)] if a > 31 else [(c, a, b)]
            for y, m, d in candidates:
                if _valid_ymd(y, m, d):
                    return datetime(y, m, d)
    low = text.lower()
    for pat in TEXT_PATTERNS:
        for match in pat.finditer(low):
            gd = match.groupdict()
            y = int(gd["y"])
            d = int(gd["d"])
            mm = MONTHS.get(gd["m"].lower()[:3])
            if mm and _valid_ymd(y, mm, d):
                return datetime(y, mm, d)
    return None


def _date_from_url(url: str) -> Optional[datetime]:
    for pat in URL_PATTERNS:
        match = pat.search(url)
        if match:
            y = int(match.group("y"))
            m = int(match.group("m"))
            d = int(match.group("d"))
            if _valid_ymd(y, m, d):
                return datetime(y, m, d)
    return None


def extract_date(
    url: str, title: str, h1: str, text_snippet: str
) -> Optional[datetime]:
    for source in (url, title, h1, text_snippet):
        if not source:
            continue
        if source is url:
            dt = _date_from_url(source)
        else:
            dt = _scan_for_date(source)
        if dt:
            return dt
    return None


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return text.strip()


def html_to_text(html: str) -> tuple[str, str, str]:
    soup = BeautifulSoup(html, "lxml")
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    h1 = ""
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()
    for cls in ["navbar", "nav", "footer", "header", "breadcrumbs"]:
        for elem in soup.select(f".{cls}"):
            elem.decompose()
    heading = soup.find(["h1", "h2"])
    if heading:
        h1 = heading.get_text(" ", strip=True)
    text = soup.get_text("\n", strip=True)
    return clean_text(text), title, h1


def extract_pdf_text(binary: bytes) -> str:
    return clean_text(pdf_extract_text(io.BytesIO(binary)))


def is_document_link(label: str, href: str) -> bool:
    if not href:
        return False
    s = (label + " " + href).lower()
    keywords = [
        "minutes",
        "statement",
        "press conference",
        "pressconf",
        "transcript",
        "pressrelease",
        "implementation note",
        "policy statement",
        "postmeeting statement",
    ]
    if any(k in s for k in keywords):
        return True
    if href.lower().endswith(".pdf"):
        return True
    return False


def iter_links_from_html(base_url: str, html: str, source: str) -> Iterator[Candidate]:
    soup = BeautifulSoup(html, "lxml")
    for a in soup.find_all("a", href=True):
        label = a.get_text(" ", strip=True)
        href = a["href"]
        url = urljoin(base_url, href)
        if is_document_link(label, url):
            yield Candidate(url=url, label=label or url, source=source)


def _is_relevant(url: str) -> bool:
    if any(block in url for block in BLACKLIST):
        return False
    return any(white in url for white in WHITELIST)


def build_federal_reserve_index_urls() -> list[str]:
    urls = {
        f"{PRESS_BASE}/fomccalendars.htm",
        f"{PRESS_BASE}/fomchistorical.htm",
        f"{PRESS_BASE}/fomcpress.htm",
        f"{PRESS_BASE}/fomcminutes.htm",
    }
    current_year = datetime.now(timezone.utc).year
    for year in range(YEAR_START, current_year + 1):
        urls.add(f"{PRESS_BASE}/fomchistorical{year}.htm")
        urls.add(f"{PRESS_BASE}/fomcpress{year}.htm")
        urls.add(f"{PRESS_BASE}/fomcminutes{year}.htm")
    return sorted(urls)


def crawl_fraser(fetcher: RateLimitedSession) -> Iterator[Candidate]:
    page = FRASER_BASE
    visited = set()
    while page and page not in visited:
        visited.add(page)
        try:
            resp = fetcher.get(page)
        except Exception as exc:
            LOGGER.warning("FRASER page failed %s (%s)", page, exc)
            break
        html = resp.text
        soup = BeautifulSoup(html, "lxml")
        for a in soup.find_all("a", href=True):
            label = a.get_text(" ", strip=True)
            href = urljoin(page, a["href"])
            if "/files/" in href or href.lower().endswith(".pdf"):
                yield Candidate(url=href, label=label or href, source="fraser")
            elif "/item/" in href:
                yield Candidate(url=href, label=label or href, source="fraser")
        next_link = soup.find("a", string=re.compile("Next", re.IGNORECASE))
        if next_link and next_link.get("href"):
            page = urljoin(page, next_link["href"])
        else:
            page = None


def load_existing_hashes() -> dict[str, Path]:
    hashes: dict[str, Path] = {}
    for path in OUT.glob("*.txt"):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="latin-1")
        digest = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()
        hashes[digest] = path
    return hashes


def ensure_unique_path(base_name: str) -> Path:
    candidate = OUT / base_name
    if not candidate.exists():
        return candidate
    stem = candidate.stem
    suffix = candidate.suffix
    for idx in range(1, 1000):
        new_candidate = OUT / f"{stem}-{idx}{suffix}"
        if not new_candidate.exists():
            return new_candidate
    raise RuntimeError(f"Unable to create unique filename for {base_name}")


def gather_candidates(fetcher: RateLimitedSession) -> list[Candidate]:
    candidates: dict[str, Candidate] = {}
    for url in build_federal_reserve_index_urls():
        try:
            resp = fetcher.get(url)
        except Exception as exc:
            LOGGER.warning("Index fetch failed %s (%s)", url, exc)
            continue
        for cand in iter_links_from_html(url, resp.text, source="federalreserve"):
            candidates[cand.url] = cand
    for cand in crawl_fraser(fetcher):
        candidates[cand.url] = cand
    filtered = [cand for cand in candidates.values() if _is_relevant(cand.url.lower())]
    LOGGER.info("Discovered %d unique candidate links (after filtering)", len(filtered))
    return filtered


def extract_document(
    fetcher: RateLimitedSession, cand: Candidate
) -> Optional[tuple[str, str, str, str]]:
    try:
        resp = fetcher.get(cand.url, stream=False)
    except Exception as exc:
        LOGGER.warning("Failed fetch %s (%s)", cand.url, exc)
        return None
    content_type = resp.headers.get("Content-Type", "").lower()
    if "pdf" in content_type or cand.url.lower().endswith(".pdf"):
        data = resp.content
        text = extract_pdf_text(data)
        title = ""
        h1 = ""
    else:
        text, title, h1 = html_to_text(resp.text)
    if not text:
        return None
    return text, title, h1, content_type


def main() -> None:
    configure_logging()
    LOGGER.info("Starting FOMC multi-source crawl")
    existing_hashes = load_existing_hashes()
    fetcher = RateLimitedSession()
    candidates = gather_candidates(fetcher)

    scanned_pages = 0
    kept_files = 0
    fallback_dates = 0
    for cand in candidates:
        scanned_pages += 1
        try:
            doc = extract_document(fetcher, cand)
            if not doc:
                continue
            text, title, h1, _ = doc
            text = clean_text(text)
            if len(text) < MIN_TEXT_CHARS:
                continue
            digest = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()
            if digest in existing_hashes:
                continue
            parsed_url = urlparse(cand.url)
            fallback_slug = os.path.splitext(Path(parsed_url.path).name)[0]
            slug = slugify(cand.label or title or fallback_slug)
            dt = extract_date(
                cand.url,
                title,
                h1,
                text[:2_500],
            )
            if not dt:
                fallback_match = re.search(
                    r"fomc_(\d{4}-\d{2}-\d{2})",
                    Path(cand.url).name,
                )
                if fallback_match:
                    dt = pd.to_datetime(fallback_match.group(1))
                    fallback_dates += 1
                else:
                    LOGGER.warning("Skip (no date): %s", cand.url)
                    continue
            dt = month_end(dt)
            filename = f"{dt:%Y-%m-%d}_{slug}.txt"
            out_path = ensure_unique_path(filename)
            out_path.write_text(text, encoding="utf-8")
            existing_hashes[digest] = out_path
            kept_files += 1
            LOGGER.info("Saved %s (%s)", out_path, cand.url)
        except Exception as exc:  # pragma: no cover - best-effort crawler
            LOGGER.warning("Skip (error %s): %s", exc.__class__.__name__, cand.url)
            continue

    total_files = len(existing_hashes)
    LOGGER.info(
        "Finished crawl: scanned=%d new=%d total=%d",
        scanned_pages,
        kept_files,
        total_files,
    )
    LOGGER.info("Date fallback applied to %d documents", fallback_dates)
    print(
        f"Scanned {scanned_pages} pages; kept {kept_files} new files; total files now: {total_files}"
    )


if __name__ == "__main__":
    main()
