from __future__ import annotations

from datetime import datetime
from pathlib import Path

from bs4 import BeautifulSoup
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from macrotones.api import loader


def _html_to_text(html_path: Path) -> list[str]:
    soup = BeautifulSoup(html_path.read_text(encoding="utf-8"), "html.parser")
    text = soup.get_text("\n")
    return text.splitlines()


def _write_pdf(lines: list[str], pdf_path: Path) -> None:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    width, height = letter
    y = height - 72
    page_num = 1
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    def _footer() -> None:
        c.drawString(72, 40, f"Page {page_num}")
        c.drawRightString(width - 72, 40, f"Generated {timestamp}")

    for line in lines:
        if not line.strip():
            y -= 14
            continue
        if y < 72:
            _footer()
            c.showPage()
            y = height - 72
            page_num += 1
        c.drawString(72, y, line[:100])
        y -= 14
    _footer()
    c.save()


def main() -> None:
    processed = loader.processed_path()
    html_path = processed / "report.html"
    if not html_path.exists():
        raise FileNotFoundError(f"HTML report not found: {html_path}")
    pdf_path = processed / "MacroTone_Report.pdf"
    lines = _html_to_text(html_path)
    _write_pdf(lines, pdf_path)
    print(f"Saved PDF -> {pdf_path}")


if __name__ == "__main__":
    main()
