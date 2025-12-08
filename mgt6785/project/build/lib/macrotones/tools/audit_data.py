from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

TARGETS = [
    ("data/raw/ff/ff_monthly.parquet", "file"),
    ("data/raw/fomc", "fomc"),
    ("data/interim/nlp_doc_scores.parquet", "file"),
    ("data/processed", "glob"),
]

MANIFEST_PATH = Path("data/manifest.json")
HASH_CHUNK_BYTES = 1 << 20


class AuditError(RuntimeError):
    """Raised when drift is detected and update flag is not set."""


@dataclass(frozen=True)
class FileStats:
    path: str
    size: int
    mtime: float
    sha1: str
    rows: int | None

    def as_row(self) -> list[str]:
        timestamp = datetime.fromtimestamp(self.mtime).isoformat(timespec="seconds")
        rows = "" if self.rows is None else f"{self.rows}"
        size = f"{self.size:,}"
        return [self.path, size, rows, timestamp, self.sha1]


def sha1_of_path(path: Path) -> str:
    hsh = hashlib.sha1()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(HASH_CHUNK_BYTES)
            if not chunk:
                break
            hsh.update(chunk)
    return hsh.hexdigest()


def count_rows(path: Path) -> int | None:
    if path.suffix == ".parquet":
        try:
            return pd.read_parquet(path).shape[0]
        except Exception:
            return None
    if path.suffix in {".csv"}:
        try:
            return pd.read_csv(path).shape[0]
        except Exception:
            return None
    if path.suffix in {".txt"}:
        try:
            with path.open("r", encoding="utf-8", errors="ignore") as handle:
                return sum(1 for _ in handle)
        except Exception:
            return None
    return None


def iter_target_files(base: Path, mode: str) -> Iterable[Path]:
    if mode == "file":
        if base.exists():
            yield base
        return
    if mode == "fomc":
        if base.exists():
            yield from sorted(base.glob("*.txt"))
        return
    if mode == "glob":
        if base.exists():
            for path in sorted(base.iterdir()):
                if path.is_file():
                    yield path
        return
    raise ValueError(f"Unknown mode: {mode}")


def collect_stats() -> dict[str, FileStats]:
    stats: dict[str, FileStats] = {}
    for target, mode in TARGETS:
        base = Path(target)
        for file_path in iter_target_files(base, mode):
            stat = file_path.stat()
            rows = count_rows(file_path)
            stats[file_path.as_posix()] = FileStats(
                path=file_path.as_posix(),
                size=stat.st_size,
                mtime=stat.st_mtime,
                sha1=sha1_of_path(file_path),
                rows=rows,
            )
    return stats


def load_manifest(path: Path) -> dict[str, FileStats]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return {k: FileStats(**v) for k, v in raw.items()}


def save_manifest(path: Path, stats: Mapping[str, FileStats]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: asdict(v) for k, v in stats.items()}
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def diff_stats(
    old: Mapping[str, FileStats], new: Mapping[str, FileStats]
) -> list[list[str]]:
    rows: list[list[str]] = []
    keys = sorted(set(old) | set(new))
    for key in keys:
        before = old.get(key)
        after = new.get(key)
        status = ""
        if before and after:
            if before == after:
                status = "OK"
            else:
                status = "CHANGED"
        elif before and not after:
            status = "REMOVED"
        elif after and not before:
            status = "ADDED"
        row = [status]
        if after:
            row.extend(after.as_row())
        else:
            # for removed files, reuse previous metadata
            row.extend(before.as_row() if before else ["", "", "", "", ""])
        rows.append(row)
    return rows


def render_table(rows: list[list[str]]) -> str:
    headers = ["Status", "Path", "Size (bytes)", "Rows/Lines", "Modified", "SHA1"]
    table = [headers, *rows]
    col_widths = [max(len(str(row[i])) for row in table) for i in range(len(headers))]

    def fmt(row: list[str]) -> str:
        cells = [str(cell).ljust(col_widths[i]) for i, cell in enumerate(row)]
        return " | ".join(cells)

    sep = "-+-".join("-" * w for w in col_widths)
    lines = [fmt(headers), sep]
    lines.extend(fmt(row) for row in rows)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit local MacroTone data artifacts."
    )
    parser.add_argument(
        "--update", action="store_true", help="Update manifest after checks."
    )
    args = parser.parse_args(argv)

    if MANIFEST_PATH.exists():
        baseline = load_manifest(MANIFEST_PATH)
    else:
        baseline = {}

    current = collect_stats()
    rows = diff_stats(baseline, current)
    print(render_table(rows))

    drift = any(row[0] not in {"", "OK"} for row in rows)

    if drift and not args.update and baseline:
        raise AuditError(
            "Data integrity drift detected. Re-run with --update to accept."
        )

    if args.update or not baseline:
        save_manifest(MANIFEST_PATH, current)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AuditError as err:
        print(f"ERROR: {err}")
        raise SystemExit(1) from err
