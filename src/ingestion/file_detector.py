"""
src/ingestion/file_detector.py
==============================
Finds Fannie Mae quarterly files in data/raw and works out which vintage
(year + quarter) each one belongs to from its filename.

Fannie Mae downloads are typically named like:
    2018Q1.csv   or   Performance_2018Q1.csv   or   2018Q1.zip
We look for a YYYYQn pattern anywhere in the name.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# Matches 2018Q1, 2018q1, 2018-Q1, etc.
_VINTAGE_RE = re.compile(r"(20\d{2})[\-_ ]?[Qq]([1-4])")


@dataclass
class QuarterlyFile:
    path: Path
    year: int
    quarter: int

    @property
    def vintage(self) -> str:
        return f"{self.year}Q{self.quarter}"


def parse_vintage(filename: str) -> tuple[int, int] | None:
    """Return (year, quarter) parsed from a filename, or None if not found."""
    m = _VINTAGE_RE.search(filename)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def detect_files(raw_dir: Path, extensions=(".csv", ".txt")) -> list[QuarterlyFile]:
    """Scan raw_dir for quarterly data files and return them sorted by vintage."""
    found: list[QuarterlyFile] = []
    for path in sorted(raw_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        parsed = parse_vintage(path.name)
        if parsed is None:
            continue
        year, quarter = parsed
        found.append(QuarterlyFile(path=path, year=year, quarter=quarter))
    found.sort(key=lambda f: (f.year, f.quarter))
    return found
