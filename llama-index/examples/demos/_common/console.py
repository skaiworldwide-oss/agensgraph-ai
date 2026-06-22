"""Small console helpers so demos print clean, readable, timed output."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator, Optional, Sequence


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n {title}\n{'=' * 70}")


def sub(title: str) -> None:
    print(f"\n--- {title} ---")


def kv(label: str, value: object) -> None:
    print(f"  {label:<24} {value}")


class _Timer:
    def __init__(self) -> None:
        self.seconds = 0.0

    def rate(self, n: int, unit: str = "items") -> str:
        if self.seconds <= 0:
            return f"{n:,} {unit}"
        return f"{n:,} {unit} in {self.seconds:.1f}s  ({n / self.seconds:,.0f} {unit}/s)"


@contextmanager
def timer(label: str) -> Iterator[_Timer]:
    """Time a block; prints '<label>: <elapsed>s' and exposes .rate(n, unit)."""
    t = _Timer()
    start = time.perf_counter()
    try:
        yield t
    finally:
        t.seconds = time.perf_counter() - start
        print(f"  ⏱  {label}: {t.seconds:.2f}s")


def table(rows: Sequence[Sequence[object]], headers: Optional[Sequence[str]] = None) -> None:
    """Print a simple left-aligned text table."""
    str_rows = [[str(c) for c in row] for row in rows]
    cols = headers if headers else (str_rows[0] if str_rows else [])
    all_rows = ([list(map(str, cols))] if headers else []) + str_rows
    if not all_rows:
        return
    widths = [max(len(r[i]) for r in all_rows) for i in range(len(all_rows[0]))]
    if headers:
        print("  " + "  ".join(h.ljust(widths[i]) for i, h in enumerate(map(str, cols))))
        print("  " + "  ".join("-" * widths[i] for i in range(len(cols))))
    for row in str_rows:
        print("  " + "  ".join(row[i].ljust(widths[i]) for i in range(len(row))))
