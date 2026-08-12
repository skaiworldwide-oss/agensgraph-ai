"""Small console helpers so demos print clean, readable, timed output."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Iterator, Optional, Sequence

# Every server logs a refused tool call at ERROR with its traceback, and so does FastMCP.
# That is right for a server and wrong for a demo whose whole point is the refusal: the
# traceback lands ahead of the section that explains it, and in a notebook it lands in a
# separate stream from the line it belongs to.
_NOISY = ("fastmcp", "mcp_agensgraph_cypher", "mcp_agensgraph_common", "mcp_agensgraph_memory")


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n {title}\n{'=' * 70}")


def sub(title: str) -> None:
    print(f"\n--- {title} ---")


def kv(label: str, value: object) -> None:
    print(f"  {label:<26} {value}")


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


@contextmanager
def expecting_refusal() -> Iterator[None]:
    """Quiet the servers' own logging around a call the demo means to be refused."""
    raised = [(logging.getLogger(name), logging.getLogger(name).level) for name in _NOISY]
    for log, _ in raised:
        log.setLevel(logging.CRITICAL)
    try:
        yield
    finally:
        for log, level in raised:
            log.setLevel(level)


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
