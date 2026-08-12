"""See what this package actually sends, and how long each statement took.

The usual question about a slow retrieval chain is which part of it was slow, and the
answer is rarely where it is looked for first: a search that takes two seconds is more
often one embedding call and twenty milliseconds of database than the other way round.
The driver reports every statement it runs, after it finishes or fails; this is that
report, scoped to a block so it can be asked of one call rather than of a process.

::

    from langchain_agensgraph import log_queries

    with log_queries() as statements:
        chain.invoke({"query": "Who works at Acme?"})

    for one in statements:
        print(f"{one.elapsed * 1000:6.1f} ms  {one.statement[:80]}")

Or send them somewhere as they happen::

    with log_queries(lambda one: log.info("%s: %.1f ms", one.statement, one.elapsed * 1000)):
        ...

The driver measures this at about a quarter of a per cent, so a block can be left wrapped.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, Iterator, List, Optional

from agensgraph.observability import add_query_logger, remove_query_logger

__all__ = ["log_queries"]


@contextmanager
def log_queries(
    report: Optional[Callable[[Any], None]] = None,
) -> Iterator[List[Any]]:
    """Collect every statement run inside the block, in the order they finished.

    Each record carries the statement, how long it took, how many rows it returned and
    whether it failed. With ``report``, one is also handed to it as it finishes -- on the
    thread or task that ran the statement, so it should not block.

    The list is yielded as well as reported, since a notebook or a one-off investigation
    wants the whole burst afterwards rather than a callback.
    """
    collected: List[Any] = []

    def collect(record: Any) -> None:
        collected.append(record)
        if report is not None:
            report(record)

    add_query_logger(collect)
    try:
        yield collected
    finally:
        remove_query_logger(collect)
