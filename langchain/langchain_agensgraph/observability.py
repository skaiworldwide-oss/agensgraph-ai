"""See what this package sends, and how long each statement took.

The usual question about a slow retrieval chain is which part of it was slow, and the answer is
rarely where it is looked for first: a search that takes two seconds is more often one embedding
call and twenty milliseconds of database than the other way round. The driver reports every
statement it runs, after it finishes or fails; :func:`log_queries` hands those reports to the block
that asked, and to no other::

    from langchain_agensgraph import log_queries

    with log_queries() as statements:
        chain.invoke({"query": "Who works at Acme?"})

    for one in statements:
        print(f"{one.elapsed * 1000:6.1f} ms  {one.statement[:80]}")

A block sees the statements run in its own context: on its thread or task, and on tasks and
LangChain worker threads started from it, which inherit the context. Another request served by the
same process at the same time is not collected. A report carries the statement text, never a
parameter's value.
"""

from __future__ import annotations

import contextvars
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager

from agensgraph.observability import QueryRecord, add_query_logger

__all__ = ["QueryRecord", "log_queries"]

_collectors: contextvars.ContextVar[tuple[Callable[[QueryRecord], None], ...]] = (
    contextvars.ContextVar("langchain_agensgraph_query_collectors", default=())
)
_installed = False
_install_lock = threading.Lock()


def _dispatch(record: QueryRecord) -> None:
    for collect in _collectors.get():
        collect(record)


def _install() -> None:
    global _installed
    with _install_lock:
        if not _installed:
            add_query_logger(_dispatch)
            _installed = True


@contextmanager
def log_queries(
    report: Callable[[QueryRecord], None] | None = None,
) -> Iterator[list[QueryRecord]]:
    """Collect every statement run in this context inside the block, in the order they finished.

    Each record carries the statement, how long it took, how many rows it returned and whether it
    failed. With ``report``, each is also handed to it as it finishes, on the thread or task that
    ran the statement, so it should not block. Blocks nest: an inner block's statements are
    collected by the outer one too.
    """
    collected: list[QueryRecord] = []

    def collect(record: QueryRecord) -> None:
        collected.append(record)
        if report is not None:
            report(record)

    _install()
    token = _collectors.set((*_collectors.get(), collect))
    try:
        yield collected
    finally:
        _collectors.reset(token)
