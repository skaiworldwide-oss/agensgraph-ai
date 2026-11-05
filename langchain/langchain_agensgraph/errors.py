"""What this package raises.

A refusal of the package's own is an :class:`AgensGraphError`. A failure the server reports reaches
the caller as the driver raised it: a psycopg error, or one of the driver's own subclasses, the
ones a caller is likely to handle being re-exported here. The exception is text the caller wrote,
which :meth:`~langchain_agensgraph.AgensGraph.query` raises as :class:`AgensQueryError` with the
server's SQLSTATE and the driver's error as its cause. No error carries a parameter's value.
"""

from __future__ import annotations

from agensgraph.errors import (
    CapabilityError,
    ConfigurationError,
    ReadOnlyGraphWrite,
    StaleVectorStorage,
    UnresolvedCommit,
    VectorStorageMismatch,
)

__all__ = [
    "AgensGraphError",
    "AgensQueryError",
    "CapabilityError",
    "ClosedError",
    "ConfigurationError",
    "DuplicateKeys",
    "ReadOnlyGraphWrite",
    "SchemaMismatchError",
    "StaleVectorStorage",
    "UnknownLabel",
    "UnresolvedCommit",
    "UnsafeCypherError",
    "VectorStorageMismatch",
]


class AgensGraphError(Exception):
    """Everything this package refuses of its own accord."""


class AgensQueryError(AgensGraphError):
    """The server refused a statement the caller wrote.

    ``sqlstate`` is the server's code for the failure, and ``__cause__`` the driver's error.
    """

    def __init__(self, message: str, *, sqlstate: str | None = None) -> None:
        super().__init__(message)
        self.sqlstate = sqlstate


class UnsafeCypherError(AgensGraphError, ValueError):
    """Model-written Cypher was refused before it was sent."""


class SchemaMismatchError(AgensGraphError, ValueError):
    """What exists in the database is not what the component was configured for."""


class DuplicateKeys(AgensGraphError):
    """A unique index could not be built because the data already holds duplicates."""


class UnknownLabel(AgensGraphError, LookupError):
    """A label the call needs does not exist, and the component was not asked to create it."""


class ClosedError(AgensGraphError, RuntimeError):
    """The component or engine was used after it was closed."""
