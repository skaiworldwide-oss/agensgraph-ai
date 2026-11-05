"""Names the package gives things in the database, and how they are written into a statement.

The server keeps 63 bytes of an identifier and silently cuts the rest, so two long names that
differ only after the cut would be the same table. A name the package derives is therefore kept
whole when it fits and otherwise shortened to a prefix and a hash of the whole, which is unique
and the same on every run. A name the caller chose is never shortened: one that does not fit, or
that holds a null byte, is refused, since the server would read a different name than was given.
"""

from __future__ import annotations

import hashlib

MAX_IDENTIFIER_BYTES = 63
_HASH_CHARS = 12


def ident(name: str) -> str:
    """A name written as a quoted identifier, so case, spaces and reserved words survive.

    For a statement the package runs: the executor always binds parameters, so psycopg reads a
    ``%`` in the text as the start of a placeholder, and one in the name is doubled.
    """
    return quoted(name).replace("%", "%%")


def literal(text: str) -> str:
    """Text written as a string literal, for where a statement cannot take a parameter.

    For a statement the package runs, like :func:`ident`.
    """
    return "'" + text.replace("'", "''").replace("%", "%%") + "'"


def quoted(name: str) -> str:
    """A quoted identifier for SQL that is printed, not run through the executor."""
    return '"' + name.replace('"', '""') + '"'


def quoted_literal(text: str) -> str:
    """A string literal for SQL that is printed, not run through the executor."""
    return "'" + text.replace("'", "''") + "'"


def checked(name: str, what: str = "name") -> str:
    """A name the caller chose, refused if the server would not read it as given."""
    if not isinstance(name, str) or not name:
        raise ValueError(f"the {what} must be a non-empty string, not {name!r}")
    if "\x00" in name:
        raise ValueError(f"the {what} {name!r} holds a null byte, which the server stops at")
    if len(name.encode()) > MAX_IDENTIFIER_BYTES:
        raise ValueError(
            f"the {what} {name!r} is longer than {MAX_IDENTIFIER_BYTES} bytes, and the server "
            "would cut it short"
        )
    return name


def bounded(name: str) -> str:
    """A derived name as the server will keep it: whole if it fits, else a prefix and a hash."""
    data = name.encode()
    if len(data) <= MAX_IDENTIFIER_BYTES:
        return name
    digest = hashlib.blake2b(data, digest_size=_HASH_CHARS // 2).hexdigest()
    prefix = data[: MAX_IDENTIFIER_BYTES - _HASH_CHARS - 1].decode(errors="ignore")
    return f"{prefix}_{digest}"


def companion_schema(graph: str) -> str:
    """The schema beside a graph that holds the package's tables for it: ``"<graph>_store"``."""
    return bounded(f"{graph}_store")


def companion_table(*parts: str) -> str:
    """A table in the companion schema named for what it belongs to: ``"Chunk__embedding__vec"``."""
    return bounded("__".join(parts))


def qualified(schema: str, name: str) -> str:
    """``"schema"."name"``."""
    return f"{ident(schema)}.{ident(name)}"


def printed(schema: str, name: str) -> str:
    """``"schema"."name"`` as a parameter's value, such as ``to_regclass`` reads: the form of
    :func:`qualified` without its ``%`` doubled."""
    return f"{quoted(schema)}.{quoted(name)}"
