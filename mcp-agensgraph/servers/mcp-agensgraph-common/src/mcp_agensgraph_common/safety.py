"""Query safety helpers shared across the AgensGraph MCP servers.

- ``quote_identifiers``  — quote mixed-case labels/properties for AgensGraph's
  case-sensitive (PostgreSQL) identifier rules.
- ``quote_label``        — quote a single label/relationship-type token safely,
  for the cases where Cypher cannot parameterize an identifier (e.g. a
  relationship *type* in a MERGE pattern).
- ``is_write_query``     — a comment/string-aware heuristic used only to return a
  friendly error in read-only mode. It is NOT the security boundary; the database
  read-only transaction (see ``connection.read_only_session``) is.
"""

from __future__ import annotations

import re

# Mixed-case labels: :Label -> :"Label" (skip already-quoted :"...").
_LABEL_RE = re.compile(r':(?!")([A-Z][a-zA-Z0-9_]*)')
# Mixed-case property keys in a map literal: {Prop:  or , Prop:  -> "Prop":
_PROP_KEY_RE = re.compile(r'([{,]\s*)([A-Z][a-zA-Z0-9_]*)\s*:')
# Mixed-case property access: .Prop -> ."Prop" (skip already-quoted)
_PROP_ACCESS_RE = re.compile(r'\.(?!")([A-Z][a-zA-Z0-9_]*)\b')

_WRITE_KEYWORDS = re.compile(
    r"\b(MERGE|CREATE|SET|DELETE|REMOVE|DETACH|DROP|LOAD)\b", re.IGNORECASE
)
# Cypher line (// ...) and block (/* ... */) comments, and quoted string literals.
_COMMENTS_AND_STRINGS = re.compile(
    r"//[^\n]*|/\*.*?\*/|'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\"",
    re.DOTALL,
)

_VALID_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def quote_identifiers(query: str) -> str:
    """Quote mixed-case labels and property names in a Cypher query string.

    AgensGraph folds unquoted identifiers to lowercase (PostgreSQL semantics), so a
    label like ``:Person`` must be written ``:"Person"`` to preserve case.
    """
    query = _LABEL_RE.sub(r':"\1"', query)
    query = _PROP_KEY_RE.sub(r'\1"\2":', query)
    query = _PROP_ACCESS_RE.sub(r'."\1"', query)
    return query


def quote_label(label: str) -> str:
    """Safely quote a single label / relationship-type token.

    Cypher cannot parameterize an identifier (label or relationship type), so when
    one comes from tool input it must be validated and quoted rather than
    interpolated raw. Rejects anything that is not a valid identifier to prevent
    breaking out of the ``:"..."`` quoting.
    """
    if not isinstance(label, str) or not _VALID_IDENTIFIER.match(label):
        raise ValueError(
            f"Invalid label/relationship type: {label!r}. Must match "
            "[A-Za-z_][A-Za-z0-9_]* (letters, digits, underscores)."
        )
    return f'"{label}"'


def strip_comments_and_strings(query: str) -> str:
    """Remove comments and string literals so keyword scanning can't be fooled."""
    return _COMMENTS_AND_STRINGS.sub(" ", query)


def is_write_query(query: str) -> bool:
    """Heuristic: does the query contain a write clause (ignoring comments/strings)?

    Used only for a fast, friendly read-only error message. The real guarantee is
    the database-side read-only transaction.
    """
    return _WRITE_KEYWORDS.search(strip_comments_and_strings(query)) is not None
