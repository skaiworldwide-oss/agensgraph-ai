"""Data-modeling-server utilities.

Config parsing comes from the shared ``mcp_agensgraph_common`` core (this server is
DB-less, so it only needs the transport section). ``_quote_identifiers`` is kept
local: it does richer, DDL-aware quoting (VLABEL/ELABEL/ON/ASSERT, camelCase
properties) than the common helper, because this server *generates* ingest and
constraint Cypher for users to run.
"""

from __future__ import annotations

import argparse
import re
from typing import Any

from mcp_agensgraph_common.config import format_namespace, transport_config

__all__ = ["process_config", "format_namespace", "_quote_identifiers"]


def process_config(args: argparse.Namespace) -> dict[str, Any]:
    """Build the config dict for ``server.main`` — transport only (no DB)."""
    return transport_config(args)


def _quote_identifiers(query: str) -> str:
    """
    Quote identifiers with uppercase letters for AgensGraph case sensitivity.

    AgensGraph (like PostgreSQL) treats unquoted identifiers as case-insensitive
    (lowercase). To preserve case, identifiers must be quoted with double quotes.

    Quotes labels (`:Person` -> `:"Person"`), property names in maps and access
    (`.FirstName` -> `."FirstName"`), and the label/property names in
    VLABEL/ELABEL/ON/ASSERT DDL clauses. Handles both PascalCase and camelCase.
    """
    # VLABEL/ELABEL IF NOT EXISTS Label -> ... "Label" (first, to avoid conflicts)
    query = re.sub(
        r'\b(VLABEL|ELABEL)\s+IF\s+NOT\s+EXISTS\s+(?!")([A-Za-z][a-zA-Z0-9_]*)\b',
        r'\1 IF NOT EXISTS "\2"',
        query,
    )
    # ON Label (constraint syntax), with and without a following ASSERT
    query = re.sub(
        r'\bON\s+(?!")([A-Za-z][a-zA-Z0-9_]*)\b(?!\s+ASSERT)', r'ON "\1"', query
    )
    query = re.sub(
        r'\bON\s+(?!")([A-Za-z][a-zA-Z0-9_]*)\b(\s+ASSERT)', r'ON "\1"\2', query
    )
    # Labels with uppercase: :Label or : Label -> :"Label"
    query = re.sub(r':\s*(?!")([A-Z][a-zA-Z0-9_]*)\b', r': "\1"', query)
    # Property keys in a map literal (PascalCase or camelCase): {PropName: -> {"PropName":
    query = re.sub(
        r"([{,]\s*)([a-zA-Z][a-zA-Z0-9_]*[A-Z][a-zA-Z0-9_]*)\s*:", r'\1"\2":', query
    )
    # Property access (PascalCase or camelCase): .PropName -> ."PropName"
    query = re.sub(
        r'\.(?!")([a-zA-Z][a-zA-Z0-9_]*[A-Z][a-zA-Z0-9_]*)\b', r'."\1"', query
    )
    # Property names in an ASSERT clause: ASSERT propName IS UNIQUE -> ASSERT "propName" ...
    query = re.sub(
        r'\bASSERT\s+(?!")([a-zA-Z][a-zA-Z0-9_]*[A-Z][a-zA-Z0-9_]*)\b',
        r'ASSERT "\1"',
        query,
    )
    return query
