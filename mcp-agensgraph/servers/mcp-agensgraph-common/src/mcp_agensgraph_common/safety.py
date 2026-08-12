"""Quoting identifiers in a Cypher statement written by a model.

``quote_identifiers`` is the whole module. What a statement is allowed to do is not decided
here and is not decided by reading it: a read runs inside ``read_only_transaction``, where the
server refuses a write with ``25006``, and ``agensgraph.cypher.writable_counters`` and
``check_single_statement`` are what the servers ask about the text.

A single name that arrives as a value -- a label, a relationship type -- is quoted with
``agensgraph.cypher.quote_identifier``, which quotes what needs quoting rather than refusing
what it does not recognise.
"""

from __future__ import annotations

import re

from agensgraph.cypher import without_literals

# Mixed-case labels: :Label -> :"Label" (skip already-quoted :"...").
_LABEL_RE = re.compile(r':(?!")([A-Z][a-zA-Z0-9_]*)')
# Mixed-case property keys in a map literal: {Prop:  or , Prop:  -> "Prop":
_PROP_KEY_RE = re.compile(r'([{,]\s*)([A-Z][a-zA-Z0-9_]*)\s*:')
# Mixed-case property access: .Prop -> ."Prop" (skip already-quoted)
_PROP_ACCESS_RE = re.compile(r'\.(?!")([A-Z][a-zA-Z0-9_]*)\b')


def quote_identifiers(query: str) -> str:
    """Quote mixed-case labels and property names in a Cypher query string.

    AgensGraph folds an unquoted identifier to lower case, so a label written ``:Person``
    reaches the server as ``person`` and matches nothing in a graph whose labels were created
    with their case kept. A statement written by a model says ``:Person``, and this is what
    makes it find what the data-modeling tools created.

    Nothing inside a string, a comment or an already-quoted name is rewritten. The rewriting is
    located against the statement with those blanked out -- the driver blanks them to spaces of
    the same length, so a position in the blanked text is the same position in the original --
    and applied to the original at the positions found. Without that, a value being searched for
    is edited: ``WHERE n.tag = 'a:Bcd'`` became ``'a:"Bcd"'``, and a row that existed stopped
    being found.
    """
    blanked = without_literals(query)
    edits: list[tuple[int, int, str]] = []
    for pattern, render in (
        (_LABEL_RE, lambda m: f':"{m.group(1)}"'),
        (_PROP_KEY_RE, lambda m: f'{m.group(1)}"{m.group(2)}":'),
        (_PROP_ACCESS_RE, lambda m: f'."{m.group(1)}"'),
    ):
        for found in pattern.finditer(blanked):
            edits.append((found.start(), found.end(), render(found)))
    if not edits:
        return query
    out: list[str] = []
    at = 0
    for start, end, replacement in sorted(edits):
        if start < at:
            continue  # Overlapping with an edit already taken.
        out.append(query[at:start])
        out.append(replacement)
        at = end
    out.append(query[at:])
    return "".join(out)
