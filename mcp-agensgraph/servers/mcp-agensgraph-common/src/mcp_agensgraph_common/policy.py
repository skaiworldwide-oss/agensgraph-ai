"""What a statement from somewhere else is allowed to be.

AgensGraph is PostgreSQL underneath, so the endpoint that takes Cypher takes SQL too. A graph
tool that passes a statement through therefore offers the whole of SQL -- measured through the
shipped tools, that meant `CREATE ROLE ... SUPERUSER LOGIN` succeeding, `pg_read_file('/etc/passwd')`
returning, and `CREATE TABLE`/`INSERT`/`DELETE` landing in the public schema.

The database's privilege system is the boundary and stays the boundary: connected as a role holding
no grants, every one of those is refused `42501` without any of this. What is here is the second
thing, for the deployment that is not that: a statement is named by what it begins with, and only
the beginnings a graph statement has are allowed through.

An allowlist rather than a list of things to refuse, because the refusing kind has to recognise
every spelling of harm and the allowing kind only has to recognise the shapes this tool is for.
The one that was here before missed `INSERT`, `TRUNCATE`, `COPY`, `ALTER`, `GRANT` and `CALL`.
"""

from __future__ import annotations

import re

from agensgraph.cypher import without_literals

__all__ = [
    "GRAPH_DDL",
    "PolicyRefusal",
    "check_statement_is_allowed",
]


class PolicyRefusal(ValueError):
    """A statement this tool does not offer, with the reason a caller can act on."""


# What a graph statement begins with. Read from the grammar's lead-clause set, plus the GQL
# clauses 2.18 added: a statement starts at one of these or it is not one of ours.
_GRAPH_LEAD = (
    "match", "optional", "create", "insert", "merge", "set", "remove", "delete", "detach",
    "with", "unwind", "return", "call", "let", "for", "filter", "finish", "load", "explain",
)

# The same word begins a graph statement and a SQL one, so the word alone cannot decide. What
# follows separates them: SQL names a table to act on, Cypher names a pattern.
_SQL_TELL = {
    "insert": re.compile(r"\Ainsert\s+into\b", re.IGNORECASE),
    "delete": re.compile(r"\Adelete\s+from\b", re.IGNORECASE),
    "create": re.compile(
        r"\Acreate\s+(?:or\s+replace\s+)?"
        r"(?:table|view|schema|function|procedure|extension|database|type|sequence|"
        r"trigger|rule|domain|operator|aggregate|server|publication|subscription|"
        r"materialized|temp|temporary|unlogged|foreign)\b",
        re.IGNORECASE,
    ),
}

# Statements about who may do what. Refused whatever else is configured: a tool that can grant
# itself a privilege cannot be constrained by any rule written after it.
_PRIVILEGE = re.compile(
    r"\A(?:grant|revoke|reassign|"
    r"(?:create|alter|drop)\s+(?:role|user|group)|"
    r"set\s+(?:role|session\s+authorization)|"
    r"alter\s+(?:default\s+privileges|system))\b",
    re.IGNORECASE,
)

# Reaches outside the database entirely, and a read-only transaction does not stop the first of
# them: it moves rows out rather than in, so there is no write for the server to refuse.
_LEAVES_THE_DATABASE = re.compile(r"\Acopy\b|\bto\s+program\b|\bfrom\s+program\b", re.IGNORECASE)

# Changing the shape of the graph rather than what is in it. Available, but only to a server
# told to offer it, because a tool that ingests data does not need to drop a graph.
GRAPH_DDL = re.compile(
    r"\A(?:create|drop|alter)\s+(?:graph|vlabel|elabel|label|property\s+index|constraint)\b"
    r"|\Acreate\s+unique\s+property\s+index\b"
    r"|\Adrop\s+(?:property\s+index|constraint)\b",
    re.IGNORECASE,
)


def _first_word(statement: str) -> str:
    stripped = statement.lstrip()
    match = re.match(r"[A-Za-z_]+", stripped)
    return match.group(0).lower() if match else ""


def check_statement_is_allowed(statement: str, *, allow_graph_ddl: bool = False) -> None:
    """Refuse a statement this tool does not offer, saying which part is the reason.

    ``allow_graph_ddl`` opens the statements that change a graph's shape rather than its
    contents. Off unless a server is told otherwise, so an ingest tool cannot drop a graph.

    Read with strings and comments blanked, so a value that happens to read like a statement is
    not mistaken for one. This does not decide whether a statement writes -- the driver's
    ``writable_counters`` does that, and the read-only transaction enforces it.
    """
    bare = without_literals(statement).strip()
    if not bare:
        raise PolicyRefusal("this is not a statement")

    if _PRIVILEGE.match(bare):
        raise PolicyRefusal(
            "a statement about privileges is not something this tool offers, whatever else it "
            "is configured to allow. Grant what the connecting role should have outside the "
            "server, once."
        )

    if _LEAVES_THE_DATABASE.search(bare):
        raise PolicyRefusal(
            "COPY reaches outside the database -- to a file, or with TO PROGRAM to a command on "
            "the server's host -- which a read-only transaction does not stop. Read the rows and "
            "write them wherever they are wanted from there."
        )

    if GRAPH_DDL.match(bare):
        if not allow_graph_ddl:
            raise PolicyRefusal(
                "changing the shape of a graph -- its graphs, labels, indexes or constraints -- "
                "is offered only by a server started with --allow-graph-ddl. This one writes to "
                "the graph rather than reshapes it."
            )
        return

    lead = _first_word(bare)
    if lead not in _GRAPH_LEAD:
        raise PolicyRefusal(
            f"a statement beginning {lead.upper() or '(nothing)'} is SQL rather than a graph "
            f"statement, and this tool takes Cypher and GQL. The graph endpoint accepts SQL "
            f"because the server is PostgreSQL underneath, which is not a reason to offer it."
        )

    tell = _SQL_TELL.get(lead)
    if tell is not None and tell.match(bare):
        raise PolicyRefusal(
            f"{lead.upper()} here names a table, so this is SQL rather than a graph statement. "
            f"The graph spelling names a pattern: {lead.upper()} (n:Label {{...}})."
        )
