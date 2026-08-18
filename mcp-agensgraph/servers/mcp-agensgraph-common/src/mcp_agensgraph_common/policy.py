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
#
# SET, REMOVE, DELETE and DETACH are not among them, though they are graph clauses: each needs a
# variable something earlier bound, so each is a syntax error as a first word and every statement
# holding one leads with MATCH or MERGE instead. Reading them as graph leads is what admitted
# `SET work_mem` and `SET search_path`, which committed and were then inherited by the next
# caller to borrow that connection.
_GRAPH_LEAD = (
    "match", "optional", "create", "insert", "merge",
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
#
# Two patterns rather than one alternation. An alternation beginning with an anchor has no literal
# prefix for the scanner to skip ahead on, so it walks the whole statement: measured, one combined
# pattern cost 244 microseconds of a 277-microsecond gate on a four-kilobyte statement. The anchored
# half now stops at the first word, and the half that has to be searched for is reached only by a
# statement that holds the word at all.
_COPIES = re.compile(r"\Acopy\b", re.IGNORECASE)
_RUNS_A_PROGRAM = re.compile(r"\b(?:to|from)\s+program\b", re.IGNORECASE)

# Changing the shape of the graph rather than what is in it. Available, but only to a server
# told to offer it, because a tool that ingests data does not need to drop a graph.
GRAPH_DDL = re.compile(
    r"\A(?:create|drop|alter)\s+(?:graph|vlabel|elabel|label|property\s+index|constraint)\b"
    r"|\Acreate\s+unique\s+property\s+index\b"
    r"|\Adrop\s+(?:property\s+index|constraint)\b",
    re.IGNORECASE,
)


# SQL that a graph lead word can be put in front of. `EXPLAIN` takes any statement at all, and
# `WITH ... AS (...)` takes a data-modifying statement in the bracket and returns its rows -- so
# reading only the first word let `EXPLAIN ANALYZE UPDATE t SET ...`, `WITH d AS (DELETE FROM t
# RETURNING v) SELECT v FROM d` and `EXPLAIN ANALYZE CREATE TABLE x AS SELECT 1` through, each
# measured landing its effect. A graph statement never holds any of these words, so finding one
# anywhere outside a literal is enough.
#
# A name is not a statement, and both positions a graph statement puts one in are excluded: a
# label or a type follows a colon, a property key is followed by one, and an alias follows AS.
# The server takes a reserved word in all three, so reading `MATCH (n:Cluster)` or
# `{select: 1, truncate: 2}` as SQL turns a legitimate query into a confusing refusal.
#
# MERGE INTO is here as well as being refused by the pattern check below, because the pattern
# check reads the clause a statement leads with and a CTE puts WITH there: `WITH s(a) AS
# (VALUES ('x')) MERGE INTO t USING s ...` reached the server and both removed rows and added
# them. The other data-modifying statements were already caught in that position by their own
# spellings here.
_SQL_ANYWHERE = re.compile(
    r"(?<![A-Za-z0-9_.\":{])(?<!as )(?:"
    r"select|insert\s+into|merge\s+into|update\s+[A-Za-z_\"][\w.\"]*\s+set|delete\s+from|truncate|"
    r"alter|grant|revoke|vacuum|analyze\s+[A-Za-z_\"]|reindex|cluster|comment\s+on|"
    r"create\s+(?:or\s+replace\s+)?(?:table|view|schema|function|procedure|extension|database|"
    r"type|sequence|trigger|rule|domain|operator|aggregate|server|publication|subscription|"
    r"materialized|temp|temporary|unlogged|foreign|index|unique\s+index)|"
    r"drop\s+(?:table|view|schema|function|procedure|extension|database|type|sequence|trigger|"
    r"rule|domain|operator|aggregate|server|publication|subscription|index)"
    r")(?![A-Za-z0-9_])(?!\s*:)",
    re.IGNORECASE,
)


# The three clauses that name what they act on, and the shape a graph one has. Each takes a
# pattern, so the next thing is a bracket -- or a variable naming the path, and then a bracket.
# SQL's spelling of each names a table there instead.
#
# Asked positively, because the question "is this SQL" cannot be answered by listing SQL. The
# list missed MERGE, which is a graph clause and, since PostgreSQL 15, a SQL statement that
# updates, deletes and inserts: `MERGE INTO t USING s ON ... WHEN MATCHED THEN DELETE` was
# accepted and ran. Whatever SQL gains next under a word a graph statement shares would be
# missed the same way; a pattern is a thing a graph statement always has.
#
# The variable naming the path may be quoted, and a quoted name is blanked to spaces before this
# reads the statement, so it is optional here for the same reason it is optional above: the
# server accepts `CREATE "p" = (a)-[:R]->(b)` and requiring a name to be visible refused it.
# The words every branch above can start with. An alternation has no literal prefix for the
# scanner to skip ahead on, so it is attempted at each of the many positions the lookbehind
# succeeds at -- measured as 72% of the gate on a four-kilobyte statement, and 86 microseconds a
# kilobyte. A statement holding none of these words cannot match any branch, and answering that
# by looking for substrings is what the scanner is fast at: 10.2x on a statement dense in
# patterns, 3.9x on one dense in projections.
_SQL_WORDS = (
    "select", "insert", "merge", "update", "delete", "truncate", "alter", "grant", "revoke",
    "vacuum", "analyze", "reindex", "cluster", "comment", "create", "drop",
)


_TAKES_A_PATTERN = re.compile(
    r"\A(?:merge|create|insert)\s*(?:[A-Za-z_]\w*\s*)?(?:=\s*)?\(", re.IGNORECASE
)


# What EXPLAIN is put in front of. It takes any statement at all, so the statement being judged
# is the one after it -- `EXPLAIN ANALYZE MERGE INTO t ...` is a MERGE, not an EXPLAIN.
_EXPLAINS = re.compile(
    r"\Aexplain\s*(?:\((?:[^()]|\([^()]*\))*\)\s*)?(?:analyz[es]e?\s+|verbose\s+|costs\s+)*",
    re.IGNORECASE,
)


# A CTE, and only a CTE. `WITH <name> AS (` is SQL: Cypher's WITH projects expressions and puts
# the name after AS, where a bracket cannot go, so the two do not overlap.
#
# The name is optional because it may not be there to read. Literals and quoted identifiers are
# blanked to spaces before any of this, so `WITH "s" AS (...)` arrives with nothing where the
# name was -- and requiring one meant the definitions were never stepped over and the statement
# behind them never judged, which returned every row of `pg_roles` through a read. Anything that
# recognises a statement by an identifier can be evaded by quoting that identifier, so a name
# that has been blanked away still counts as one.
_OPENS_A_CTE = re.compile(
    r"\Awith\s+(?:recursive\s+)?(?:[A-Za-z_]\w*\s*)?(?:\([^()]*\)\s*)?as\s*"
    r"(?:(?:not\s+)?materialized\s*)?\(",
    re.IGNORECASE,
)

_MORE_CTES = re.compile(
    r"\A\s*,\s*(?:[A-Za-z_]\w*\s*)?(?:\([^()]*\)\s*)?as\s*(?:(?:not\s+)?materialized\s*)?\(",
    re.IGNORECASE,
)


def _past_ctes(bare: str) -> str:
    """The statement the CTEs are in front of, or the statement itself.

    A CTE puts WITH at the front of anything SQL can end with, so reading the first word asks
    about the CTE rather than about what it feeds. That is how `WITH s(a) AS (VALUES ('x')) MERGE
    INTO t ...` ran, and `WITH s(a) AS (VALUES (1)) TABLE pg_roles` after it -- one is not a
    reason to name the other, since the list of what can follow is the whole of SQL.
    """
    seen = _OPENS_A_CTE.match(bare)
    while seen is not None:
        depth, index = 1, seen.end()
        while index < len(bare) and depth:
            depth += (bare[index] == "(") - (bare[index] == ")")
            index += 1
        if depth:
            return bare[seen.end():]
        bare = bare[index:]
        seen = _MORE_CTES.match(bare)
        if seen is None:
            return bare.lstrip()
    return bare


def _past_explain(bare: str) -> str:
    """The statement EXPLAIN was asked about, or the statement itself."""
    seen = _EXPLAINS.match(bare)
    return bare[seen.end():].lstrip() if seen else bare


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

    if _COPIES.match(bare) or ("program" in bare.lower() and _RUNS_A_PROGRAM.search(bare)):
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

    lowered = bare.lower()
    found = _SQL_ANYWHERE.search(bare) if any(w in lowered for w in _SQL_WORDS) else None
    if found is not None:
        raise PolicyRefusal(
            f"this holds {' '.join(found.group(0).split()).upper()}, which is SQL rather than a "
            f"graph clause, so the statement is SQL whatever it begins with. A lead word a graph "
            f"statement shares -- EXPLAIN, WITH, CALL -- does not make what follows one."
        )

    explained = _past_ctes(_past_explain(bare))
    inner = _first_word(explained)
    if inner != lead and inner not in _GRAPH_LEAD:
        raise PolicyRefusal(
            f"what these definitions feed begins {inner.upper() or '(nothing)'}, which is SQL "
            f"rather than a graph statement. WITH names a query before whatever SQL ends with, "
            f"so what it is in front of is the statement being asked for."
        )
    if inner in ("merge", "create", "insert") and not _TAKES_A_PATTERN.match(explained):
        raise PolicyRefusal(
            f"{inner.upper()} here does not take a pattern, so it names a table rather than a "
            f"graph element and is SQL. The graph spelling is {inner.upper()} (n:Label {{...}})."
        )

    tell = _SQL_TELL.get(lead)
    if tell is not None and tell.match(bare):
        raise PolicyRefusal(
            f"{lead.upper()} here names a table, so this is SQL rather than a graph statement. "
            f"The graph spelling names a pattern: {lead.upper()} (n:Label {{...}})."
        )
