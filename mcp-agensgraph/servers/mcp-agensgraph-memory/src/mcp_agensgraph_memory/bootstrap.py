"""What the memory graph needs to exist before a tool writes to it.

Three things, all of them made at startup and all of them checked afterwards:

* the ``Memory`` vertex label and the relationship types the tools write to, so that a write
  does not have to make a label and put DDL inside its own transaction;
* a **unique** property index on ``Memory(name)``, which is what makes ``MERGE`` on a name find
  one element rather than make another;
* a full-text index over the properties, built out of PostgreSQL's own functions.

A store written before those existed can hold things they forbid -- several elements sharing a
name, a relationship type spelled in three cases, an index over a function this package
installed -- so each of them is migrated first, and what the migration collapsed is reported.

Nothing here is best-effort. A failure raises, because a server that starts without the unique
index writes duplicates and a server that starts without the label answers every search with an
empty memory.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from hashlib import sha1
from typing import Any, Sequence

import agensgraph
from agensgraph import AsyncConnection, AsyncConnectionPool, DesiredIndex, DesiredLabel
from agensgraph.cypher import quote_identifier
from agensgraph.introspect import MAX_IDENTIFIER
from mcp_agensgraph_common.connection import ensure_graph
from psycopg.types.json import Jsonb

__all__ = [
    "BootstrapReport",
    "MEMORY_LABEL",
    "RELATION_VOCABULARY",
    "bootstrap",
    "edge_index_name",
    "edge_labels",
    "ensure_edge_uniqueness",
    "ensure_graph",
    "fulltext_expression",
    "make_pool",
    "verify",
]

logger = logging.getLogger("mcp_agensgraph_memory")

MEMORY_LABEL = "Memory"
"""The one vertex label the memory graph holds."""

NAME_INDEX = DesiredIndex(MEMORY_LABEL, ("name",), unique=True)
"""What makes a name identify one element.

Without it ``MERGE (e:"Memory" {name: ...})`` is a full label scan that finds nothing when
another writer is halfway through making the same element: eight writers over twenty-five
shared names were measured making 124 elements. With it they make 25, and the writers that
arrive second are told ``23505`` and run the statement again.
"""

FULLTEXT_INDEX = "memory_fulltext_idx"

FULLTEXT_EXPRESSION = """
    to_tsvector('english', coalesce({alias}name, ''))
    || to_tsvector('english', coalesce({alias}type, ''))
    || to_tsvector('english', coalesce({alias}observations, ''))
"""
"""Name, type and every observation as one document.

``to_tsvector(regconfig, jsonb)`` reads the strings out of a jsonb value itself, including the
ones in an array, so the observations need nothing to flatten them first. The expression the
index is built on and the expression a search is written with have to be the same one, which is
why there is a single copy of it with a place for the alias a Cypher pattern gives the element.
"""

RELATION_VOCABULARY = (
    "COLLABORATES_WITH",
    "KNOWS",
    "LIVES_IN",
    "LOCATED_IN",
    "MANAGES",
    "PART_OF",
    "RELATED_TO",
    "WORKS_AT",
)
"""The relationship types declared at startup, being the ones the tools document.

A type outside this list is declared when a call first names it, in a transaction of its own.
Declaring the common ones up front means the usual call does not spend a statement on it.
"""


def fulltext_expression(alias: str = "") -> str:
    """The full-text expression, keyed on an element the caller has named."""
    return FULLTEXT_EXPRESSION.format(alias=f"{alias}." if alias else "")


@dataclass
class BootstrapReport:
    """What starting up found and what it had to change."""

    statements: list[str] = field(default_factory=list)
    """The DDL that was not already there."""

    merged_names: dict[str, int] = field(default_factory=dict)
    """Name -> how many elements shared it, for every name that had more than one."""

    merged_observations: int = 0
    """How many observations the survivors gained from the elements folded into them."""

    conflicting_types: list[str] = field(default_factory=list)
    """Names whose copies disagreed about the entity's type."""

    relabelled: dict[str, int] = field(default_factory=dict)
    """Relationship label -> how many relationships were moved to its canonical spelling."""

    merged_relations: int = 0
    """How many duplicate relationships were folded into the one written first."""

    reindexed: bool = False
    """Whether the full-text index was rebuilt without the installed helper function."""

    def describe(self) -> str:
        """One line an operator can read in the log."""
        parts = [f"{len(self.statements)} statements"]
        if self.merged_names:
            parts.append(
                f"{sum(self.merged_names.values()) - len(self.merged_names)} duplicate entities "
                f"merged into {len(self.merged_names)}"
            )
        if self.merged_observations:
            parts.append(f"{self.merged_observations} observations carried over")
        if self.conflicting_types:
            parts.append(
                f"{len(self.conflicting_types)} of them whose copies disagreed about the type"
            )
        if self.relabelled:
            parts.append(
                f"{sum(self.relabelled.values())} relationships moved off "
                f"{len(self.relabelled)} miscased labels"
            )
        if self.merged_relations:
            parts.append(f"{self.merged_relations} duplicate relationships collapsed")
        if self.reindexed:
            parts.append("full-text index rebuilt")
        return ", ".join(parts)


async def bootstrap(pool: AsyncConnectionPool, graphname: str) -> BootstrapReport:
    """Make the labels and indexes the tools write through, migrating what is in the way.

    In this order, and it matters: the labels first, so that the migration can move
    relationships onto a canonical one; the duplicates next, because a unique index cannot be
    built over a name two elements share; then the indexes.
    """
    report = BootstrapReport()
    async with pool.connection() as conn:
        report.statements += await conn.ensure_labels(
            [DesiredLabel(MEMORY_LABEL, "v")]
            + [DesiredLabel(name, "e") for name in RELATION_VOCABULARY]
        )
        await _merge_duplicate_entities(conn, report)
        await _canonicalise_relation_labels(conn, report)
        await conn.commit()

        report.statements += await conn.ensure_indexes([NAME_INDEX])
        await ensure_edge_uniqueness(conn, graphname, await edge_labels(conn), report)
        await _install_fulltext_index(conn, report)
        await conn.commit()

        await verify(conn, graphname)
    logger.info("Memory graph %r ready: %s", graphname, report.describe())
    return report


async def edge_labels(conn: AsyncConnection) -> list[str]:
    """Every relationship label a caller could have written to."""
    return [label.name for label in await conn.labels() if label.is_edge and not label.is_builtin]


async def verify(conn: AsyncConnection, graphname: str) -> None:
    """Refuse to serve a graph that is missing what every tool depends on.

    A missing ``Memory`` label is not an error the server sees: a pattern naming a label the
    graph does not have returns no rows, so a graph name with a typo in it reads as a memory
    with nothing in it. It is asked about here, once, where it can still be reported.
    """
    labels = {label.name for label in await conn.labels(graph=graphname)}
    if MEMORY_LABEL not in labels:
        raise RuntimeError(
            f"graph {graphname!r} has no {MEMORY_LABEL!r} label, so every read would return "
            f"nothing and report no error"
        )
    unique = [
        index
        for index in await conn.indexes(MEMORY_LABEL, graph=graphname)
        if index.unique and "name" in index.definition
    ]
    if not unique:
        raise RuntimeError(
            f"nothing makes a name unique on {MEMORY_LABEL!r} in graph {graphname!r}, so two "
            f"callers writing the same entity would each make one"
        )


DUPLICATE_NAMES = f"""
    MATCH (e:{quote_identifier(MEMORY_LABEL)})
    WITH e.name AS name, count(*) AS copies
    WHERE copies > 1
    RETURN name, copies
"""

COPIES_OF = f"""
    MATCH (e:{quote_identifier(MEMORY_LABEL)} {{name: %(name)s}})
    RETURN e.type AS type, e.observations AS observations
    ORDER BY id(e)
"""

WRITE_SURVIVOR = f"""
    MATCH (k:{quote_identifier(MEMORY_LABEL)} {{name: %(name)s}})
    WITH k ORDER BY id(k) LIMIT 1
    SET k.type = %(type)s, k.observations = %(observations)s
"""

DROP_COPIES = f"""
    MATCH (k:{quote_identifier(MEMORY_LABEL)} {{name: %(name)s}})
    WITH k ORDER BY id(k) LIMIT 1
    MATCH (d:{quote_identifier(MEMORY_LABEL)} {{name: %(name)s}})
    WHERE id(d) <> id(k)
    DETACH DELETE d
"""


def _relink(label: str, *, outgoing: bool) -> str:
    """Move the relationships of the copies of a name onto the one being kept."""
    memory = quote_identifier(MEMORY_LABEL)
    edge = quote_identifier(label)
    pattern = (
        f"MATCH (d:{memory} {{name: %(name)s}})-[r:{edge}]->(o)"
        if outgoing
        else f"MATCH (o)-[r:{edge}]->(d:{memory} {{name: %(name)s}})"
    )
    merged = f"MERGE (k)-[n:{edge}]->(o)" if outgoing else f"MERGE (o)-[n:{edge}]->(k)"
    return f"""
        MATCH (k:{memory} {{name: %(name)s}})
        WITH k ORDER BY id(k) LIMIT 1
        {pattern}
        WHERE id(d) <> id(k)
        {merged}
        SET n = properties(r)
        DELETE r
    """


async def _merge_duplicate_entities(conn: AsyncConnection, report: BootstrapReport) -> None:
    """Fold every element sharing a name into one, keeping everything they held.

    A unique index cannot be built over a name two elements share, and dropping one of them to
    make room would throw away whatever was written to it. So the observations are unioned in
    the order they were written, the relationships of the copies are moved onto the element
    being kept, and only then are the copies removed. A type the copies disagreed about is
    reported rather than picked silently.
    """
    duplicates = await conn.execute_query(DUPLICATE_NAMES)
    if not duplicates.records:
        return
    edges = await edge_labels(conn)
    logger.warning(
        "%d entity names are held by more than one element; merging them so a name can be "
        "made unique",
        len(duplicates.records),
    )
    for name, copies in duplicates.records:
        report.merged_names[name] = int(copies)
        held = await conn.execute_query(COPIES_OF, {"name": Jsonb(name)})
        observations: list[str] = []
        for _, values in held.records:
            for value in values or []:
                if value not in observations:
                    observations.append(value)
        types = [value for value, _ in held.records if value is not None]
        if len(set(types)) > 1:
            report.conflicting_types.append(name)
        report.merged_observations += len(observations) - len(held.records[0][1] or [])
        for label in edges:
            for outgoing in (True, False):
                await conn.execute_query(_relink(label, outgoing=outgoing), {"name": Jsonb(name)})
        await conn.execute_query(
            WRITE_SURVIVOR,
            {
                "name": Jsonb(name),
                "type": Jsonb(types[0] if types else None),
                "observations": Jsonb(observations),
            },
        )
        await conn.execute_query(DROP_COPIES, {"name": Jsonb(name)})


def _relabel(old: str, new: str) -> str:
    """Move every relationship of one label onto another, properties and all."""
    return f"""
        MATCH (a)-[r:{quote_identifier(old)}]->(b)
        MERGE (a)-[n:{quote_identifier(new)}]->(b)
        SET n = properties(r)
        DELETE r
    """


async def _canonicalise_relation_labels(
    conn: AsyncConnection, report: BootstrapReport
) -> None:
    """Move relationships written under a miscased type onto the canonical spelling.

    A label is quoted, so ``WORKS_AT``, ``Works_At`` and ``works_at`` are three labels backed by
    three tables. Reading returns whichever spelling was written, but a delete naming another
    one matches nothing and reports success -- three spellings of one relationship were measured
    leaving three relationships that no delete could reach.
    """
    miscased = [
        label.name
        for label in await conn.labels()
        if label.is_edge and not label.is_builtin and label.name != label.name.upper()
    ]
    if not miscased:
        return
    await conn.ensure_labels([DesiredLabel(name.upper(), "e") for name in miscased])
    for name in miscased:
        moved = await conn.execute_query(_relabel(name, name.upper()), counts_=True)
        report.relabelled[name] = moved.counts.inserted_edges or 0
        await conn.execute_query(f"drop elabel {quote_identifier(name)}")
    logger.warning("Moved relationships off %s onto their upper-case spelling", miscased)


def edge_index_name(label: str) -> str:
    """What the index keeping one relationship per pair of ends is called.

    An identifier is 63 bytes, and a label may be nearly that on its own, so a name that would
    not fit is cut and given a digest of what was cut -- two labels sharing a prefix would
    otherwise share an index name and the second could not be made.
    """
    name = f"{label}_start_end_uniq"
    if len(name.encode()) <= MAX_IDENTIFIER:
        return name
    digest = sha1(label.encode(), usedforsecurity=False).hexdigest()[:8]
    return f"{label[:40]}_{digest}_start_end_uniq"


DUPLICATE_ENDS = """
    MATCH (a)-[r:{label}]->(b)
    WITH a, b, count(r) AS copies
    WHERE copies > 1
    RETURN id(a) AS a, id(b) AS b, copies
"""

RELATIONS_BETWEEN = """
    MATCH (a)-[r:{label}]->(b)
    WHERE id(a) = %(a)s AND id(b) = %(b)s
    RETURN id(r) AS id
    ORDER BY id(r)
"""

DROP_RELATIONS = """
    MATCH ()-[r:{label}]->()
    WHERE id(r) IN %(ids)s
    DELETE r
"""


async def ensure_edge_uniqueness(
    conn: AsyncConnection, graphname: str, labels: Sequence[str], report: BootstrapReport
) -> None:
    """Keep one relationship of a type between any two entities.

    ``MERGE`` finds a relationship or makes one, and with nothing enforcing that there is only
    one, two callers arriving together each find none and each make one: eight callers over
    twenty-five pairs were measured leaving 200 relationships. A relationship's ends are columns
    rather than properties, so the index is written as ordinary DDL -- there is no property
    index that can say this.

    Duplicates already stored are collapsed onto the one written first, keeping its properties,
    because a unique index cannot be built over them.
    """
    for label in labels:
        index = edge_index_name(label)
        found = await conn.execute_query(
            "select 1 from pg_class c join pg_namespace n on n.oid = c.relnamespace "
            "where n.nspname = %s and c.relname = %s",
            (graphname, index),
        )
        if found.records:
            continue
        await _collapse_duplicate_relations(conn, label, report)
        statement = (
            f"create unique index {quote_identifier(index)} on "
            f"{quote_identifier(graphname)}.{quote_identifier(label)} (start, \"end\")"
        )
        await conn.execute_query(statement)
        report.statements.append(statement)


async def _collapse_duplicate_relations(
    conn: AsyncConnection, label: str, report: BootstrapReport
) -> None:
    """Fold every extra relationship between one pair of ends into the first of them."""
    edge = quote_identifier(label)
    pairs = await conn.execute_query(DUPLICATE_ENDS.format(label=edge))
    if not pairs.records:
        return
    extra: list[Any] = []
    for start, end, _ in pairs.records:
        found = await conn.execute_query(
            RELATIONS_BETWEEN.format(label=edge), {"a": start, "b": end}
        )
        extra += [row[0] for row in found.records[1:]]
    await conn.execute_query(DROP_RELATIONS.format(label=edge), {"ids": extra})
    report.merged_relations += len(extra)
    logger.warning(
        "Collapsed %d duplicate %s relationships onto the one written first", len(extra), label
    )


async def _install_fulltext_index(conn: AsyncConnection, report: BootstrapReport) -> None:
    """Build the full-text index out of PostgreSQL's own functions.

    An index whose expression calls a function this package installs is an index PostgreSQL
    cannot know is stale: the function is declared immutable, so replacing its body leaves every
    entry that was built with the old one in place and searches answer from them. Nothing is
    installed now, and an index built over the old helper is rebuilt here -- one that is left
    alone would keep answering from entries no statement can reproduce.
    """
    present = [index for index in await conn.indexes(MEMORY_LABEL) if index.name == FULLTEXT_INDEX]
    stale = [index for index in present if "jsonb_to_string" in index.definition]
    for index in stale:
        await conn.execute_query(f"drop property index {quote_identifier(index.name)}")
        report.reindexed = True
        logger.warning(
            "Rebuilding %s: it was built over an installed helper function, which PostgreSQL "
            "cannot notice a change to",
            index.name,
        )
    if present and not stale:
        return
    statement = (
        f"create property index {quote_identifier(FULLTEXT_INDEX)} "
        f"on {quote_identifier(MEMORY_LABEL)} using gin (({fulltext_expression()}))"
    )
    await conn.execute_query(statement)
    report.statements.append(statement)


def make_pool(dsn: str, graphname: str, **kwargs: object) -> AsyncConnectionPool:
    """A pool whose connections are already reading the memory graph.

    Selecting a graph is a statement, so doing it per call is a round trip per call -- measured
    as an exact doubling, since every one of these tools sends one statement at a time. The pool
    runs it once for each connection it makes instead. The driver does not read a graph path set
    by hand, so this is also the only way to tell it which graph its label table describes.
    """
    return agensgraph.AsyncConnectionPool(dsn, graph=graphname, **kwargs)  # type: ignore[arg-type]
