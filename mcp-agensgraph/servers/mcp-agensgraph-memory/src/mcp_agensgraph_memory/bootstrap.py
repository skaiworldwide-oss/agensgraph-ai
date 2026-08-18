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

from agensgraph import AsyncConnection, AsyncConnectionPool, DesiredIndex, DesiredLabel
from agensgraph.cypher import quote_identifier
from agensgraph.introspect import MAX_IDENTIFIER
from mcp_agensgraph_common.connection import create_pool, ensure_graph
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

RECENCY_INDEX = DesiredIndex(MEMORY_LABEL, ("updated",))
"""What makes a page of the most recently written cost what a page of the first ones does.

Ordering by anything the graph is not indexed for is a scan of the whole label and a top-N sort:
on twenty thousand entities, 195 ms against 8.6 with this index, and 0.4 ms at plan level against
179. Writing costs 1.24x for it, measured server-side on a create of twenty entities -- 0.566 ms
against 0.704 -- which is what an index over a property expression costs to keep.
"""

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

    adopted: bool = False
    """Whether this start was the one that claimed the graph."""

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


OWNERSHIP_LABEL = "MemoryServerMarker"
"""Written the first time a graph is made, and looked for on every start afterwards.

Starting up is destructive: it folds elements sharing a name into one, moves relationships onto a
canonical spelling and drops the label they came off, and imposes one relationship per pair of
endpoints. All of that is justified by owning the data and by nothing else.

Ownership was read from the ``Memory`` label instead, which is a generic name carrying no claim.
An application that had used it first had two of its vertices folded into one and a vertex
destroyed outright, and its relationships adopted and then collapsed. Asking whether this server
made the graph is a question with an answer; asking whether the data looks like this server's is
not.
"""


async def check_ownership(conn: AsyncConnection, graphname: str, *, adopt: bool) -> bool:
    """Refuse a graph holding someone else's data, and say what to do about it.

    Returns whether the marker had to be written, so a caller can report a graph being taken
    over. ``adopt`` is the operator saying the data is theirs after all, which is the only thing
    that can settle it -- a graph another application filled is not distinguishable from this
    one's by reading it.
    """
    labels = {label.name for label in await conn.labels(graph=graphname)}
    if OWNERSHIP_LABEL in labels:
        return False
    # A store an earlier version of this server left has no marker, but it does carry the
    # full-text index this server builds. That is the same claim the marker makes, made by
    # something already there, so an existing store is adopted rather than refused.
    #
    # Asked of the index on the Memory label rather than of a name: `<label>_fulltext_idx` is the
    # ordinary way to name such an index, so another application with a Memory label and a
    # full-text index over it would most likely have named it exactly this. What is checked is
    # the expression, which is this server's own and which nothing else has reason to write --
    # short of another implementation of this same schema, whose index over a name, a type and a
    # list of observations would be built on the same expression and would be adopted. The
    # servers this one is a port of use those three names too, so that is not far-fetched; a
    # graph shared with one of them needs a graph of its own instead.
    built = await conn.execute_query(
        "select 1 from pg_indexes i "
        "join pg_catalog.ag_label l on l.labname = %s "
        "join pg_catalog.ag_graph g on l.graphid = g.oid and g.graphname = %s "
        "where i.schemaname = %s and i.tablename = l.labname "
        "and i.indexdef like %s",
        (MEMORY_LABEL, graphname, graphname, "%to_tsvector%observations%"),
    )
    if built.records:
        await conn.execute_query(f"create vlabel {quote_identifier(OWNERSHIP_LABEL)}")
        return True
    if MEMORY_LABEL in labels and not adopt:
        raise RuntimeError(
            f"graph {graphname!r} already holds a {MEMORY_LABEL!r} label that this server did "
            f"not make. Starting up would fold elements sharing a name into one, move "
            f"relationships onto another label and drop the one they came off -- so it stops "
            f"here instead. Point --graphname at a graph of this server's own, or pass "
            f"--adopt-existing-graph if the data in this one is this server's."
        )
    await conn.execute_query(f"create vlabel {quote_identifier(OWNERSHIP_LABEL)}")
    return True


async def bootstrap(
    pool: AsyncConnectionPool, graphname: str, *, adopt: bool = False
) -> BootstrapReport:
    """Make the labels and indexes the tools write through, migrating what is in the way.

    In this order, and it matters: whose graph it is first, because everything after it rewrites
    what is already there; then the labels, so that the migration can move relationships onto a
    canonical one; the duplicates next, because a unique index cannot be built over a name two
    elements share; then the indexes.
    """
    report = BootstrapReport()
    async with pool.connection() as conn:
        report.adopted = await check_ownership(conn, graphname, adopt=adopt)
        if report.adopted and adopt:
            # Marking is permanent: every start after this one migrates without asking again.
            logger.warning(
                "Graph %r was not made by this server and has been marked as its own because "
                "--adopt-existing-graph was given. Later starts will migrate it without asking.",
                graphname,
            )
        await conn.commit()
        report.statements += await conn.ensure_labels(
            [DesiredLabel(MEMORY_LABEL, "v")]
            + [DesiredLabel(name, "e") for name in RELATION_VOCABULARY]
        )
        await _merge_duplicate_entities(conn, report)
        await _canonicalise_relation_labels(conn, report)
        await conn.commit()

        report.statements += await conn.ensure_indexes([NAME_INDEX, RECENCY_INDEX])
        await ensure_edge_uniqueness(conn, graphname, await edge_labels(conn), report)
        await _install_fulltext_index(conn, report)
        await conn.commit()

        await verify(conn, graphname)
    logger.info("Memory graph %r ready: %s", graphname, report.describe())
    return report


async def edge_labels(conn: AsyncConnection) -> list[str]:
    """The relationship labels this server's own data is held under.

    A graph can hold more than one application's data, and the migrations below rewrite and drop
    what they are given. Reading every label in the graph handed them another application's
    edges: pointed at a graph holding two distinct ``shipsTo`` edges, the run reported relabelling
    one and left one, having destroyed the other.

    So a label counts as this server's when **every** relationship under it joins two ``Memory``
    vertices, which is the only shape any tool here writes. One such relationship is not enough:
    a type carrying one Memory-to-Memory relationship and one reaching elsewhere had all of them
    moved, the second included. The declared vocabulary is included whether or not anything has
    been written under it yet.

    A graph where another application also labels its vertices ``Memory`` cannot be told apart
    from this one's by any of this, so the relationships that are moved are constrained to that
    shape as well, and a label is dropped only once nothing is left under it.
    """
    present = [
        label.name
        for label in await conn.labels()
        if label.is_edge and not label.is_builtin
    ]
    # Reading the label table is a catalog lookup; deciding by shape is a scan of every
    # relationship in the graph -- 0.8 milliseconds against 488 on four hundred thousand of them.
    # A label this server declared needs no deciding, so a store holding only those is answered
    # without the scan, which is every store this server has to itself.
    undeclared = [name for name in present if name not in RELATION_VOCABULARY]
    if not undeclared:
        return present

    # Two counts rather than one grouped by the labels at both ends. Naming those labels reads
    # both vertex tables for every relationship, where a pattern naming Memory restricts by the
    # label's graphid range instead: on two hundred thousand relationships the single grouped
    # count cost 2591 milliseconds against 534 for the pair.
    memory = quote_identifier(MEMORY_LABEL)
    every = await conn.execute_query("MATCH ()-[r]->() RETURN label(r) AS name, count(*) AS n")
    ours = await conn.execute_query(
        f"MATCH (:{memory})-[r]->(:{memory}) RETURN label(r) AS name, count(*) AS n"
    )
    mine = {name: n for name, n in ours.records}
    owned = {name for name, n in every.records if mine.get(name) == n}
    owned.update(RELATION_VOCABULARY)
    return [name for name in present if name in owned]


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
        if not held.records:
            continue
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
    """Move every relationship of one label onto another, properties and all.

    One new relationship per old one. Merging on the pair instead folds every relationship
    between the same two vertices into a single survivor holding the last one's properties, so
    two edges that differ only in what they carry leave one -- measured, a pair distinguished
    only by a property came back as one edge.
    """
    memory = quote_identifier(MEMORY_LABEL)
    return f"""
        MATCH (a:{memory})-[r:{quote_identifier(old)}]->(b:{memory})
        CREATE (a)-[n:{quote_identifier(new)}]->(b)
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
    miscased = [name for name in await edge_labels(conn) if name != name.upper()]
    if not miscased:
        return
    await conn.ensure_labels([DesiredLabel(name.upper(), "e") for name in miscased])
    for name in miscased:
        moved = await conn.execute_query(_relabel(name, name.upper()), counts_=True)
        report.relabelled[name] = moved.counts.inserted_edges or 0
        remaining = await conn.execute_query(
            f"MATCH ()-[r:{quote_identifier(name)}]->() RETURN count(r) AS n"
        )
        if remaining.records[0][0]:
            raise RuntimeError(
                f"moving relationships off {name!r} onto {name.upper()!r} left "
                f"{remaining.records[0][0]} behind, so dropping it would destroy them"
            )
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


DEFAULT_TIMEOUT = 30.0
"""How long any one statement may hold a connection, in seconds.

Without it every one of these tools is unbounded: a write ran for 35 seconds, and a call behind
a lock waited past forty. The connections carry it from the moment they are made.
"""


def make_pool(
    dsn: str, graphname: str, *, read_timeout: float = DEFAULT_TIMEOUT, **kwargs: Any
) -> AsyncConnectionPool:
    """A pool whose connections are already reading the memory graph, under a time limit.

    Selecting a graph is a statement, so doing it per call is a round trip per call -- measured
    as an exact doubling, since every one of these tools sends one statement at a time. The pool
    runs it once for each connection it makes instead. The driver does not read a graph path set
    by hand, so this is also the only way to tell it which graph its label table describes.
    """
    return create_pool(dsn, graphname, read_timeout=read_timeout, **kwargs)
