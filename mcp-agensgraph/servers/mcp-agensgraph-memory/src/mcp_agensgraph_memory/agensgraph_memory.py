"""The knowledge-graph memory itself: entities, relationships and observations.

Every tool call is one connection out of a pool that is already reading the memory graph, and
every operation over a batch is one statement rather than one statement per item. Both are
measured: selecting the graph per statement doubled the round trips, and reading a list into
Python to filter it and write the whole list back lost seven of eight concurrent deletions.
"""

from __future__ import annotations

import asyncio
import logging
import unicodedata
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence

import agensgraph
from agensgraph import AsyncConnection, AsyncConnectionPool, RetryPolicy, TokenBucket
from agensgraph.cypher import quote_identifier
from agensgraph.introspect import MAX_IDENTIFIER
from psycopg.types.json import Jsonb
from pydantic import BaseModel, Field, field_validator

from .bootstrap import (
    MEMORY_LABEL,
    RELATION_VOCABULARY,
    BootstrapReport,
    ensure_edge_uniqueness,
    fulltext_expression,
)

logger = logging.getLogger("mcp_agensgraph_memory")
logger.setLevel(logging.INFO)

MEMORY = quote_identifier(MEMORY_LABEL)

MAX_LIMIT = 1000
"""The most entities any read returns.

An unbounded one was measured returning 20,100 entities as 3.8 MB of JSON, which is about a
million tokens of a caller's context spent on a single call.
"""

MERGE_ATTEMPTS = 8
"""How many times a write that only makes what is missing is tried.

A caller that loses the race writes nothing and has to run again, so the number of attempts
has to cover the number of callers that can be writing the same names at once, not a fixed
small number: eight callers over twenty-five shared names were measured needing more than
four.
"""

MERGE_ALLOWANCE = 2000
"""The size of this server's retry allowance.

A merge conflict is another caller having succeeded, which is contention rather than a server
in trouble, and the driver's default allowance is drained by about four of them.
"""

def canonical_name(value: str) -> str:
    """One spelling of a name, whichever way the letters were composed.

    The same visible name has more than one encoding: an accented letter can be one code point
    or a letter followed by a combining mark, and macOS hands over the second when text is
    pasted. They are different strings, so they were different entities -- two rows both reading
    ``Cafe`` with an accent, which the unique index on the name cannot collapse because it is
    doing exactly what it was asked. Looking one up found whichever spelling was asked for and
    left the other unreachable.

    Composed form on the way in, so a name identifies an element the way it looks rather than
    the way it was typed. The same reasoning as folding case on a relationship type: one thing
    the caller sees, one thing the graph holds.
    """
    return unicodedata.normalize("NFC", value)


def canonical_relation_type(relation_type: str) -> str:
    """One spelling of a relationship type, whatever case it was written in.

    A label is quoted, so ``WORKS_AT``, ``Works_At`` and ``works_at`` would otherwise be three
    labels backed by three tables -- and a delete naming the case the caller did not write
    matches nothing and reports success. Upper case is the spelling the tools document and the
    one every example uses.

    What a type may be is what the server will store. Asked of the server rather than assumed:
    a quoted label takes spaces, accents, other scripts and emoji, and every one of them came
    back spelled the way it was written. Requiring ASCII letters instead meant ``'A B'``,
    ``'RELE'`` with an accent and a Hangul type were refused although the server takes all
    three, so a deployment not writing in English could not name a relationship at all.

    Two things it will not store. A null byte cannot go in an identifier, which the driver's
    quoting already refuses. And an identifier is cut to 63 **bytes** -- past that the server
    does not complain but truncates, and the next type sharing those bytes comes back as
    ``DuplicateTable`` naming a label the caller never wrote. That is measured after upper
    casing, because upper casing can lengthen: ``'\u01f0'`` is two bytes and ``'J\u030c'`` is three.
    """
    if not isinstance(relation_type, str) or not relation_type:
        raise ValueError(f"a relationship type is a name, not {relation_type!r}")
    if "\x00" in relation_type:
        raise ValueError("a relationship type cannot hold a null byte")
    canonical = canonical_name(relation_type).upper()
    if len(canonical.encode("utf-8")) > MAX_IDENTIFIER:
        raise ValueError(
            f"a relationship type is at most {MAX_IDENTIFIER} bytes once upper-cased, and "
            f"{relation_type!r} is {len(canonical.encode('utf-8'))}. A longer one is cut to "
            f"that length by the server rather than refused, so two would collide."
        )
    return canonical


UNTYPED = "unknown"
"""What an element with no type of its own is read as.

An entity is written with one, but a graph can hold an element that has none -- one written by
something other than these tools, or one merged out of copies that all lacked it. A read that
refused such an element would refuse the whole page it is on, so one element with no type would
be a memory that cannot be read at all.
"""


def _unique(values: Iterable[str]) -> list[str]:
    """The values composed one way, each kept once, in the order they were given.

    Names arrive here from a caller that may have typed them any way, so they are put in the
    same form the write path stored them in -- two spellings of one visible name would otherwise
    look up as two different elements.

    Observations go through here too, and want the same thing for the same reason: deleting one
    names the text to remove and matches it exactly, so an observation stored one way and named
    the other would not be found, and the caller would be told it had been removed.
    """
    seen: dict[str, None] = {}
    for value in map(canonical_name, values):
        seen.setdefault(value, None)
    return list(seen)


# Models for our knowledge graph
class Entity(BaseModel):
    """Represents a memory entity in the knowledge graph.

    Example:
    {
        "name": "John Smith",
        "type": "person",
        "observations": ["Works at SKAI Worldwide", "Lives in San Francisco", "Expert in graph databases"]
    }
    """

    updated: Optional[str] = Field(
        default=None,
        description=(
            "When this entity was last written, UTC and ISO-8601. Absent for one written "
            "before the memory recorded it."
        ),
    )

    name: str = Field(
        description="Unique identifier/name for the entity. Should be descriptive and specific.",
        min_length=1,
        examples=["John Smith", "SKAI Worldwide Inc", "San Francisco"],
    )
    type: str = Field(
        description="Category or classification of the entity. Common types: 'person', 'company', 'location', 'concept', 'event'",
        min_length=1,
        examples=["person", "company", "location", "concept", "event"],
    )
    observations: List[str] = Field(
        description="List of facts, observations, or notes about this entity. Each observation should be a complete, standalone fact.",
        examples=[
            ["Works at SKAI Worldwide", "Lives in San Francisco"],
            ["Headquartered in Sweden", "Graph database company"],
        ],
    )

    @field_validator("name", mode="after")
    @classmethod
    def _one_spelling(cls, value: str) -> str:
        """A name identifies an element the way it looks, not the way it was composed."""
        return canonical_name(value)


class Relation(BaseModel):
    """Represents a relationship between two entities in the knowledge graph.

    Example:
    {
        "source": "John Smith",
        "target": "SKAI Worldwide Inc",
        "relationType": "WORKS_AT"
    }
    """

    source: str = Field(
        description="Name of the source entity (must match an existing entity name exactly)",
        min_length=1,
        examples=["John Smith", "SKAI Worldwide Inc"],
    )
    target: str = Field(
        description="Name of the target entity (must match an existing entity name exactly)",
        min_length=1,
        examples=["SKAI Worldwide Inc", "San Francisco"],
    )
    relationType: str = Field(
        description=(
            "Type of relationship between source and target, as letters, digits and "
            "underscores. It is stored upper-cased, so a type given in any other case names "
            "the same relationship."
        ),
        min_length=1,
        examples=["WORKS_AT", "LIVES_IN", "MANAGES", "COLLABORATES_WITH", "LOCATED_IN"],
    )

    @field_validator("source", "target", mode="after")
    @classmethod
    def _one_spelling(cls, value: str) -> str:
        """A name identifies an element the way it looks, not the way it was composed."""
        return canonical_name(value)


class KnowledgeGraph(BaseModel):
    """Complete knowledge graph containing entities and their relationships."""

    entities: List[Entity] = Field(
        description="List of all entities in the knowledge graph", default=[]
    )
    relations: List[Relation] = Field(
        description="List of all relationships between entities", default=[]
    )
    truncated: bool = Field(
        default=False,
        description=(
            "True if the entity list was capped by the limit. Narrow with "
            "search_memories or request a higher limit to see more."
        ),
    )


class ObservationAddition(BaseModel):
    """Request to add new observations to an existing entity.

    Example:
    {
        "entityName": "John Smith",
        "observations": ["Recently promoted to Senior Engineer", "Speaks fluent German"]
    }
    """

    entityName: str = Field(
        description="Exact name of the existing entity to add observations to",
        min_length=1,
        examples=["John Smith", "SKAI Worldwide Inc"],
    )
    observations: List[str] = Field(
        description="New observations/facts to add to the entity. Each should be unique and informative.",
        min_length=1,
    )

    @field_validator("entityName", mode="after")
    @classmethod
    def _one_spelling(cls, value: str) -> str:
        """A name identifies an element the way it looks, not the way it was composed."""
        return canonical_name(value)


class ObservationDeletion(BaseModel):
    """Request to delete specific observations from an existing entity.

    Example:
    {
        "entityName": "John Smith",
        "observations": ["Old job title", "Outdated contact info"]
    }
    """

    entityName: str = Field(
        description="Exact name of the existing entity to remove observations from",
        min_length=1,
        examples=["John Smith", "SKAI Worldwide Inc"],
    )
    observations: List[str] = Field(
        description="Exact observation texts to delete from the entity (must match existing observations exactly)",
        min_length=1,
    )

    @field_validator("entityName", mode="after")
    @classmethod
    def _one_spelling(cls, value: str) -> str:
        """A name identifies an element the way it looks, not the way it was composed."""
        return canonical_name(value)


BY_NAME = "name"
"""A page in name order, which is what the unique index on the name already holds.

The default because it is the one ordering this graph can serve from an index: measured on
twenty thousand entities, an index scan at 0.094 ms against 31.4 ms to sort by anything else.
"""

BY_RECENCY = "recent"
"""A page of what was written most recently, newest first.

What a capped read of a memory often wants -- the alphabetically-first hundred is an arbitrary
hundred. Starting up indexes it, so asking for it costs 8.6 ms on twenty thousand entities
rather than the 195 ms it takes to sort a whole label, and the index costs 1.24x on a write:
0.566 ms against 0.704 measured server-side on a create of twenty.

Name is still what a page comes back in when nothing is said, because it is the ordering the
unique key already provides and it is cheaper again -- 0.238 ms at plan level. Neither ordering
taxes the other.
"""

_SORT_KEYS = {BY_NAME: "entity.name", BY_RECENCY: "entity.updated DESC, entity.name"}

def _sort_key(order: str) -> str:
    """What to sort a page by, refusing anything a caller invented."""
    try:
        return _SORT_KEYS[order]
    except KeyError:
        raise ValueError(
            f"a page is ordered {BY_NAME!r} or {BY_RECENCY!r}, not {order!r}"
        ) from None


def written_now() -> str:
    """When a write is happening, as a string that sorts the way time runs.

    UTC and ISO-8601, so comparing two of them as text puts them in the order they were written
    -- a local offset would not, since the same instant written in two zones compares wrong.

    Read from this process rather than the server. The server's clock is available to a Cypher
    statement as ``timezone('UTC', now())`` and to nothing else: writing an entity goes through
    ``upsert_vertices``, which carries values rather than expressions, so a server-side stamp
    would be available to some write paths and not others. Two clocks ordering one store is
    worse than one clock that is not the database's, and asking the server for the time costs a
    round trip on every write.
    """
    return datetime.now(tz=timezone.utc).isoformat()


def _entity(row: Sequence[Any]) -> Entity:
    """An entity out of a name, a type, a list of observations and when it was last written."""
    return Entity(
        name=row[0],
        type=row[1] or UNTYPED,
        observations=row[2] or [],
        updated=row[3] if len(row) > 3 else None,
    )


ENTITIES_BY_NAME = f"""
    UNWIND %(names)s AS nm
    MATCH (e:{MEMORY} {{name: nm}})
    RETURN e.name AS name, e.type AS type, e.observations AS observations,
           e.updated AS updated
"""

RELATIONS_WITHIN = f"""
    UNWIND %(names)s AS nm
    MATCH (source:{MEMORY} {{name: nm}})-[r]->(target:{MEMORY})
    WHERE target.name IN %(names)s
    RETURN source.name AS source, target.name AS target, label(r) AS "relationType"
"""

RELATIONS_OUT = f"""
    UNWIND %(names)s AS nm
    MATCH (source:{MEMORY} {{name: nm}})-[r]->(target:{MEMORY})
    RETURN source.name AS source, target.name AS target, label(r) AS "relationType",
           target.name AS other, target.type AS "otherType",
           target.observations AS "otherObservations"
    LIMIT %(cap)s
"""

RELATIONS_IN = f"""
    UNWIND %(names)s AS nm
    MATCH (source:{MEMORY})-[r]->(target:{MEMORY} {{name: nm}})
    RETURN source.name AS source, target.name AS target, label(r) AS "relationType",
           source.name AS other, source.type AS "otherType",
           source.observations AS "otherObservations"
    LIMIT %(cap)s
"""

ADD_OBSERVATIONS = f"""
    UNWIND %(batch)s AS row
    MATCH (e:{MEMORY} {{name: row.name}})
    WITH e, row, [o IN row.observations WHERE NOT o IN coalesce(e.observations, [])] AS added
    SET e.observations = coalesce(e.observations, []) + added, e.updated = %(now)s
    RETURN row.name AS name, added
"""

DELETE_OBSERVATIONS = f"""
    UNWIND %(batch)s AS row
    MATCH (e:{MEMORY} {{name: row.name}})
    WITH e, row,
         [o IN coalesce(e.observations, []) WHERE o IN row.observations] AS removed,
         [o IN coalesce(e.observations, []) WHERE NOT o IN row.observations] AS kept
    SET e.observations = kept, e.updated = %(now)s
    RETURN row.name AS name, removed
"""

DELETE_ENTITIES = f"""
    UNWIND %(names)s AS nm
    MATCH (e:{MEMORY} {{name: nm}})
    DETACH DELETE e
"""


def _merge_edges(relation_type: str) -> str:
    """Write the relationships of one type that are not there, and say which ones matched."""
    return f"""
        UNWIND %(pairs)s AS p
        MATCH (s:{MEMORY} {{name: p.source}})
        MATCH (t:{MEMORY} {{name: p.target}})
        MERGE (s)-[:{quote_identifier(relation_type)}]->(t)
        RETURN p.source AS source, p.target AS target
    """


def _delete_edges(relation_type: str) -> str:
    """Remove the relationships of one type between the pairs given."""
    return f"""
        UNWIND %(pairs)s AS p
        MATCH (s:{MEMORY} {{name: p.source}})-[r:{quote_identifier(relation_type)}]->
              (t:{MEMORY} {{name: p.target}})
        DELETE r
    """


class AgensGraphMemory:
    """The memory, over a pool whose connections already read the memory graph."""

    def __init__(
        self,
        connection_pool: AsyncConnectionPool,
        graphname: str,
        *,
        max_limit: int = MAX_LIMIT,
    ) -> None:
        self.pool = connection_pool
        self.graphname = graphname
        self.max_limit = max(1, int(max_limit))
        # An allowance of this server's own, and a large one. The driver's default is shared by
        # every policy in the process that does not ask for one, is spent by about four
        # failures, and is refilled a token at a time -- so a server sharing it stops retrying
        # after the first burst of contention and does not start again. Measured with the
        # default: eight callers writing twenty-five shared names left three of them raising a
        # duplicate key for work that had been done.
        self._allowance = TokenBucket(capacity=MERGE_ALLOWANCE)
        self._declared = set(RELATION_VOCABULARY)

    def _page(self, limit: Optional[int]) -> int:
        """How many entities a read may return, whatever it asked for."""
        if limit is None:
            return self.max_limit
        return max(1, min(int(limit), self.max_limit))

    async def _read(self, statement: str, params: Optional[dict] = None) -> list[Any]:
        """Run one read and return its rows.

        A read ends by rolling back rather than committing: it wrote nothing, and a connection
        going back to the pool with an open transaction holds a snapshot for whoever borrows it
        next.
        """
        async with self.pool.connection() as conn:
            result = await conn.execute_query(statement, params)
            await conn.rollback()
            return result.records

    async def _write(self, work: Any, *, merging: bool = False) -> Any:
        """Run a write, and run it again when another writer got there first.

        ``merging`` says the statements only make what is missing, which is what turns a
        duplicate key or a label another writer created underneath this one from a failure into
        a reason to look again: eight writers over twenty-five shared names were measured
        reporting seven failures for work that had in fact been done.

        Re-selecting the graph after the rollback is not optional. Selecting one is a statement
        inside the transaction, so rolling back returns the session to wherever it was and the
        next statement fails with no graph path at all.
        """
        policy = RetryPolicy(attempts=MERGE_ATTEMPTS if merging else 3, bucket=self._allowance)
        number = 1
        async with self.pool.connection() as conn:
            while True:
                try:
                    outcome = await work(conn)
                    await conn.commit()
                    policy.succeeded()
                    return outcome
                except Exception as exc:
                    await conn.rollback()
                    await conn.graph(self.graphname)
                    attempt = policy.decide(exc, number=number, wrote=True, merging=merging)
                    if not attempt.retry:
                        raise
                    logger.info("Retrying a memory write: %s", attempt.reason)
                    await asyncio.sleep(attempt.delay)
                    number += 1

    # -- reads ---------------------------------------------------------------------------

    async def load_graph(
        self,
        filter_query: Optional[str] = None,
        limit: Optional[int] = None,
        order: str = BY_NAME,
    ) -> KnowledgeGraph:
        """A page of the memory: entities, and the relationships between them.

        ``filter_query`` is a full-text search over name, type and observations. Everything is
        matched, not any -- ``plainto_tsquery`` joins the words with AND -- and the words are
        stemmed, so ``engineers`` finds ``engineering`` but ``eng`` finds neither. A stop word on
        its own matches nothing, since the dictionary drops it.

        ``"*"`` asks for everything rather than for a word, and is the same as reading without
        a search at all.

        The page is the first ``limit`` entities by name, and the relationships returned are
        those whose **both** ends are on the page. Anything else would name entities the caller
        was not given: reading with a cap of 100 was measured returning 193 relationships of
        which 186 pointed outside it.
        """
        page = self._page(limit)
        sort = _sort_key(order)
        params: dict[str, Any] = {}
        condition = ""
        if filter_query and filter_query != "*":
            condition = f"WHERE ({fulltext_expression('entity')}) @@ plainto_tsquery('english', %(query)s)"
            params["query"] = filter_query
        # One more than the page, so that a full page can be told from a page with more behind
        # it. The limit is an int this class bounded, so writing it into the statement binds
        # nothing a caller chose.
        rows = await self._read(
            f"""
            MATCH (entity:{MEMORY})
            {condition}
            RETURN entity.name AS name, entity.type AS type,
                   entity.observations AS observations, entity.updated AS updated
            ORDER BY {sort} LIMIT {page + 1}
            """,
            params or None,
        )
        truncated = len(rows) > page
        rows = rows[:page]
        entities = [_entity(row) for row in rows]
        names = [entity.name for entity in entities]
        relations: list[Relation] = []
        if names:
            found = await self._read(RELATIONS_WITHIN, {"names": Jsonb(names)})
            relations = [
                Relation(source=row[0], target=row[1], relationType=row[2]) for row in found
            ]
        logger.info(
            "Read %d entities and %d relations%s",
            len(entities),
            len(relations),
            " (more remain)" if truncated else "",
        )
        return KnowledgeGraph(entities=entities, relations=relations, truncated=truncated)

    async def read_graph(
        self, limit: Optional[int] = None, order: str = BY_NAME
    ) -> KnowledgeGraph:
        """A page of the memory, by name or by what was written most recently."""
        return await self.load_graph(limit=limit, order=order)

    async def search_memories(
        self, query: str, limit: Optional[int] = None, order: str = BY_NAME
    ) -> KnowledgeGraph:
        """The entities a full-text search matches, and the relationships among them."""
        logger.info("Searching memories")
        return await self.load_graph(query, limit=limit, order=order)

    async def _entities_named(self, names: Sequence[str]) -> list[Entity]:
        """The entities with exactly these names, each found through the unique index."""
        rows = await self._read(ENTITIES_BY_NAME, {"names": Jsonb(list(names))})
        return [_entity(row) for row in rows]

    async def find_memories_by_name(
        self, names: List[str], limit: Optional[int] = None
    ) -> KnowledgeGraph:
        """The named entities, the ones they are connected to, and the relationships.

        The connected entities are returned as well as named, so that no relationship points at
        an entity the caller was not given. Each name is looked up through the unique index
        rather than by testing a list against every entity: five names cost 5 index probes
        rather than a scan of the whole label.

        The limit bounds what comes back, not what was asked for. Bounding the names alone left
        the response to whatever those names happened to be connected to -- one entity with
        three hundred neighbours answered with 301 entities and ninety-eight kilobytes, and said
        it had not truncated anything. Names dropped for being past the limit were not reported
        either, so a caller was told about entities it had not asked after and not told about
        the ones it had.

        The cut is made by the server. Reading every relationship and dropping most of them
        costs the whole neighbourhood in bandwidth to answer with a page of it.
        """
        page = self._page(limit)
        asked = _unique(names)
        wanted = asked[:page]
        if not wanted:
            return KnowledgeGraph(truncated=bool(asked))
        # One row past the page is what tells us more remain without reading them all.
        bound = {"names": Jsonb(wanted), "cap": page + 1}
        entities = {
            entity.name: entity for entity in await self._entities_named(wanted)
        }
        relations: dict[tuple[str, str, str], Relation] = {}
        truncated = len(asked) > page
        for statement in (RELATIONS_OUT, RELATIONS_IN):
            rows = await self._read(statement, bound)
            if len(rows) > page:
                truncated = True
                rows = rows[:page]
            for row in rows:
                relations[(row[0], row[1], row[2])] = Relation(
                    source=row[0], target=row[1], relationType=row[2]
                )
                if row[3] not in entities:
                    entities[row[3]] = _entity(row[3:])
        logger.info(
            "Found %d entities and %d relations%s",
            len(entities),
            len(relations),
            " (more remain)" if truncated else "",
        )
        return KnowledgeGraph(
            entities=list(entities.values()),
            relations=list(relations.values()),
            truncated=truncated,
        )

    # -- writes --------------------------------------------------------------------------

    async def create_entities(self, entities: List[Entity]) -> List[Entity]:
        """Write the entities that are not there and merge into the ones that are.

        An entity already in the memory keeps its observations and gains the ones given, which
        is what "create" has to mean for a store a model writes to repeatedly: replacing them
        loses whatever it recorded on an earlier turn.

        Returns the entities as they now stand, read back, rather than the request.
        """
        wanted = list(entities)
        if not wanted:
            return []
        logger.info("Writing %d entities", len(wanted))
        now = written_now()
        rows = [
            {"name": entity.name, "type": entity.type, "updated": now} for entity in wanted
        ]
        batch = [
            {"name": entity.name, "observations": _unique(entity.observations)}
            for entity in wanted
        ]

        async def work(conn: AsyncConnection) -> None:
            # The type is written by the upsert and the observations by the statement after it,
            # because the upsert would set the property to the list given and the contract is
            # to add to it.
            await conn.upsert_vertices(MEMORY_LABEL, "name", rows, on_existing="update")
            await conn.execute_query(
                ADD_OBSERVATIONS, {"batch": Jsonb(batch), "now": Jsonb(now)}
            )

        await self._write(work, merging=True)
        return await self._entities_named([entity.name for entity in wanted])

    async def create_relations(self, relations: List[Relation]) -> Dict[str, Any]:
        """Write the relationships whose two entities are both there.

        A relationship whose source or target is missing is not written, and is reported as
        such: writing one to a graph that held neither end returned the request as though it had
        been stored.

        Returns ``{"created": [...], "skipped": [...]}``.
        """
        wanted = [
            Relation(
                source=relation.source,
                target=relation.target,
                relationType=canonical_relation_type(relation.relationType),
            )
            for relation in relations
        ]
        if not wanted:
            return {"created": [], "skipped": []}
        logger.info("Writing %d relations", len(wanted))
        by_type: dict[str, list[dict[str, str]]] = {}
        for relation in wanted:
            pairs = by_type.setdefault(relation.relationType, [])
            pair = {"source": relation.source, "target": relation.target}
            if pair not in pairs:
                pairs.append(pair)
        await self._declare(list(by_type))

        async def work(conn: AsyncConnection) -> set[tuple[str, str, str]]:
            written: set[tuple[str, str, str]] = set()
            for relation_type, pairs in by_type.items():
                result = await conn.execute_query(
                    _merge_edges(relation_type), {"pairs": Jsonb(pairs)}
                )
                written.update((row[0], row[1], relation_type) for row in result.records)
            return written

        written = await self._write(work, merging=True)
        created = [r for r in wanted if (r.source, r.target, r.relationType) in written]
        skipped = [r for r in wanted if (r.source, r.target, r.relationType) not in written]
        if skipped:
            logger.info("%d relations name an entity that is not in the memory", len(skipped))
        return {"created": created, "skipped": skipped}

    async def _declare(self, relation_types: Sequence[str]) -> None:
        """Make sure a relationship label exists before a write needs it.

        Writing to a label that is not there makes one, which is DDL inside the write's own
        transaction, and two callers arriving together each see the other's label appear
        underneath them: eight concurrent writers were measured failing six times with
        ``42P07`` before the labels were declared. This runs in a transaction of its own, so
        the write that follows carries no DDL.
        """
        missing = [name for name in relation_types if name not in self._declared]
        if not missing:
            return

        async def work(conn: AsyncConnection) -> None:
            await conn.ensure_labels([agensgraph.DesiredLabel(name, "e") for name in missing])
            # And the index that keeps one relationship of this type between any two entities,
            # which a fresh label has nothing to collapse first.
            await ensure_edge_uniqueness(conn, self.graphname, missing, BootstrapReport())

        # Two callers naming the same new type declare it together, and the one that arrives
        # second is told the label it was about to make is already there. That is the label
        # existing, which is what was asked for, so it is run again and finds nothing to do.
        await self._write(work, merging=True)
        self._declared.update(missing)

    async def add_observations(
        self, observations: List[ObservationAddition]
    ) -> List[Dict[str, Any]]:
        """Add observations to entities already in the memory, in one statement.

        Which of them are new is decided by the server as it writes, not read into Python
        first: eight callers adding the same observation at once against a list they had each
        read beforehand stored eight copies of it.
        """
        batch = [
            {"name": item.entityName, "observations": _unique(item.observations)}
            for item in observations
        ]
        if not batch:
            return []
        logger.info("Adding observations to %d entities", len(batch))

        async def work(conn: AsyncConnection) -> list[Any]:
            result = await conn.execute_query(
                ADD_OBSERVATIONS, {"batch": Jsonb(batch), "now": Jsonb(written_now())}
            )
            return result.records

        rows = await self._write(work)
        added = {row[0]: row[1] for row in rows}
        return [
            {
                "entityName": item["name"],
                "addedObservations": added.get(item["name"], []),
                "found": item["name"] in added,
            }
            for item in batch
        ]

    async def delete_observations(
        self, deletions: List[ObservationDeletion]
    ) -> List[Dict[str, Any]]:
        """Remove observations from entities, in one statement.

        The list is filtered where it is stored. Reading it, filtering it in Python and writing
        the whole list back is a lost update: eight callers each removing a different
        observation left seven of them in place and told all eight it had worked.
        """
        batch = [
            {"name": item.entityName, "observations": _unique(item.observations)}
            for item in deletions
        ]
        if not batch:
            return []
        logger.info("Deleting observations from %d entities", len(batch))

        async def work(conn: AsyncConnection) -> list[Any]:
            result = await conn.execute_query(
                DELETE_OBSERVATIONS, {"batch": Jsonb(batch), "now": Jsonb(written_now())}
            )
            return result.records

        rows = await self._write(work)
        removed = {row[0]: row[1] for row in rows}
        return [
            {
                "entityName": item["name"],
                "deletedObservations": removed.get(item["name"], []),
                "found": item["name"] in removed,
            }
            for item in batch
        ]

    async def delete_entities(self, entity_names: List[str]) -> Dict[str, Any]:
        """Remove entities and everything joined to them, in one statement.

        Returns which names were there and which were not, since a name that is not in the
        memory is not an error and is not a deletion either.
        """
        wanted = _unique(entity_names)
        if not wanted:
            return {"deleted": [], "notFound": [], "deletedRelations": 0}
        logger.info("Deleting %d entities", len(wanted))
        bound = {"names": Jsonb(wanted)}

        async def work(conn: AsyncConnection) -> tuple[list[str], int]:
            rows = await conn.execute_query(ENTITIES_BY_NAME, bound)
            present = [row[0] for row in rows.records]
            result = await conn.execute_query(DELETE_ENTITIES, bound, counts_=True)
            return present, result.counts.deleted_edges or 0

        present, edges = await self._write(work)
        return {
            "deleted": present,
            "notFound": [name for name in wanted if name not in set(present)],
            "deletedRelations": edges,
        }

    async def delete_relations(self, relations: List[Relation]) -> Dict[str, Any]:
        """Remove the relationships named, and report how many there were.

        The type is upper-cased first, so a delete written in another case reaches the
        relationship it names rather than reporting success against a label that does not
        exist.
        """
        by_type: dict[str, list[dict[str, str]]] = {}
        for relation in relations:
            relation_type = canonical_relation_type(relation.relationType)
            pairs = by_type.setdefault(relation_type, [])
            pair = {"source": relation.source, "target": relation.target}
            if pair not in pairs:
                pairs.append(pair)
        requested = sum(len(pairs) for pairs in by_type.values())
        if not by_type:
            return {"requested": 0, "deletedRelations": 0}
        logger.info("Deleting relations of %d types", len(by_type))

        async def work(conn: AsyncConnection) -> int:
            deleted = 0
            for relation_type, pairs in by_type.items():
                # A pattern naming a label the graph does not have matches nothing and raises
                # nothing, so a type nobody ever wrote costs one statement and deletes none.
                result = await conn.execute_query(
                    _delete_edges(relation_type), {"pairs": Jsonb(pairs)}, counts_=True
                )
                deleted += result.counts.deleted_edges or 0
            return deleted

        return {"requested": requested, "deletedRelations": await self._write(work)}
