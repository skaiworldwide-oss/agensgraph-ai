"""What the memory has to hold when more than one caller writes to it at once.

Every number in here was measured against the server before the invariant it asserts existed,
and is written into the test that keeps it.
"""

import asyncio

import psycopg
import pytest

from mcp_agensgraph_memory.agensgraph_memory import (
    AgensGraphMemory,
    Entity,
    ObservationAddition,
    ObservationDeletion,
    Relation,
    canonical_relation_type,
)
from mcp_agensgraph_memory.bootstrap import edge_index_name

WRITERS = 8
KEYS = 25

INDEX_IS_UNIQUE = """
    select i.indisunique
    from pg_index i
    join pg_class c on c.oid = i.indexrelid
    join pg_namespace n on n.oid = c.relnamespace
    where n.nspname = %s and c.relname = %s
"""


async def is_unique(memory: AgensGraphMemory, graphname: str, index: str):
    async with memory.pool.connection() as conn:
        found = await conn.execute_query(INDEX_IS_UNIQUE, (graphname, index))
        await conn.rollback()
    return [row[0] for row in found.records]


async def count(memory: AgensGraphMemory, statement: str) -> int:
    async with memory.pool.connection() as conn:
        result = await conn.execute_query(statement)
        await conn.rollback()
    return int(result.records[0][0]) if result.records else 0


@pytest.mark.asyncio
async def test_writers_sharing_names_leave_one_entity_each(
    memory: AgensGraphMemory, graphname
):
    """Eight callers writing the same twenty-five names left 124 entities and three errors."""
    assert await is_unique(memory, graphname, "Memory_name_idx") == [True], (
        "the unique index is what makes MERGE on a name find rather than make"
    )
    entities = [
        Entity(name=f"K{i:03d}", type="person", observations=[f"obs {i}"])
        for i in range(KEYS)
    ]
    outcomes = await asyncio.gather(
        *(memory.create_entities(list(entities)) for _ in range(WRITERS)),
        return_exceptions=True,
    )
    failures = [o for o in outcomes if isinstance(o, BaseException)]
    assert failures == [], "a caller that lost the race runs again rather than reporting"
    assert await count(memory, 'MATCH (n:"Memory") RETURN count(*)') == KEYS


@pytest.mark.asyncio
async def test_writers_sharing_relations_leave_one_relationship_each(
    memory: AgensGraphMemory, graphname
):
    """The same eight callers left 615 relationships for twenty-five pairs, and seven errors."""
    await memory.create_entities(
        [Entity(name=f"K{i:03d}", type="person", observations=[]) for i in range(KEYS)]
    )
    assert await is_unique(memory, graphname, edge_index_name("KNOWS")) == [True], (
        "one relationship per pair of ends is a unique index on the columns holding them"
    )
    relations = [
        Relation(source=f"K{i:03d}", target=f"K{(i + 1) % KEYS:03d}", relationType="KNOWS")
        for i in range(KEYS)
    ]
    outcomes = await asyncio.gather(
        *(memory.create_relations(list(relations)) for _ in range(WRITERS)),
        return_exceptions=True,
    )
    assert [o for o in outcomes if isinstance(o, BaseException)] == []
    assert await count(memory, "MATCH ()-[r]->() RETURN count(*)") == KEYS


@pytest.mark.asyncio
async def test_a_type_nobody_declared_is_written_once_by_eight_callers(
    memory: AgensGraphMemory, graphname
):
    """Writing to an undeclared label puts DDL in the write: six of eight failed 42P07."""
    await memory.create_entities(
        [
            Entity(name="A1", type="person", observations=[]),
            Entity(name="A2", type="person", observations=[]),
        ]
    )
    outcomes = await asyncio.gather(
        *(
            memory.create_relations(
                [Relation(source="A1", target="A2", relationType="INVENTED_TYPE")]
            )
            for _ in range(WRITERS)
        ),
        return_exceptions=True,
    )
    assert [o for o in outcomes if isinstance(o, BaseException)] == []
    assert await count(memory, 'MATCH ()-[r:"INVENTED_TYPE"]->() RETURN count(*)') == 1
    assert await is_unique(memory, graphname, edge_index_name("INVENTED_TYPE")) == [True]


@pytest.mark.asyncio
async def test_concurrent_deletions_of_distinct_observations_all_apply(
    memory: AgensGraphMemory,
):
    """Eight callers each deleting one of eight observations left seven, and all were told it worked."""
    items = [f"item {i}" for i in range(WRITERS)]
    await memory.create_entities(
        [Entity(name="target", type="thing", observations=list(items))]
    )
    outcomes = await asyncio.gather(
        *(
            memory.delete_observations(
                [ObservationDeletion(entityName="target", observations=[item])]
            )
            for item in items
        )
    )
    assert all(result[0]["found"] for result in outcomes)
    left = await memory.find_memories_by_name(["target"])
    assert left.entities[0].observations == []


@pytest.mark.asyncio
async def test_the_same_observation_added_by_eight_callers_is_stored_once(
    memory: AgensGraphMemory,
):
    """The filter was a read taken before the write, so eight callers stored eight copies."""
    await memory.create_entities([Entity(name="target", type="thing", observations=[])])
    await asyncio.gather(
        *(
            memory.add_observations(
                [ObservationAddition(entityName="target", observations=["same"])]
            )
            for _ in range(WRITERS)
        )
    )
    left = await memory.find_memories_by_name(["target"])
    assert left.entities[0].observations == ["same"]


@pytest.mark.asyncio
async def test_a_write_that_rolled_back_is_still_reading_the_memory_graph(
    memory: AgensGraphMemory, graphname
):
    """Selecting a graph is part of the transaction, so a rollback takes it back with it."""
    seen = []

    async def work(conn):
        seen.append(conn.label_table.graph)
        if len(seen) == 1:
            raise psycopg.errors.UniqueViolation("another writer got there first")
        await conn.execute_query("""CREATE (:"Memory" {name: 'after a rollback'})""")

    await memory._write(work, merging=True)
    assert seen == [graphname, graphname]
    found = await memory.find_memories_by_name(["after a rollback"])
    assert [e.name for e in found.entities] == ["after a rollback"]


@pytest.mark.asyncio
async def test_creating_an_entity_twice_keeps_both_sets_of_observations(
    memory: AgensGraphMemory,
):
    """Writing an entity again replaced its observations, so the first turn's were lost."""
    await memory.create_entities(
        [Entity(name="A1", type="person", observations=["first"])]
    )
    final = await memory.create_entities(
        [Entity(name="A1", type="robot", observations=["second", "first"])]
    )
    assert len(final) == 1
    assert final[0].observations == ["first", "second"]
    assert final[0].type == "robot"
    assert await count(memory, 'MATCH (n:"Memory") RETURN count(*)') == 1


@pytest.mark.asyncio
async def test_a_relation_to_an_entity_that_is_not_there_is_reported(
    memory: AgensGraphMemory,
):
    """A MERGE whose MATCH found nothing returned the request as though it had been stored."""
    await memory.create_entities([Entity(name="A1", type="person", observations=[])])
    result = await memory.create_relations(
        [Relation(source="A1", target="ghost", relationType="KNOWS")]
    )
    assert result["created"] == []
    assert [r.target for r in result["skipped"]] == ["ghost"]
    assert await count(memory, "MATCH ()-[r]->() RETURN count(*)") == 0


@pytest.mark.asyncio
async def test_adding_observations_to_an_entity_that_is_not_there_is_reported(
    memory: AgensGraphMemory,
):
    result = await memory.add_observations(
        [ObservationAddition(entityName="ghost", observations=["x"])]
    )
    assert result == [
        {"entityName": "ghost", "addedObservations": [], "found": False}
    ]


@pytest.mark.asyncio
async def test_one_relationship_type_written_in_three_cases_is_one_label(
    memory: AgensGraphMemory,
):
    """Three spellings were three labels backed by three tables, and a wrong-case delete
    reported success while removing none of them."""
    await memory.create_entities(
        [
            Entity(name="A1", type="person", observations=[]),
            Entity(name="A2", type="company", observations=[]),
        ]
    )
    for spelling in ("WORKS_AT", "Works_At", "works_at"):
        await memory.create_relations(
            [Relation(source="A1", target="A2", relationType=spelling)]
        )
    assert await count(memory, "MATCH ()-[r]->() RETURN count(distinct label(r))") == 1
    assert await count(memory, "MATCH ()-[r]->() RETURN count(*)") == 1

    removed = await memory.delete_relations(
        [Relation(source="A1", target="A2", relationType="works_AT")]
    )
    assert removed == {"requested": 1, "deletedRelations": 1}
    assert await count(memory, "MATCH ()-[r]->() RETURN count(*)") == 0


@pytest.mark.asyncio
async def test_deleting_entities_says_which_names_were_there(memory: AgensGraphMemory):
    await memory.create_entities(
        [
            Entity(name="A1", type="person", observations=[]),
            Entity(name="A2", type="person", observations=[]),
        ]
    )
    await memory.create_relations(
        [Relation(source="A1", target="A2", relationType="KNOWS")]
    )
    result = await memory.delete_entities(["A1", "nobody"])
    assert result["deleted"] == ["A1"]
    assert result["notFound"] == ["nobody"]
    assert result["deletedRelations"] == 1


@pytest.mark.asyncio
async def test_a_capped_read_names_no_entity_it_did_not_return(memory: AgensGraphMemory):
    """The relations query joined the page with OR, so 186 of 193 pointed off it."""
    await memory.create_entities(
        [Entity(name=f"N{i:03d}", type="thing", observations=[]) for i in range(10)]
    )
    await memory.create_relations(
        [
            Relation(source=f"N{i:03d}", target=f"N{i + 5:03d}", relationType="KNOWS")
            for i in range(5)
        ]
    )
    page = await memory.read_graph(limit=5)
    assert page.truncated is True
    names = {entity.name for entity in page.entities}
    assert len(names) == 5
    dangling = [
        r for r in page.relations if r.source not in names or r.target not in names
    ]
    assert dangling == []


@pytest.mark.asyncio
async def test_a_read_is_capped_however_large_a_limit_is_asked_for(
    memory: AgensGraphMemory,
):
    """`read_graph(limit=10**12)` returned 20,100 entities as 3.8 MB of JSON."""
    memory.max_limit = 3
    await memory.create_entities(
        [Entity(name=f"N{i:03d}", type="thing", observations=[]) for i in range(10)]
    )
    page = await memory.read_graph(limit=10**12)
    assert len(page.entities) == 3
    assert page.truncated is True
    named = await memory.find_memories_by_name([f"N{i:03d}" for i in range(10)])
    assert len(named.entities) == 3, "looking names up is capped the same way"


@pytest.mark.asyncio
async def test_finding_by_name_returns_the_entities_its_relations_name(
    memory: AgensGraphMemory,
):
    await memory.create_entities(
        [
            Entity(name="Alice", type="person", observations=[]),
            Entity(name="Bob", type="person", observations=[]),
            Entity(name="Carol", type="person", observations=[]),
        ]
    )
    await memory.create_relations(
        [
            Relation(source="Alice", target="Bob", relationType="KNOWS"),
            Relation(source="Carol", target="Alice", relationType="KNOWS"),
        ]
    )
    found = await memory.find_memories_by_name(["Alice"])
    names = {entity.name for entity in found.entities}
    assert names == {"Alice", "Bob", "Carol"}
    for relation in found.relations:
        assert relation.source in names and relation.target in names


def test_a_relationship_type_is_canonicalised_or_refused():
    assert canonical_relation_type("works_at") == "WORKS_AT"
    assert canonical_relation_type("WORKS_AT") == "WORKS_AT"
    for bad in ('a"b', "1_START", "has space", ""):
        with pytest.raises(ValueError):
            canonical_relation_type(bad)
