"""Starting up over a store written before any of the invariants existed.

The graph these build is the one the old server left: several elements sharing a name, a
relationship type spelled in the wrong case, several relationships between one pair, and a
full-text index over a function the package installed.
"""

import pytest
import pytest_asyncio

from mcp_agensgraph_memory.agensgraph_memory import AgensGraphMemory
from agensgraph.cypher import quote_identifier
from mcp_agensgraph_memory.bootstrap import (
    FULLTEXT_INDEX,
    MEMORY_LABEL,
    bootstrap,
    edge_index_name,
    ensure_graph,
    make_pool,
    verify,
)

from conftest import dsn, settings

JSONB_TO_STRING = r"""
    CREATE OR REPLACE FUNCTION jsonb_to_string(j jsonb, sep text DEFAULT ', ')
    RETURNS text AS $$
    SELECT CASE
        WHEN jsonb_typeof(j) = 'array' THEN (
          SELECT string_agg(value::text, sep) FROM jsonb_array_elements_text(j))
        WHEN jsonb_typeof(j) = 'object' THEN (
          SELECT string_agg(key || '=' || value, sep) FROM jsonb_each_text(j))
        ELSE j::text END;
    $$ LANGUAGE sql IMMUTABLE
"""

OLD_FULLTEXT_INDEX = """
    CREATE PROPERTY INDEX memory_fulltext_idx ON "Memory" USING gin
    ((
        setweight(to_tsvector('english', coalesce(name, '')), 'A') ||
        setweight(to_tsvector('english', coalesce(type, '')), 'B') ||
        setweight(to_tsvector('english', coalesce(jsonb_to_string(observations, ' '), '')), 'C')
    ))
"""

OLD_STORE = [
    """CREATE (:"Memory" {name: 'Alice', type: 'person', observations: ['a1', 'shared']})""",
    """CREATE (:"Memory" {name: 'Alice', type: 'robot', observations: ['a2', 'shared']})""",
    """CREATE (:"Memory" {name: 'Alice', type: 'person', observations: ['a3']})""",
    """CREATE (:"Memory" {name: 'Bob', type: 'person', observations: ['b1']})""",
    """CREATE (:"Memory" {name: 'Carol', type: 'person', observations: []})""",
    'CREATE ELABEL "KNOWS"',
    'CREATE ELABEL "works_at"',
    """MATCH (a:"Memory" {name: 'Alice'}), (b:"Memory" {name: 'Bob'})
       CREATE (a)-[:"KNOWS"]->(b)""",
    """MATCH (a:"Memory" {name: 'Alice'}), (b:"Memory" {name: 'Bob'})
       CREATE (a)-[:"works_at" {since: 2020}]->(b)""",
    """MATCH (b:"Memory" {name: 'Bob'}), (c:"Memory" {name: 'Carol'})
       CREATE (b)-[:"KNOWS" {written: 1}]->(c)""",
    """MATCH (b:"Memory" {name: 'Bob'}), (c:"Memory" {name: 'Carol'})
       CREATE (b)-[:"KNOWS" {written: 2}]->(c)""",
]


@pytest_asyncio.fixture(scope="function")
async def old_store():
    """A memory graph as the previous version of this server left it."""
    where = settings()
    graphname = f"{where['graphname']}_old"
    connection = dsn(where)
    await ensure_graph(connection, graphname)
    pool = make_pool(connection, graphname)
    await pool.open()
    async with pool.connection() as conn:
        await conn.execute_query("MATCH (n) DETACH DELETE n")
        for label in await conn.labels():
            if not label.is_builtin:
                await conn.execute_query(
                    f'drop {"elabel" if label.is_edge else "vlabel"} "{label.name}" cascade'
                )
        await conn.execute_query(JSONB_TO_STRING)
        await conn.execute_query('CREATE VLABEL "Memory"')
        for statement in OLD_STORE:
            await conn.execute_query(statement)
        await conn.execute_query(OLD_FULLTEXT_INDEX)
        await conn.commit()
    yield pool, graphname
    await pool.close()


async def index_definitions(pool, graphname):
    async with pool.connection() as conn:
        found = await conn.execute_query(
            "select indexname, indexdef from pg_indexes where schemaname = %s", (graphname,)
        )
        await conn.rollback()
    return {row[0]: row[1] for row in found.records}


@pytest.mark.asyncio
async def test_starting_up_merges_the_elements_that_share_a_name(old_store):
    """A unique index cannot be built over a name two elements hold, and dropping one of them
    would throw away what was written to it."""
    pool, graphname = old_store
    report = await bootstrap(pool, graphname)

    assert report.merged_names == {"Alice": 3}
    assert report.conflicting_types == ["Alice"]
    assert report.merged_observations == 2

    memory = AgensGraphMemory(pool, graphname)
    found = await memory.find_memories_by_name(["Alice"])
    alice = next(entity for entity in found.entities if entity.name == "Alice")
    assert alice.observations == ["a1", "shared", "a2", "a3"], (
        "every copy's observations survive, each kept once, in the order they were written"
    )
    async with pool.connection() as conn:
        copies = await conn.execute_query('MATCH (n:"Memory" {name: \'Alice\'}) RETURN count(*)')
        await conn.rollback()
    assert copies.records[0][0] == 1


@pytest.mark.asyncio
async def test_starting_up_keeps_the_relationships_of_the_copies(old_store):
    pool, graphname = old_store
    await bootstrap(pool, graphname)
    memory = AgensGraphMemory(pool, graphname)
    graph = await memory.read_graph()
    written = {(r.source, r.target, r.relationType) for r in graph.relations}
    assert written == {
        ("Alice", "Bob", "KNOWS"),
        ("Alice", "Bob", "WORKS_AT"),
        ("Bob", "Carol", "KNOWS"),
    }


@pytest.mark.asyncio
async def test_starting_up_moves_a_miscased_type_onto_its_canonical_spelling(old_store):
    pool, graphname = old_store
    report = await bootstrap(pool, graphname)
    assert report.relabelled == {"works_at": 1}
    async with pool.connection() as conn:
        labels = {label.name for label in await conn.labels() if label.is_edge}
        properties = await conn.execute_query(
            'MATCH ()-[r:"WORKS_AT"]->() RETURN properties(r)'
        )
        await conn.rollback()
    assert "works_at" not in labels
    assert properties.records == [({"since": 2020},)], "the relationship keeps what it held"


@pytest.mark.asyncio
async def test_starting_up_collapses_duplicate_relationships_and_keeps_the_first(old_store):
    pool, graphname = old_store
    report = await bootstrap(pool, graphname)
    assert report.merged_relations == 1
    async with pool.connection() as conn:
        properties = await conn.execute_query(
            """MATCH (:"Memory" {name: 'Bob'})-[r:"KNOWS"]->(:"Memory" {name: 'Carol'})
               RETURN properties(r)"""
        )
        await conn.rollback()
    assert properties.records == [({"written": 1},)]


@pytest.mark.asyncio
async def test_starting_up_rebuilds_the_index_that_needed_an_installed_function(old_store):
    """An index over a function declared immutable is one PostgreSQL cannot notice a change to,
    so a rewritten body leaves searches answering from entries nothing can reproduce."""
    pool, graphname = old_store
    before = await index_definitions(pool, graphname)
    assert "jsonb_to_string" in before[FULLTEXT_INDEX]

    report = await bootstrap(pool, graphname)
    assert report.reindexed is True
    after = await index_definitions(pool, graphname)
    assert "jsonb_to_string" not in after[FULLTEXT_INDEX]
    assert "to_tsvector" in after[FULLTEXT_INDEX]

    memory = AgensGraphMemory(pool, graphname)
    assert {entity.name for entity in (await memory.search_memories("shared")).entities} == {
        "Alice"
    }


@pytest.mark.asyncio
async def test_starting_up_again_changes_nothing(old_store):
    pool, graphname = old_store
    await bootstrap(pool, graphname)
    second = await bootstrap(pool, graphname)
    assert second.statements == []
    assert second.merged_names == {}
    assert second.relabelled == {}
    assert second.merged_relations == 0
    assert second.reindexed is False


@pytest.mark.asyncio
async def test_starting_up_leaves_the_indexes_every_tool_depends_on(old_store):
    pool, graphname = old_store
    await bootstrap(pool, graphname)
    definitions = await index_definitions(pool, graphname)
    assert "UNIQUE" in definitions["Memory_name_idx"]
    assert "UNIQUE" in definitions[edge_index_name("KNOWS")]
    assert "UNIQUE" in definitions[edge_index_name("WORKS_AT")]


@pytest.mark.asyncio
async def test_a_graph_without_the_memory_label_is_refused(old_store):
    """A pattern naming a label the graph does not have returns nothing and raises nothing, so
    a graph name with a typo in it reads as a memory with nothing in it. Starting up says so
    instead, once, where it can still be reported."""
    pool, graphname = old_store
    async with pool.connection() as conn:
        with pytest.raises(RuntimeError, match="no 'Memory' label"):
            await verify(conn, "a graph nobody made")
        await conn.rollback()


@pytest.mark.asyncio
async def test_a_memory_whose_names_are_not_unique_is_refused(old_store):
    """Without the unique index two callers writing one entity each make one."""
    pool, graphname = old_store
    await bootstrap(pool, graphname)
    async with pool.connection() as conn:
        await conn.execute_query('drop property index "Memory_name_idx"')
        with pytest.raises(RuntimeError, match="unique"):
            await verify(conn, graphname)
        await conn.rollback()


@pytest_asyncio.fixture(scope="function")
async def shared_graph():
    """A graph holding another application's data as well as this server's."""
    where = settings()
    graphname = f"{where['graphname']}_shared"
    connection = dsn(where)
    await ensure_graph(connection, graphname)
    pool = make_pool(connection, graphname)
    await pool.open()
    async with pool.connection() as conn:
        await conn.execute_query("MATCH (n) DETACH DELETE n")
        for label in await conn.labels():
            if not label.is_builtin:
                await conn.execute_query(
                    f'drop {"elabel" if label.is_edge else "vlabel"} "{label.name}" cascade'
                )
        for mode in ("air", "sea"):
            await conn.execute_query(
                f"""CREATE (:"Depot" {{n: '{mode}-from'}})
                    -[:"shipsTo" {{mode: '{mode}'}}]->
                    (:"Depot" {{n: '{mode}-to'}})"""
            )
        await conn.commit()
    yield pool, graphname
    await pool.close()


class TestAGraphThisServerShares:
    """Starting up rewrites and drops labels, so it must touch only the ones it wrote.

    Reading every label in the graph handed the migrations another application's edges: a graph
    holding two distinct ``shipsTo`` relationships came back holding one, the report saying it
    had relabelled one and mentioning nothing lost.
    """

    @pytest.mark.asyncio
    async def test_a_label_this_server_did_not_write_is_left_alone(self, shared_graph):
        pool, graphname = shared_graph
        report = await bootstrap(pool, graphname, adopt=True)
        assert report.relabelled == {}
        async with pool.connection() as conn:
            kept = await conn.execute_query(
                'MATCH ()-[r:"shipsTo"]->() RETURN r.mode ORDER BY r.mode'
            )
            labels = {label.name for label in await conn.labels()}
            await conn.rollback()
        assert [mode for mode, in kept.records] == ["air", "sea"]
        assert "shipsTo" in labels

    @pytest.mark.asyncio
    async def test_two_relationships_of_one_label_both_survive_being_moved(self, shared_graph):
        """Moving relationships onto a canonical spelling keeps one for one.

        Merging on the pair instead leaves a single relationship carrying the last one's
        properties, so a pair differing only in what it holds comes out as one.
        """
        pool, graphname = shared_graph
        async with pool.connection() as conn:
            for name in ("x", "y"):
                await conn.execute_query(f"CREATE (:\"Memory\" {{name: '{name}'}})")
            for since in (2020, 2024):
                await conn.execute_query(
                    f"""MATCH (a:"Memory" {{name: 'x'}}), (b:"Memory" {{name: 'y'}})
                        CREATE (a)-[:"Works_At" {{since: {since}}}]->(b)"""
                )
            await conn.commit()
        report = await bootstrap(pool, graphname, adopt=True)
        assert report.relabelled == {"Works_At": 2}


class TestAGraphWhereAnotherApplicationAlsoUsesTheLabel:
    """``Memory`` is a generic name, so another application can be using it too.

    Ownership was read from one relationship joining two ``Memory`` vertices, and the move that
    followed had no such constraint -- so a type carrying one relationship of that shape had all
    of them moved, including the ones reaching elsewhere, and the label was then dropped.
    """

    @pytest.mark.asyncio
    async def test_a_type_reaching_beyond_this_server_is_not_migrated(self, shared_graph):
        pool, graphname = shared_graph
        async with pool.connection() as conn:
            await conn.execute_query(
                """CREATE (:"Memory" {name: 'm1'}), (:"Memory" {name: 'm2'}),
                          (:"Company" {n: 'c'})"""
            )
            await conn.execute_query(
                """MATCH (a:"Memory" {name: 'm1'}), (b:"Memory" {name: 'm2'})
                   CREATE (a)-[:"partnersWith" {k: 1}]->(b)"""
            )
            await conn.execute_query(
                """MATCH (a:"Memory" {name: 'm1'}), (b:"Company")
                   CREATE (a)-[:"partnersWith" {k: 2}]->(b)"""
            )
            await conn.commit()
        report = await bootstrap(pool, graphname, adopt=True)
        assert "partnersWith" not in report.relabelled
        async with pool.connection() as conn:
            kept = await conn.execute_query(
                'MATCH ()-[r:"partnersWith"]->() RETURN r.k ORDER BY r.k'
            )
            labels = {label.name for label in await conn.labels()}
            await conn.rollback()
        assert [k for k, in kept.records] == [1, 2]
        assert "partnersWith" in labels


class TestWhoseGraphItIs:
    """Starting up rewrites what is already there, so it asks whose data that is.

    The question was answered by looking for the ``Memory`` label, which is a generic name any
    application might use. One that had used it first had two vertices sharing a name folded into
    one -- a vertex destroyed -- and its relationships adopted and then collapsed to one per pair.
    """

    @pytest.mark.asyncio
    async def test_a_graph_someone_else_filled_is_refused(self, shared_graph):
        pool, graphname = shared_graph
        async with pool.connection() as conn:
            await conn.execute_query(
                """CREATE (:"Memory" {name: 'shared', payload: 'A'}),
                          (:"Memory" {name: 'shared', payload: 'B'})"""
            )
            await conn.commit()
        with pytest.raises(RuntimeError, match="did not make"):
            await bootstrap(pool, graphname)
        async with pool.connection() as conn:
            kept = await conn.execute_query('MATCH (n:"Memory") RETURN count(n)')
            await conn.rollback()
        assert kept.records[0][0] == 2

    @pytest.mark.asyncio
    async def test_it_is_taken_when_the_operator_says_so(self, shared_graph):
        pool, graphname = shared_graph
        async with pool.connection() as conn:
            await conn.execute_query('CREATE (:"Memory" {name: \'mine\'})')
            await conn.commit()
        report = await bootstrap(pool, graphname, adopt=True)
        assert report.adopted is True

    @pytest.mark.asyncio
    async def test_a_graph_it_made_itself_needs_no_saying(self, shared_graph):
        pool, graphname = shared_graph
        first = await bootstrap(pool, graphname)
        second = await bootstrap(pool, graphname)
        assert first.adopted is True
        assert second.adopted is False

    @pytest.mark.asyncio
    async def test_a_store_an_earlier_version_left_is_taken_as_its_own(self, old_store):
        """No marker, but the full-text index only this server builds is already there."""
        pool, graphname = old_store
        report = await bootstrap(pool, graphname)
        assert report.adopted is True

    @pytest.mark.asyncio
    async def test_an_index_of_the_same_name_is_not_the_same_claim(self, shared_graph):
        """``<label>_fulltext_idx`` is the ordinary way to name such an index.

        So another application with a ``Memory`` label and a full-text index over it would most
        likely have named it exactly what this server names its own. The expression is this
        server's own and is what the claim is read from.
        """
        pool, graphname = shared_graph
        async with pool.connection() as conn:
            await conn.execute_query(f'CREATE VLABEL {quote_identifier(MEMORY_LABEL)}')
            await conn.execute_query(
                f'CREATE (:{quote_identifier(MEMORY_LABEL)} '
                f"{{name: 'theirs', payload: 'keep me'}})"
            )
            await conn.commit()
            await conn.execute(
                f'create index {FULLTEXT_INDEX} on {quote_identifier(graphname)}.'
                f"{quote_identifier(MEMORY_LABEL)} using gin "
                f"(to_tsvector('english', coalesce(properties->>'body', '')))"
            )
            await conn.commit()
        with pytest.raises(RuntimeError, match="did not make"):
            await bootstrap(pool, graphname)
        async with pool.connection() as conn:
            kept = await conn.execute_query(
                f'MATCH (n:{quote_identifier(MEMORY_LABEL)}) RETURN count(n)'
            )
            await conn.rollback()
        assert kept.records[0][0] == 1


class TestOrderingByRecencyIsIndexed:
    """A page of the most recently written is what a capped read of a memory often wants.

    Without an index it is a scan of the whole label and a top-N sort -- 195 ms on twenty
    thousand entities against 8.6 with one -- and the index costs 1.24x on a write, measured
    server-side on a create of twenty: 0.566 ms against 0.704.
    """

    @pytest.mark.asyncio
    async def test_starting_up_indexes_what_a_page_is_ordered_by(self, old_store):
        pool, graphname = old_store
        await bootstrap(pool, graphname)
        async with pool.connection() as conn:
            found = await conn.indexes(MEMORY_LABEL, graph=graphname)
            await conn.rollback()
        assert any("updated" in index.definition for index in found)
