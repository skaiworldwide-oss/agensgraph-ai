"""What the tools gained by asking the driver instead of asking the database by hand."""

import json
from typing import Any

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError
from fastmcp.server import FastMCP
from mcp_agensgraph_common.results import OMITTED
from mcp_agensgraph_cypher.server import (
    create_mcp_server,
    ensure_graph,
    server_has_gql_clauses,
)

CATALOG_CONTENTS = """
    SELECT n.nspname || '.' || c.relname
      FROM pg_catalog.pg_class c
      JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
     WHERE n.nspname = %s
    UNION ALL
    SELECT n.nspname || '.' || p.proname
      FROM pg_catalog.pg_proc p
      JOIN pg_catalog.pg_namespace n ON n.oid = p.pronamespace
     WHERE n.nspname IN (%s, 'public')
"""
"""What a server could have left behind, in the places it could leave it.

Counted over the whole database, and by size, this failed whenever anything else in the database
made or dropped a relation -- a count that went *down* by one, which nothing being installed can
do. What is being asked is whether this server added something, so it is asked of the names
themselves, which also says which name appeared.

Relations are asked about in the graph's own schema, which nothing else writes. In ``public``
only functions are, because that is where a helper function would land -- the thing this is
guarding against, since a server here once installed a plpgsql ``typeof`` on every startup --
and because relations in a shared ``public`` are what another session creates and drops all day.
Measured across three runs of the suite: ``pg_proc`` identical every time, ``pg_class`` moving
in two of them.
"""


async def _call(server: FastMCP, name: str, args: dict | None = None):
    tool = await server.get_tool(name)
    return json.loads((await tool.run(args or {})).content[0].text)


async def _contents(pool, graphname: str) -> set[str]:
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(CATALOG_CONTENTS, (graphname, graphname))
            found = await cur.fetchall()
        await conn.rollback()
    return {name for name, in found}


async def _run(pool, statement: str) -> None:
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(statement)
        await conn.commit()


class TestInstallsNothing:
    @pytest.mark.asyncio(loop_scope="function")
    async def test_a_full_startup_leaves_nothing_behind_in_the_catalogs(
        self, setup, graphname, db_url
    ):
        """Serving a graph is not a reason to create anything in somebody's database.

        The graph already exists, which is the state a server starts in every time after the
        first, so a start that leaves behind a relation or a function nobody asked for has
        installed something. Only what appeared is asked about: something else dropping a
        relation in the meantime is not this server's doing.
        """
        before = await _contents(setup, graphname)
        await ensure_graph(db_url, graphname)
        gql = await server_has_gql_clauses(setup)
        server = create_mcp_server(
            setup, allow_server_programs=True, graphname=graphname, gql_clauses=gql
        )
        async with Client(server) as client:
            for name in (
                "get_agensgraph_schema",
                "agensgraph_health",
                "top_cypher_queries",
            ):
                await client.call_tool(name, {})
            await client.call_tool(
                "read_agensgraph_cypher", {"query": "MATCH (n) RETURN count(*) AS c"}
            )
        after = await _contents(setup, graphname)
        assert after - before == set()


class TestSchema:
    @pytest.mark.asyncio(loop_scope="function")
    async def test_a_property_index_is_reported_on_the_property_it_covers(
        self, mcp_server: FastMCP, graphname, setup, init_data: Any
    ):
        await _run(
            setup,
            'CREATE UNIQUE PROPERTY INDEX IF NOT EXISTS person_name_uq ON "Person" (name)',
        )
        try:
            schema = await _call(mcp_server, "get_agensgraph_schema")
            assert schema["Person"]["properties"]["name"]["indexed"] is True
            assert schema["Person"]["properties"]["name"]["unique"] is True
            assert schema["Person"]["properties"]["age"]["indexed"] is False
        finally:
            await _run(setup, "DROP PROPERTY INDEX person_name_uq")

    @pytest.mark.asyncio(loop_scope="function")
    async def test_a_uniqueness_assertion_is_found_too(
        self, mcp_server: FastMCP, graphname, setup, init_data: Any
    ):
        """A uniqueness assertion is kept as an exclusion, which the index view hides."""
        await _run(
            setup,
            'CREATE CONSTRAINT person_age_uq ON "Person" ASSERT age IS UNIQUE',
        )
        try:
            schema = await _call(mcp_server, "get_agensgraph_schema")
            assert schema["Person"]["properties"]["age"]["unique"] is True
        finally:
            await _run(setup, 'DROP CONSTRAINT person_age_uq ON "Person"')

    @pytest.mark.asyncio(loop_scope="function")
    async def test_a_relationship_reports_the_properties_it_carries(
        self, mcp_server: FastMCP, graphname, setup, clear_data: Any
    ):
        await _run(
            setup,
            "CREATE (a:\"Person\" {name: 'Ann'})-[:\"FRIEND\" {since: 2020}]->"
            "(b:\"Person\" {name: 'Ben'})",
        )
        schema = await _call(mcp_server, "get_agensgraph_schema")
        friend = schema["Person"]["relationships"]["FRIEND"]
        assert friend["labels"] == ["Person"]
        assert "since" in friend["properties"]


class TestNativeTypes:
    @pytest.mark.asyncio(loop_scope="function")
    async def test_an_edge_read_on_its_own_names_both_of_its_endpoints(
        self, mcp_server: FastMCP, init_data: Any
    ):
        """An edge carries the identities at each end, so it does not need its vertices."""
        out = await _call(
            mcp_server,
            "read_agensgraph_cypher",
            {"query": "MATCH ()-[r]->() RETURN r", "limit": 1},
        )
        edge = out["rows"][0]["r"]
        assert edge["label"] == "FRIEND"
        assert edge["start"] and edge["end"]
        assert edge["start"] != edge["end"]


class TestSuppressedValues:
    @pytest.mark.asyncio(loop_scope="function")
    async def test_a_list_dropped_for_its_size_says_so(
        self, mcp_server: FastMCP, graphname, setup, clear_data: Any
    ):
        """A model asking for a value and receiving nothing cannot tell which happened."""
        await _run(
            setup,
            'CREATE (:"Bulky" {v: [' + ",".join(str(i) for i in range(200)) + "]})",
        )
        out = await _call(
            mcp_server,
            "read_agensgraph_cypher",
            {"query": 'MATCH (n:"Bulky") RETURN n.v AS v'},
        )
        assert out["rows"][0]["v"] == OMITTED.format(count=200)


class TestBoundedResponses:
    @pytest.mark.asyncio(loop_scope="function")
    async def test_whole_rows_are_dropped_and_the_reply_still_parses(
        self, setup, graphname, clear_data: Any
    ):
        server = create_mcp_server(
            setup, allow_server_programs=True, graphname=graphname, token_limit=200
        )
        await _run(
            setup,
            "UNWIND range(1, 60)::jsonb AS i CREATE (:\"Wordy\" {t: 'word word word word'})",
        )
        out = await _call(
            server,
            "read_agensgraph_cypher",
            {"query": 'MATCH (n:"Wordy") RETURN n.t AS t', "limit": 60},
        )
        assert 0 < out["row_count"] < 60
        assert out["rows_omitted"] == 60 - out["row_count"]
        assert all(row["t"] for row in out["rows"])

    @pytest.mark.asyncio(loop_scope="function")
    async def test_nothing_is_dropped_when_it_all_fits(
        self, setup, graphname, init_data: Any
    ):
        server = create_mcp_server(
            setup, allow_server_programs=True, graphname=graphname, token_limit=10_000
        )
        out = await _call(
            server,
            "read_agensgraph_cypher",
            {"query": 'MATCH (n:"Person") RETURN n.name AS name'},
        )
        assert out["row_count"] == 3
        assert "rows_omitted" not in out


class TestPagingAndTheGrammar:
    @pytest.mark.asyncio(loop_scope="function")
    async def test_offset_is_a_synonym_for_skip(
        self, mcp_server: FastMCP, setup, init_data: Any
    ):
        """A query ending in OFFSET already carries paging, so ours goes outside it."""
        if not await server_has_gql_clauses(setup):
            pytest.skip("this server has no OFFSET clause")
        out = await _call(
            mcp_server,
            "read_agensgraph_cypher",
            {"query": 'MATCH (n:"Person") RETURN n.name AS name ORDER BY name OFFSET 1'},
        )
        assert [row["name"] for row in out["rows"]] == ["Bob", "Charlie"]

    @pytest.mark.asyncio(loop_scope="function")
    @pytest.mark.parametrize(
        "query,clause",
        [
            (
                'MATCH (n:"Person") WITH n.name AS name FILTER name IS NOT NULL '
                "RETURN name LIMIT 2",
                "FILTER",
            ),
            (
                'MATCH (n:"Person") RETURN n.name AS name NEXT RETURN name LIMIT 2',
                "NEXT",
            ),
        ],
    )
    async def test_a_top_of_statement_clause_that_cannot_be_paged_is_refused_by_name(
        self, mcp_server: FastMCP, setup, init_data: Any, query: str, clause: str
    ):
        """Neither shape fits, so say which clause and what to do instead."""
        if not await server_has_gql_clauses(setup):
            pytest.skip("this server has none of the GQL clauses")
        tool = await mcp_server.get_tool("read_agensgraph_cypher")
        with pytest.raises(ToolError) as raised:
            await tool.run({"query": query})
        assert clause in str(raised.value)
        assert "top of a statement" in str(raised.value)

    @pytest.mark.asyncio(loop_scope="function")
    async def test_the_same_clause_pages_fine_without_its_own_paging(
        self, mcp_server: FastMCP, setup, init_data: Any
    ):
        if not await server_has_gql_clauses(setup):
            pytest.skip("this server has none of the GQL clauses")
        out = await _call(
            mcp_server,
            "read_agensgraph_cypher",
            {
                "query": 'MATCH (n:"Person") WITH n.name AS name ORDER BY name '
                "FILTER name IS NOT NULL RETURN name"
            },
        )
        assert [row["name"] for row in out["rows"]] == ["Alice", "Bob", "Charlie"]

    @pytest.mark.asyncio(loop_scope="function")
    async def test_paging_words_inside_a_string_are_not_clauses(
        self, mcp_server: FastMCP, init_data: Any
    ):
        out = await _call(
            mcp_server,
            "read_agensgraph_cypher",
            {"query": "MATCH (n:\"Person\") WHERE n.name = 'no LIMIT 3' RETURN n.name AS name"},
        )
        assert out["rows"] == []


class TestWalk:
    @pytest.mark.asyncio(loop_scope="function")
    async def test_a_walk_reads_the_whole_result_and_counts_all_of_it(
        self, mcp_server: FastMCP, init_data: Any
    ):
        out = await _call(
            mcp_server,
            "read_agensgraph_cypher",
            {"query": 'MATCH (n:"Person") RETURN n.name AS name', "limit": 2, "walk": True},
        )
        assert out["total_rows"] == 3
        assert out["row_count"] == 2
        assert out["has_more"] is True
        # Paging is the thing this call exists to avoid, so it does not offer one.
        assert out["next_offset"] is None

    @pytest.mark.asyncio(loop_scope="function")
    async def test_a_walk_refuses_a_clause_a_cursor_cannot_hold(
        self, mcp_server: FastMCP, setup, init_data: Any
    ):
        if not await server_has_gql_clauses(setup):
            pytest.skip("this server has none of the GQL clauses")
        tool = await mcp_server.get_tool("read_agensgraph_cypher")
        with pytest.raises(ToolError) as raised:
            await tool.run(
                {
                    "query": 'MATCH (n:"Person") WITH n.name AS name '
                    "FILTER name IS NOT NULL RETURN name",
                    "walk": True,
                }
            )
        assert "FILTER" in str(raised.value)


class TestWriteReporting:
    @pytest.mark.asyncio(loop_scope="function")
    async def test_a_write_reports_what_it_changed_and_what_it_returned(
        self, mcp_server: FastMCP, clear_data: Any
    ):
        out = await _call(
            mcp_server,
            "write_agensgraph_cypher",
            {"query": 'CREATE (n:"Kept" {name: \'x\'}) RETURN n.name AS name'},
        )
        assert out["insertedvertices"] == 1
        assert out["rows"] == [{"name": "x"}]

    @pytest.mark.asyncio(loop_scope="function")
    async def test_a_committed_write_is_never_reported_as_a_failure(
        self, mcp_server: FastMCP, clear_data: Any, monkeypatch
    ):
        """Reporting a committed write as failed tells the caller the opposite of the truth.

        A model told a write failed retries it, and the retry applies it a second time. So a
        failure after the commit is a failure to describe work that was kept, and is reported
        as one.
        """
        from mcp_agensgraph_cypher import server as module

        def explode(_value, *args, **kwargs):
            raise RuntimeError("shaping the reply went wrong")

        monkeypatch.setattr(module, "value_sanitize", explode)
        out = await _call(
            mcp_server,
            "write_agensgraph_cypher",
            {"query": 'CREATE (n:"Survivor" {name: \'y\'}) RETURN n.name AS name'},
        )
        assert out["insertedvertices"] == 1
        assert "note" in out
        monkeypatch.undo()

        found = await _call(
            mcp_server,
            "read_agensgraph_cypher",
            {"query": 'MATCH (n:"Survivor") RETURN count(*) AS c'},
        )
        assert found["rows"][0]["c"] == 1


class TestErrorSurface:
    @pytest.mark.asyncio(loop_scope="function")
    async def test_a_syntax_error_names_its_sqlstate_and_what_the_server_said(
        self, mcp_server: FastMCP, init_data: Any
    ):
        tool = await mcp_server.get_tool("read_agensgraph_cypher")
        with pytest.raises(ToolError) as raised:
            await tool.run({"query": 'MATCH (n:"Person") RETURN n.'})
        assert "42601" in str(raised.value)
        assert "syntax error" in str(raised.value)

    @pytest.mark.asyncio(loop_scope="function")
    async def test_a_timeout_is_told_apart_from_a_syntax_error(
        self, mcp_server_short_timeout: FastMCP, init_data: Any
    ):
        tool = await mcp_server_short_timeout.get_tool("read_agensgraph_cypher")
        slow = """
        UNWIND range(1, 500000)::jsonb AS x
        WITH x WHERE x % 2 = 0
        RETURN count(x) AS result
        """
        with pytest.raises(ToolError) as raised:
            await tool.run({"query": slow})
        assert "57014" in str(raised.value)


class TestToolSurface:
    @pytest.mark.asyncio(loop_scope="function")
    async def test_every_tool_describes_what_it_returns(self, mcp_server: FastMCP):
        """A tool that returns a structure and does not describe it cannot be checked."""
        async with Client(mcp_server) as client:
            tools = await client.list_tools()
        assert tools
        assert [t.name for t in tools if not t.outputSchema] == []

    @pytest.mark.asyncio(loop_scope="function")
    async def test_the_gql_spellings_are_advertised_only_where_they_exist(self, setup, graphname):
        with_gql = create_mcp_server(
            setup, allow_server_programs=True, graphname=graphname, gql_clauses=True
        )
        without = create_mcp_server(
            setup, allow_server_programs=True, graphname=graphname, gql_clauses=False
        )

        async with Client(with_gql) as client:
            named = {t.name: t for t in await client.list_tools()}
        assert "INSERT" in named["write_agensgraph_cypher"].description
        assert "FILTER" in named["read_agensgraph_cypher"].inputSchema["properties"]["query"][
            "description"
        ]

        async with Client(without) as client:
            named = {t.name: t for t in await client.list_tools()}
        assert "INSERT" not in named["write_agensgraph_cypher"].description
        assert "FILTER" not in named["read_agensgraph_cypher"].inputSchema["properties"]["query"][
            "description"
        ]

    @pytest.mark.asyncio(loop_scope="function")
    async def test_the_server_reads_the_dialect_off_the_connection(self, setup):
        assert isinstance(await server_has_gql_clauses(setup), bool)
