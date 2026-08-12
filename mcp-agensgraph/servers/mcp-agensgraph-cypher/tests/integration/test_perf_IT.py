"""Integration tests for the plan, advice and health tools."""

import json
from typing import Any

import pytest
from fastmcp.exceptions import ToolError
from fastmcp.server import FastMCP


async def _call(server: FastMCP, name: str, args: dict | None = None):
    tool = await server.get_tool(name)
    return json.loads((await tool.run(args or {})).content[0].text)


class TestExplain:
    @pytest.mark.asyncio(loop_scope="function")
    async def test_returns_a_plan_without_running_the_query(
        self, mcp_server: FastMCP, init_data: Any
    ):
        plan = await _call(
            mcp_server,
            "explain_agensgraph_cypher",
            {"query": 'MATCH (n:"Person") RETURN n.name'},
        )
        node = plan[0]["Plan"]
        assert "Node Type" in node
        # planned only, so no measured timing is present
        assert "Actual Total Time" not in node

    @pytest.mark.asyncio(loop_scope="function")
    async def test_analyze_reports_actual_timings(
        self, mcp_server: FastMCP, init_data: Any
    ):
        plan = await _call(
            mcp_server,
            "explain_agensgraph_cypher",
            {"query": 'MATCH (n:"Person") RETURN n.name', "analyze": True},
        )
        assert "Actual Total Time" in plan[0]["Plan"]

    @pytest.mark.asyncio(loop_scope="function")
    @pytest.mark.parametrize(
        "statement",
        [
            'CREATE (:"Person" {name: \'analyze_probe\'})',
            'INSERT (:"Person" {name: \'analyze_probe\'})',
            "INSERT INTO analyze_probe_tbl VALUES ('x')",
            "MATCH (n:\"Person\") RETURN n; CREATE (:\"Person\")",
        ],
    )
    async def test_analyze_is_refused_for_a_write(
        self, mcp_server: FastMCP, init_data: Any, statement: str
    ):
        """ANALYZE executes, so it must not be a way to run a write through a read tool.

        Refused by the read-only transaction rather than by a reading of the text, which is why
        the GQL spelling and the SQL one are refused alongside the Cypher one.
        """
        tool = await mcp_server.get_tool("explain_agensgraph_cypher")
        with pytest.raises(ToolError):
            await tool.run({"query": statement, "analyze": True})

    @pytest.mark.asyncio(loop_scope="function")
    async def test_and_the_write_it_refused_did_not_happen(
        self, mcp_server: FastMCP, init_data: Any
    ):
        """The refusal is only worth anything if nothing landed."""
        read = await mcp_server.get_tool("read_agensgraph_cypher")
        explain = await mcp_server.get_tool("explain_agensgraph_cypher")
        before = await read.run(
            {"query": 'MATCH (n:"Person") RETURN count(*) AS c'}
        )
        with pytest.raises(ToolError):
            await explain.run(
                {"query": 'CREATE (:"Person" {name: \'never\'})', "analyze": True}
            )
        after = await read.run({"query": 'MATCH (n:"Person") RETURN count(*) AS c'})
        assert before.content[0].text == after.content[0].text

    @pytest.mark.asyncio(loop_scope="function")
    async def test_malformed_cypher_is_reported(self, mcp_server: FastMCP, init_data: Any):
        tool = await mcp_server.get_tool("explain_agensgraph_cypher")
        with pytest.raises(ToolError):
            await tool.run({"query": 'MATCH (n:"Person" RETURN n'})


class TestRecommendations:
    @pytest.mark.asyncio(loop_scope="function")
    async def test_reports_existing_indexes_and_says_it_did_not_verify(
        self, mcp_server: FastMCP, init_data: Any
    ):
        out = await _call(
            mcp_server,
            "recommend_property_indexes",
            {"query": 'MATCH (n:"Person") WHERE n.name = \'a\' RETURN n'},
        )
        assert out["verified"] is False
        assert "cannot be simulated" in out["note"]
        assert "existing_indexes" in out
        assert isinstance(out["findings"], list)

    @pytest.mark.asyncio(loop_scope="function")
    async def test_flags_starts_with_as_never_indexable(
        self, mcp_server: FastMCP, init_data: Any
    ):
        out = await _call(
            mcp_server,
            "recommend_property_indexes",
            {"query": 'MATCH (n:"Person") WHERE n.name STARTS WITH \'a\' RETURN n'},
        )
        kinds = {f["kind"] for f in out["findings"]}
        assert "starts_with_not_indexable" in kinds
        finding = next(
            f for f in out["findings"] if f["kind"] == "starts_with_not_indexable"
        )
        assert "AGV2-514" in finding["suggestion"]

    @pytest.mark.asyncio(loop_scope="function")
    async def test_small_labels_do_not_produce_index_advice(
        self, mcp_server: FastMCP, init_data: Any
    ):
        """A sequential scan over a handful of rows is the right plan, not a finding."""
        out = await _call(
            mcp_server,
            "recommend_property_indexes",
            {"query": 'MATCH (n:"Person") WHERE n.name = \'a\' RETURN n'},
        )
        assert not [f for f in out["findings"] if f["kind"] == "missing_index"]


class TestHealth:
    @pytest.mark.asyncio(loop_scope="function")
    async def test_reports_each_check_and_which_extensions_are_present(
        self, mcp_server: FastMCP, init_data: Any
    ):
        out = await _call(mcp_server, "agensgraph_health")
        assert "extensions" in out
        for check in ("cache_hit_ratio", "unused_indexes", "vacuum_age", "connections"):
            assert check in out, check

    @pytest.mark.asyncio(loop_scope="function")
    async def test_a_missing_extension_is_reported_not_raised(
        self, mcp_server: FastMCP, init_data: Any
    ):
        out = await _call(mcp_server, "agensgraph_health")
        for name in ("pgstattuple", "pg_buffercache"):
            if not out["extensions"].get(name):
                assert out[name]["available"] is False
                assert name in out[name]["note"]


class TestTopQueries:
    @pytest.mark.asyncio(loop_scope="function")
    async def test_returns_rows_or_explains_the_missing_extension(
        self, mcp_server: FastMCP, init_data: Any
    ):
        out = await _call(mcp_server, "top_cypher_queries", {"limit": 5})
        if isinstance(out, dict):
            assert out["available"] is False
            assert "pg_stat_statements" in out["note"]
        else:
            assert isinstance(out, list)
