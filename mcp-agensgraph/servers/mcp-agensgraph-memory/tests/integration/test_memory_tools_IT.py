"""The tools as an MCP client meets them: their schemas, and what a failure tells it."""

import json

import pytest
from fastmcp.exceptions import ToolError

from mcp_agensgraph_memory.agensgraph_memory import MAX_LIMIT, AgensGraphMemory, Entity
from mcp_agensgraph_memory.server import create_mcp_server


async def tools_of(memory: AgensGraphMemory):
    server = create_mcp_server(memory)
    return {tool.name: tool for tool in await server.list_tools()}


@pytest.mark.asyncio
async def test_every_read_advertises_its_ceiling(memory: AgensGraphMemory):
    """A limit with no upper bound let one call ask for the whole memory."""
    tools = await tools_of(memory)
    for name in ("read_graph", "search_memories", "find_memories_by_name"):
        limit = tools[name].parameters["properties"]["limit"]
        assert limit["maximum"] == MAX_LIMIT, f"{name} does not bound its limit"
        assert limit["minimum"] == 1


@pytest.mark.asyncio
async def test_a_refused_relationship_type_is_explained(memory: AgensGraphMemory):
    tools = await tools_of(memory)
    await memory.create_entities(
        [
            Entity(name="A1", type="person", observations=[]),
            Entity(name="A2", type="person", observations=[]),
        ]
    )
    with pytest.raises(ToolError, match="relationship type"):
        await tools["create_relations"].run(
            {"relations": [{"source": "A1", "target": "A2", "relationType": 'a"b'}]}
        )


@pytest.mark.asyncio
async def test_a_tool_result_carries_what_was_written_and_what_was_not(
    memory: AgensGraphMemory,
):
    tools = await tools_of(memory)
    await memory.create_entities([Entity(name="A1", type="person", observations=[])])
    result = await tools["create_relations"].run(
        {"relations": [{"source": "A1", "target": "ghost", "relationType": "knows"}]}
    )
    payload = json.loads(result.content[0].text)
    assert payload["created"] == []
    assert payload["skipped"][0]["target"] == "ghost"
    assert payload["skipped"][0]["relationType"] == "KNOWS"
