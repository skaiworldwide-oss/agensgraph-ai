import json
from typing import Any

import pytest
from mcp.server import FastMCP


@pytest.mark.asyncio(loop_scope="function")
async def test_get_agensgraph_schema(mcp_server: FastMCP, init_data: Any):
    response = await mcp_server.call_tool("get_agensgraph_schema", dict())

    temp_parsed = json.loads(response[0].text)['content'][0]['text']
    schema = json.loads(temp_parsed)[0]

    # Verify the schema result
    assert "label" in schema
    assert "attributes" in schema
    assert "relationships" in schema


@pytest.mark.asyncio(loop_scope="function")
async def test_write_agensgraph_cypher(mcp_server: FastMCP):
    # Execute a Cypher query to create a node
    query = "CREATE (n:Test {name: 'test', age: 123}) RETURN n.name"
    response = await mcp_server.call_tool("write_agensgraph_cypher", dict(query=query))
    result = json.loads(json.loads(response[0].text)['content'][0]['text'])

    # Verify the node creation
    assert len(result) == 5
    assert result["insertedvertices"] == 1

@pytest.mark.asyncio(loop_scope="function")
async def test_read_agensgraph_cypher(mcp_server: FastMCP, init_data: Any):
    # Prepare test data

    # Execute a complex read query
    query = """
    MATCH (p:Person)-[:FRIEND]->(friend)
    RETURN p.name AS person, friend.name AS friend_name
    ORDER BY person, friend_name
    """

    response = await mcp_server.call_tool("read_agensgraph_cypher", dict(query=query))
    result = json.loads(json.loads(response[0].text)['content'][0]['text'])
    # # Verify the query result
    assert len(result) == 2
    assert result[0]["person"] == "Alice"
    assert result[0]["friend_name"] == "Bob"
    assert result[1]["person"] == "Bob"
    assert result[1]["friend_name"] == "Charlie"
