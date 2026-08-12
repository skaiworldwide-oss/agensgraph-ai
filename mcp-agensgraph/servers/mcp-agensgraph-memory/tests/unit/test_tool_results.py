"""What a tool tells its caller when something fails, and what it will not tell them."""

import psycopg
import pytest
from fastmcp.exceptions import ToolError

from mcp_agensgraph_memory.agensgraph_memory import MAX_LIMIT
from mcp_agensgraph_memory.server import memory_limit_from_env, tool_error

# What PostgreSQL puts in a failure's DETAIL, which is a value the tool was never given.
LEAKING = (
    'duplicate key value violates unique constraint "Memory_name_idx"\n'
    "DETAIL:  Key ((properties.'name'::text))=(\"alice@example.com\") already exists.\n"
    "CONTEXT:  COPY Memory, line 1"
)


def test_a_database_failure_does_not_reach_the_caller_carrying_row_data():
    """A tool result goes to a model and from there wherever the conversation goes."""
    error = tool_error("creating entities", psycopg.errors.UniqueViolation(LEAKING))
    assert isinstance(error, ToolError)
    assert "alice@example.com" not in str(error)
    assert "DETAIL" not in str(error)
    assert "CONTEXT" not in str(error)
    assert "Memory_name_idx" not in str(error)
    assert "23505" in str(error), "the SQLSTATE says what kind of failure it was, and no more"
    assert "creating entities" in str(error)


def test_a_failure_with_no_sqlstate_says_where_to_look():
    error = tool_error("reading the knowledge graph", RuntimeError(LEAKING))
    assert "alice@example.com" not in str(error)
    assert "server log" in str(error)


def test_a_refused_request_is_explained_to_the_caller():
    """This server's own reading of the request. Saying what was wrong with it is how the
    caller writes one that works."""
    error = tool_error(
        "creating relations", ValueError("a relationship type is a letter or underscore")
    )
    assert "a relationship type is a letter or underscore" in str(error)


@pytest.mark.parametrize(
    "given,expected",
    [(None, 1000), ("50", 50), ("0", 1), ("999999", MAX_LIMIT), ("many", 1000), ("", 1000)],
)
def test_the_page_size_is_bounded_however_it_is_configured(monkeypatch, given, expected):
    """An unbounded read returned 20,100 entities as 3.8 MB of JSON."""
    monkeypatch.delenv("AGENSGRAPH_MEMORY_LIMIT", raising=False)
    if given is not None:
        monkeypatch.setenv("AGENSGRAPH_MEMORY_LIMIT", given)
    assert memory_limit_from_env() == expected
