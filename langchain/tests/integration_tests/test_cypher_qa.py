"""Integration tests for AgensCypherQAChain against a real graph.

The model is scripted rather than real, so the assertions are about the pipeline —
repair, refusal, validation, execution — and not about a model's Cypher-writing ability.
"""

import os

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from langchain_agensgraph import AgensCypherQAChain, AgensGraph, create_cypher_tool


def _conf():
    return {
        "dbname": os.getenv("AGENSGRAPH_DB"),
        "user": os.getenv("AGENSGRAPH_USER"),
        "password": os.getenv("AGENSGRAPH_PASSWORD"),
        "host": os.getenv("AGENSGRAPH_HOST", "localhost"),
        "port": int(os.getenv("AGENSGRAPH_PORT", 5432)),
    }


@pytest.fixture
def graph():
    g = AgensGraph("cypher_qa_it", _conf(), create=True)
    g.query("MATCH (n) DETACH DELETE n")
    # the key is quoted so the graph stores "firstName" case-preserved; written bare it
    # would be stored as "firstname" and the repair would rightly leave it alone
    g.query('CREATE (:"Person" {name: \'Alice\', "firstName": \'Alice\'})')
    g.query('CREATE (:"Person" {name: \'Bob\', "firstName": \'Bob\'})')
    g.refresh_schema(force=True)
    yield g
    g.close()


def _chain(graph, cypher: str, answer: str = "An answer.", **kwargs):
    """A chain whose model emits `cypher` first, then `answer`."""
    return AgensCypherQAChain.from_llm(
        FakeListChatModel(responses=[cypher, answer]), graph=graph, **kwargs
    )


class TestHappyPath:
    def test_generated_query_runs_and_is_answered(self, graph):
        chain = _chain(graph, "MATCH (n:Person) RETURN n.name AS name LIMIT 10")
        out = chain.invoke({"query": "Who is in the graph?"})
        assert out["query"] == "Who is in the graph?"
        assert out["result"] == "An answer."

    def test_intermediate_steps_expose_the_query_and_rows(self, graph):
        chain = _chain(
            graph,
            "MATCH (n:Person) RETURN n.name AS name LIMIT 10",
            return_intermediate_steps=True,
        )
        out = chain.invoke({"query": "Who?"})
        steps = out["intermediate_steps"]
        assert steps[0]["query"].startswith("MATCH")
        assert {r["name"] for r in steps[1]["context"]} == {"Alice", "Bob"}

    def test_an_unquoted_label_is_repaired_and_still_matches(self, graph):
        """Unrepaired, this is the failure the repair exists for: it runs and finds nothing."""
        chain = _chain(
            graph,
            "MATCH (n:Person) RETURN n.name AS name LIMIT 10",
            return_intermediate_steps=True,
        )
        out = chain.invoke({"query": "Who?"})
        assert '"Person"' in out["intermediate_steps"][0]["query"]
        assert len(out["intermediate_steps"][1]["context"]) == 2

    def test_a_mixed_case_property_is_repaired(self, graph):
        chain = _chain(
            graph,
            "MATCH (n:Person) RETURN n.firstName AS f LIMIT 10",
            return_intermediate_steps=True,
        )
        out = chain.invoke({"query": "First names?"})
        rows = out["intermediate_steps"][1]["context"]
        assert {r["f"] for r in rows} == {"Alice", "Bob"}

    def test_fences_are_stripped(self, graph):
        chain = _chain(
            graph,
            "```cypher\nMATCH (n:Person) RETURN n.name AS name LIMIT 10\n```",
            return_intermediate_steps=True,
        )
        out = chain.invoke({"query": "Who?"})
        assert "```" not in out["intermediate_steps"][0]["query"]

    def test_top_k_bounds_the_rows_handed_to_the_model(self, graph):
        chain = _chain(
            graph,
            "MATCH (n:Person) RETURN n.name AS name LIMIT 10",
            top_k=1,
            return_intermediate_steps=True,
        )
        out = chain.invoke({"query": "Who?"})
        assert len(out["intermediate_steps"][1]["context"]) == 1


class TestRefusals:
    def test_a_write_is_refused(self, graph):
        chain = _chain(graph, "CREATE (:Person {name: 'Mallory'})")
        with pytest.raises(ValueError, match="writes"):
            chain.invoke({"query": "Add someone"})
        # and nothing was written
        assert graph.query('MATCH (n:"Person") RETURN count(*) AS c')[0]["c"] == 2

    def test_a_write_is_allowed_when_explicitly_permitted(self, graph):
        chain = _chain(
            graph,
            "CREATE (:Person {name: 'Carol'})",
            allow_dangerous_requests=True,
        )
        chain.invoke({"query": "Add Carol"})
        assert graph.query('MATCH (n:"Person") RETURN count(*) AS c')[0]["c"] == 3

    def test_malformed_cypher_is_caught_before_it_runs(self, graph):
        chain = _chain(graph, "MATCH (n:Person RETURN n")
        with pytest.raises(ValueError, match="not runnable"):
            chain.invoke({"query": "Broken"})

    def test_validation_can_be_turned_off(self, graph):
        chain = _chain(graph, "MATCH (n:Person RETURN n", validate_cypher=False)
        # without EXPLAIN the failure surfaces from execution instead
        with pytest.raises(Exception):
            chain.invoke({"query": "Broken"})


class TestTool:
    def test_tool_answers_a_question(self, graph):
        tool = create_cypher_tool(
            graph,
            FakeListChatModel(
                responses=["MATCH (n:Person) RETURN n.name AS name LIMIT 10", "Two people."]
            ),
        )
        assert tool.invoke({"question": "Who?"}) == "Two people."

    def test_tool_can_return_rows_instead_of_prose(self, graph):
        tool = create_cypher_tool(
            graph,
            FakeListChatModel(responses=["MATCH (n:Person) RETURN n.name AS name LIMIT 10"]),
            answer=False,
        )
        rows = tool.invoke({"question": "Who?"})
        assert {r["name"] for r in rows} == {"Alice", "Bob"}

    def test_tool_refuses_a_write(self, graph):
        tool = create_cypher_tool(
            graph,
            FakeListChatModel(responses=["CREATE (:Person {name: 'Mallory'})"]),
            answer=False,
        )
        with pytest.raises(ValueError, match="writes"):
            tool.invoke({"question": "Add someone"})


class TestAsync:
    @pytest.mark.asyncio
    async def test_async_matches_sync(self, graph):
        chain = _chain(
            graph,
            "MATCH (n:Person) RETURN n.name AS name LIMIT 10",
            return_intermediate_steps=True,
        )
        out = await chain.ainvoke({"query": "Who?"})
        assert out["result"] == "An answer."
        assert len(out["intermediate_steps"][1]["context"]) == 2

    @pytest.mark.asyncio
    async def test_async_refuses_a_write(self, graph):
        chain = _chain(graph, "CREATE (:Person {name: 'Mallory'})")
        with pytest.raises(ValueError, match="writes"):
            await chain.ainvoke({"query": "Add someone"})
