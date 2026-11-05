"""The QA chain, its tool and the text2cypher retriever against a server, with scripted models.

Model-written Cypher runs as ``llm_ro`` and, for writes, ``llm_rw``, after ``render_grants``; the
assertions are about what the pipeline sends and what comes back, not about a model.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import psycopg
import pytest
from agensgraph.errors import CapabilityError
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import BaseTool
from langchain_tests.integration_tests import ToolsIntegrationTests
from psycopg.conninfo import make_conninfo

from langchain_agensgraph._core.engine import AgensEngine
from langchain_agensgraph.chains import (
    DIALECT_EXAMPLES,
    AgensCypherQAChain,
    create_cypher_tool,
    prepare,
)
from langchain_agensgraph.chains.cypher_qa import _EXAMPLE_NEEDS
from langchain_agensgraph.errors import UnsafeCypherError
from langchain_agensgraph.graphs import AgensGraph, GraphDocument, Node, Relationship
from langchain_agensgraph.retrievers import AgensText2CypherRetriever
from langchain_agensgraph.roles import render_grants

pytestmark = pytest.mark.server


def _world() -> GraphDocument:
    """One of everything the dialect examples ask about, so each of them finds a row."""

    def node(key: str, label: str, **properties: Any) -> Node:
        return Node(id=key, type=label, properties=properties)

    ada, bob = node("ada", "Person", name="Ada", age=36), node("bob", "Person", name="Bob", age=25)
    acme, prolog = node("acme", "Company", name="Acme"), node("prolog", "Product", name="Prolog")
    author = node("knuth", "Author", name="Knuth")
    papers = [node(f"p{i}", "Paper", title=f"Paper {i}") for i in range(6)]
    customer, movie = node("c1", "Customer", name="Cy"), node("m1", "Movie", title="Heat")
    genres = [node("crime", "Genre", name="Crime"), node("drama", "Genre", name="Drama")]
    nodes = [ada, bob, acme, prolog, author, customer, movie, *papers, *genres]
    nodes += [
        node("b1", "Book", title="graph theory"),
        node("i1", "Item", name="lamp", price=60),
        node("i2", "Item", name="pen", price=10),
        node("k1", "Code", value="123"),
    ]
    edges = [
        Relationship(source=ada, target=acme, type="WORKS_AT"),
        Relationship(source=ada, target=bob, type="FRIENDS_WITH"),
        Relationship(source=customer, target=prolog, type="ORDERED"),
        *(Relationship(source=paper, target=author, type="AUTHORED_BY") for paper in papers),
        *(Relationship(source=movie, target=genre, type="IN_GENRE") for genre in genres),
    ]
    return GraphDocument(nodes=nodes, relationships=edges)


@pytest.fixture
def graph(dsn: str, graph_name: str) -> Iterator[AgensGraph]:
    """The world, readable by ``llm_ro`` and writable by ``llm_rw``, through an engine that runs
    model-written Cypher as them."""
    with AgensEngine(dsn) as owner:
        AgensGraph(owner, graph_name=graph_name, create_graph=True).add_graph_documents([_world()])
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute(render_grants(graph_name, writer_role="llm_rw"))
    with AgensEngine(
        dsn,
        untrusted_conninfo=make_conninfo(dsn, user="llm_ro", password="agens"),
        writer_conninfo=make_conninfo(dsn, user="llm_rw", password="agens"),
    ) as engine:
        yield AgensGraph(engine, graph_name=graph_name)


def scripted(*replies: str) -> RunnableLambda[Any, AIMessage]:
    queue = list(replies)
    return RunnableLambda(lambda _prompt: AIMessage(content=queue.pop(0)))


def qa(graph: AgensGraph, *replies: str, **kwargs: Any) -> AgensCypherQAChain:
    return AgensCypherQAChain.from_llm(
        graph=graph,
        cypher_llm=scripted(*replies),
        qa_llm=scripted("An answer."),
        allow_dangerous_requests=True,
        **kwargs,
    )


def test_each_dialect_example_offered_to_a_release_runs_on_it(graph: AgensGraph) -> None:
    caps = graph.capabilities
    for question, cypher in DIALECT_EXAMPLES:
        needs = _EXAMPLE_NEEDS.get(cypher)
        if needs is not None and not caps.has(needs):
            continue
        rows = graph.query_untrusted(prepare(cypher, caps), max_rows=10)
        assert rows or "graph id" in question, question


def test_a_question_becomes_a_statement_run_as_the_untrusted_role(graph: AgensGraph) -> None:
    chain = qa(graph, "MATCH (p:Person) RETURN p.name AS name ORDER BY name", return_direct=True)
    out = chain.invoke("Who is there?")
    assert out["result"] == [{"name": "Ada"}, {"name": "Bob"}]


def test_what_the_server_or_the_reading_refuses_goes_back_to_the_model(
    graph: AgensGraph,
) -> None:
    chain = qa(
        graph,
        'MATCH (p:"Person") SET p.seen = true RETURN p.name AS name',
        'MATCH (p:"Person") RETURN p.name AS name ORDER',
        'MATCH (p:"Person") WHERE p.age > 30 RETURN p.name AS name',
        return_direct=True,
        return_intermediate_steps=True,
    )
    out = chain.invoke("Who is over thirty?")
    assert out["result"] == [{"name": "Ada"}]
    refused, failed = out["intermediate_steps"][:2]
    assert "a write" in refused["error"] and "syntax error" in failed["error"]
    assert graph.query('MATCH (p:"Person") WHERE p.seen IS NOT NULL RETURN p') == []


def test_rows_are_read_up_to_top_k(graph: AgensGraph) -> None:
    out = qa(graph, 'MATCH (p:"Paper") RETURN p.title AS t', top_k=4, return_direct=True).invoke(
        "Papers?"
    )
    assert len(out["result"]) == 4


def test_writes_run_as_the_writer_role_where_the_server_checks_them(graph: AgensGraph) -> None:
    chain = qa(
        graph,
        "MATCH (p:\"Person\") WHERE p.name = 'Bob' SET p.retired = true RETURN p.name AS name",
        allow_writes=True,
        return_direct=True,
    )
    if not graph.capabilities.has("cypher_write_enforced"):
        with pytest.raises(CapabilityError):
            chain.invoke("Retire Bob.")
        return
    assert chain.invoke("Retire Bob.")["result"] == [{"name": "Bob"}]
    assert graph.query('MATCH (p:"Person") WHERE p.retired = true RETURN p.name AS n') == [
        {"n": "Bob"}
    ]


async def test_awaiting_the_chain_and_the_retriever(graph: AgensGraph) -> None:
    out = await qa(graph, 'MATCH (c:"Company") RETURN c.name AS name').ainvoke("Companies?")
    assert out == {"query": "Companies?", "result": "An answer."}
    retriever = AgensText2CypherRetriever(
        graph=graph,
        llm=scripted("MATCH (p:\"Person\") WHERE p.name = 'Ada' RETURN p.age AS age"),
        allow_dangerous_requests=True,
    )
    (document,) = await retriever.ainvoke("How old is Ada?")
    assert json.loads(document.page_content) == {"age": 36}
    await graph.engine.aclose()


def test_text_the_reading_refuses_never_reaches_the_server(graph: AgensGraph) -> None:
    with pytest.raises(UnsafeCypherError, match="more than one statement"):
        graph.query_untrusted('MATCH (n) RETURN n; DROP GRAPH "x" CASCADE')


class TestCypherToolStandard(ToolsIntegrationTests):
    @pytest.fixture(autouse=True)
    def _graph(self, graph: AgensGraph) -> None:
        self.graph = graph

    @property
    def tool_constructor(self) -> BaseTool:
        return create_cypher_tool(
            self.graph,
            RunnableLambda(
                lambda _prompt: AIMessage(content='MATCH (p:"Person") RETURN count(*) AS n')
            ),
            answer=False,
            allow_dangerous_requests=True,
        )

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"question": "How many people are there?"}
