"""What the graph transformer extracts goes into a graph as it came out: typed values, the
caller's spelling, and each source document."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from langchain_agensgraph._core.engine import AgensEngine
from langchain_agensgraph.graph_transformers import LLMGraphTransformer
from langchain_agensgraph.graphs import AgensGraph

pytestmark = pytest.mark.server

PAYLOAD = {
    "nodes": [
        {"id": "Alice", "type": "person", "properties": [{"key": "age", "value": 36}]},
        {"id": "Acme", "type": "Company", "properties": [{"key": "public", "value": True}]},
    ],
    "relationships": [
        {
            "source_id": "Alice",
            "source_type": "PERSON",
            "target_id": "Acme",
            "target_type": "company",
            "type": "works_at",
            "properties": [{"key": "since", "value": 2019}],
        }
    ],
}


class StructuredLLM:
    def with_structured_output(self, schema: Any, **kwargs: Any) -> RunnableLambda[Any, dict]:
        return RunnableLambda(lambda _messages: dict(PAYLOAD))


@pytest.fixture
def graph(dsn: str, graph_name: str) -> Iterator[AgensGraph]:
    with AgensEngine(dsn) as engine:
        yield AgensGraph(engine, graph_name=graph_name, create_graph=True)


def test_what_the_transformer_extracts_is_ingested_as_extracted(graph: AgensGraph) -> None:
    transformer = LLMGraphTransformer(
        StructuredLLM(),
        allowed_nodes=["Person", "Company"],
        allowed_relationships=[("Person", "WORKS_AT", "Company")],
        node_properties=True,
        relationship_properties=True,
    )
    documents = transformer.convert_to_graph_documents(
        [Document(page_content="Alice, 36, works at Acme.", metadata={"id": "d1"})]
    )
    graph.add_graph_documents(documents, include_source=True)
    rows = graph.query(
        'MATCH (p:"Person")-[w:"WORKS_AT"]->(c:"Company") '
        "RETURN p.id AS who, p.age AS age, c.public AS public, w.since AS since"
    )
    assert rows == [{"who": "Alice", "age": 36, "public": True, "since": 2019}]
    mentions = graph.query(
        'MATCH (d:"Document")-[:"MENTIONS"]->(n) RETURN d.id AS doc, count(*) AS n'
    )
    assert mentions == [{"doc": "d1", "n": 2}]
