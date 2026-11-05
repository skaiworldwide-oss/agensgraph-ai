"""The graph store has the shape a Cypher QA chain asks of a graph, blocking and awaiting."""

from __future__ import annotations

from langchain_agensgraph.graphs import AgensGraph, GraphStore


def test_the_graph_store_is_a_graph_store() -> None:
    graph = AgensGraph("host=nowhere dbname=none", graph_name="g")
    try:
        assert isinstance(graph, GraphStore)
        for name in ("query", "refresh_schema", "add_graph_documents", "query_untrusted"):
            assert callable(getattr(graph, name))
            assert callable(getattr(graph, f"a{name}"))
    finally:
        graph.close()
