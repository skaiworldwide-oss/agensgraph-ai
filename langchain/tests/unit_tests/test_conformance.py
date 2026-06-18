"""GraphStore-protocol conformance for AgensGraph.

These assert the surface that a Cypher graph-QA chain reads off a graph
store, so AgensGraph can be driven by such a chain.
"""

from __future__ import annotations

from langchain_agensgraph.graphs.agensgraph import AgensGraph
from langchain_agensgraph.graphs.graph_store import GraphStore


def test_is_graphstore_subclass():
    assert issubclass(AgensGraph, GraphStore)


def test_exposes_chain_facing_surface():
    # GraphCypherQAChain reads `get_schema` and calls `query`; the LLMGraph
    # ingestion path uses `add_graph_documents`; `refresh_schema` keeps it fresh.
    for name in ("get_schema", "get_structured_schema", "query", "refresh_schema",
                 "add_graph_documents"):
        assert hasattr(AgensGraph, name), name


def test_async_surface_present():
    for name in ("aquery", "aclose"):
        assert hasattr(AgensGraph, name), name
