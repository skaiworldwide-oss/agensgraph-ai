"""What a Cypher QA chain asks of a graph, as a protocol.

The same shape as ``langchain_neo4j.graphs.graph_store.GraphStore``, so code written against one
reads the other. Blocking only; :class:`~langchain_agensgraph.AgensGraph` adds the awaiting forms.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from langchain_agensgraph.graphs.graph_document import GraphDocument


@runtime_checkable
class GraphStore(Protocol):
    """A graph a chain can describe to a model, query, and write graph documents into."""

    @property
    def get_schema(self) -> str:
        """The schema as text for a prompt."""
        ...

    @property
    def get_structured_schema(self) -> dict[str, Any]:
        """The schema as a mapping: ``node_props``, ``rel_props``, ``relationships``."""
        ...

    def query(self, query: str, params: Any = None) -> list[dict[str, Any]]:
        """Run a statement; its rows as mappings."""
        ...

    def refresh_schema(self) -> None:
        """Read the schema again."""
        ...

    def add_graph_documents(
        self, graph_documents: Sequence[GraphDocument], include_source: bool = False
    ) -> None:
        """Write graph documents into the graph."""
        ...


__all__ = ["GraphStore"]
