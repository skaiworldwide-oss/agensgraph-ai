"""Graph store base class.

Vendored locally so this package does not depend on the archived
``langchain-community``. It is a lightweight ABC mirroring the interface
that Cypher graph-QA chains expect.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, List

from langchain_agensgraph.graphs.graph_document import GraphDocument


class GraphStore:
    """Abstract interface for a graph database backing a LangChain graph store."""

    @property
    @abstractmethod
    def get_schema(self) -> str:
        """Schema of the graph database as a human-readable string."""

    @property
    @abstractmethod
    def get_structured_schema(self) -> Dict[str, Any]:
        """Schema as a structured dict (node_props, rel_props, relationships)."""

    @abstractmethod
    def query(self, query: str, params: dict = {}) -> List[Dict[str, Any]]:
        """Run a Cypher query and return result rows."""

    @abstractmethod
    def refresh_schema(self) -> None:
        """Re-read the schema from the live database."""

    @abstractmethod
    def add_graph_documents(
        self,
        graph_documents: List[GraphDocument],
        include_source: bool = False,
    ) -> None:
        """Persist a list of ``GraphDocument`` instances into the graph."""


__all__ = ["GraphStore"]
