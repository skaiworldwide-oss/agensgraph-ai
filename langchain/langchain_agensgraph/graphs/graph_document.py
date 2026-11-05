"""Graph document, node, and relationship models.

The same models as ``langchain_community.graphs.graph_document``, kept here so the package does
not depend on ``langchain-community``; code importing them from there can import them from here.
"""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from langchain_core.load.serializable import Serializable
from pydantic import Field


class Node(Serializable):
    """A node in a graph."""

    id: str | int
    type: str = "Node"
    properties: dict[str, Any] = Field(default_factory=dict)


class Relationship(Serializable):
    """A directed relationship between two nodes."""

    source: Node
    target: Node
    type: str
    properties: dict[str, Any] = Field(default_factory=dict)


class GraphDocument(Serializable):
    """A graph document: nodes + relationships + the source ``Document``, if there is one."""

    nodes: list[Node]
    relationships: list[Relationship]
    source: Document | None = None


__all__ = ["Node", "Relationship", "GraphDocument"]
