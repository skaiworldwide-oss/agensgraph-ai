"""Graph document, node, and relationship models.

Vendored locally so this package does not depend on the archived
``langchain-community`` (sunset 2026-05-26). The schema mirrors the one
``langchain-community`` previously exposed at
``langchain_community.graphs.graph_document`` so user code that imports
from there can be retargeted without behavioral changes.
"""

from __future__ import annotations

from typing import List, Union

from langchain_core.documents import Document
from langchain_core.load.serializable import Serializable
from pydantic import Field


class Node(Serializable):
    """A node in a graph."""

    id: Union[str, int]
    type: str = "Node"
    properties: dict = Field(default_factory=dict)


class Relationship(Serializable):
    """A directed relationship between two nodes."""

    source: Node
    target: Node
    type: str
    properties: dict = Field(default_factory=dict)


class GraphDocument(Serializable):
    """A graph document: nodes + relationships + the source ``Document``."""

    nodes: List[Node]
    relationships: List[Relationship]
    source: Document


__all__ = ["Node", "Relationship", "GraphDocument"]
