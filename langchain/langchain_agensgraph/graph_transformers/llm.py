"""Extract graph documents from text with an LLM.

A dependency-light text-to-graph transformer: it drives any chat model that
supports ``with_structured_output`` to turn unstructured text into
:class:`~langchain_agensgraph.graphs.graph_document.GraphDocument` objects
ready for ``AgensGraph.add_graph_documents``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, List, Optional, Sequence

from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from langchain_agensgraph.graphs.graph_document import (
    GraphDocument,
    Node,
    Relationship,
)

logger = logging.getLogger(__name__)


class _Property(BaseModel):
    """A single key/value property on a node or relationship."""

    key: str = Field(description="property name")
    value: str = Field(description="property value")


class _Node(BaseModel):
    id: str = Field(description="unique, human-readable identifier of the entity")
    type: str = Field(description="the entity type / label, e.g. Person, Company")
    properties: List[_Property] = Field(default_factory=list)


class _Relationship(BaseModel):
    source_id: str = Field(description="id of the source node")
    source_type: str = Field(description="type of the source node")
    target_id: str = Field(description="id of the target node")
    target_type: str = Field(description="type of the target node")
    type: str = Field(description="the relationship type, e.g. WORKS_AT")
    properties: List[_Property] = Field(default_factory=list)


class _GraphSchema(BaseModel):
    """Structured graph extracted from a piece of text."""

    nodes: List[_Node] = Field(default_factory=list)
    relationships: List[_Relationship] = Field(default_factory=list)


def _props_to_dict(props: Sequence[_Property]) -> dict:
    return {p.key: p.value for p in props}


class LLMGraphTransformer:
    """Transform documents into graph documents using an LLM.

    Args:
        llm: A chat model supporting ``with_structured_output``.
        allowed_nodes: If non-empty, only these node types are kept.
        allowed_relationships: If non-empty, only these relationship types are
            kept.
        node_properties: If True, the LLM is asked to extract node properties.
        strict_mode: If True (default), nodes/relationships outside the allowed
            lists are filtered out after extraction.
        prompt: Optional override for the system prompt.
    """

    def __init__(
        self,
        llm: BaseLanguageModel,
        *,
        allowed_nodes: Optional[List[str]] = None,
        allowed_relationships: Optional[List[str]] = None,
        node_properties: bool = False,
        strict_mode: bool = True,
        prompt: Optional[str] = None,
    ) -> None:
        if not hasattr(llm, "with_structured_output"):
            raise ValueError(
                "LLMGraphTransformer requires an LLM that supports "
                "`with_structured_output`."
            )
        self.allowed_nodes = allowed_nodes or []
        self.allowed_relationships = allowed_relationships or []
        self.node_properties = node_properties
        self.strict_mode = strict_mode
        # A type is admitted whatever its case, and then written in the case the caller
        # asked for. A model that answers "PERSON" where the list says "Person" is naming
        # the allowed type, but a label is case-sensitive, so keeping the model's spelling
        # would put those elements on a second label that no query for the first will find.
        self._node_case = {name.lower(): name for name in self.allowed_nodes}
        self._rel_case = {name.lower(): name for name in self.allowed_relationships}
        self._system_prompt = prompt or self._default_prompt()
        self._structured = llm.with_structured_output(_GraphSchema)

    def _default_prompt(self) -> str:
        parts = [
            "You are an information extraction system. Extract entities (nodes) "
            "and the relationships between them from the user's text.",
            "Use the entity's natural name as its `id`. Reuse the same id when "
            "the same entity appears again so the graph stays connected.",
        ]
        if self.allowed_nodes:
            parts.append(
                "Only extract nodes of these types: "
                + ", ".join(self.allowed_nodes)
                + "."
            )
        if self.allowed_relationships:
            parts.append(
                "Only extract relationships of these types: "
                + ", ".join(self.allowed_relationships)
                + "."
            )
        if not self.node_properties:
            parts.append("Do not extract node or relationship properties.")
        return " ".join(parts)

    def _messages(self, text: str):
        return [
            SystemMessage(content=self._system_prompt),
            HumanMessage(content=text),
        ]

    # ---- filtering ----

    def _keep_node(self, type_: str) -> bool:
        if not self.allowed_nodes:
            return True
        return type_.lower() in {a.lower() for a in self.allowed_nodes}

    def _keep_rel(self, type_: str) -> bool:
        if not self.allowed_relationships:
            return True
        return type_.lower() in {a.lower() for a in self.allowed_relationships}

    def _node_type(self, type_: str) -> str:
        """The node type as the caller spelled it, for one they named."""
        return self._node_case.get(type_.lower(), type_)

    def _rel_type(self, type_: str) -> str:
        """The relationship type as the caller spelled it, for one they named."""
        return self._rel_case.get(type_.lower(), type_)

    def _to_graph_document(
        self, schema: _GraphSchema, source: Document
    ) -> GraphDocument:
        # Keyed by the type as the caller spells it, which is also how a relationship's
        # endpoint is looked up. A model naming the same entity `Person` here and
        # `PERSON` in a relationship is naming one entity, and keying on what it wrote
        # would miss the lookup and build a second node of the same type carrying none of
        # the properties -- which then wins, being written last.
        nodes: dict = {}
        for n in schema.nodes:
            if self.strict_mode and not self._keep_node(n.type):
                continue
            props = _props_to_dict(n.properties) if self.node_properties else {}
            nodes[(n.id, self._node_type(n.type))] = Node(
                id=n.id, type=self._node_type(n.type), properties=props
            )

        rels: List[Relationship] = []
        for r in schema.relationships:
            if self.strict_mode and not self._keep_rel(r.type):
                continue
            if self.strict_mode and not (
                self._keep_node(r.source_type) and self._keep_node(r.target_type)
            ):
                continue
            source_node = nodes.get(
                (r.source_id, self._node_type(r.source_type))
            ) or Node(id=r.source_id, type=self._node_type(r.source_type))
            target_node = nodes.get(
                (r.target_id, self._node_type(r.target_type))
            ) or Node(id=r.target_id, type=self._node_type(r.target_type))
            props = _props_to_dict(r.properties) if self.node_properties else {}
            rels.append(
                Relationship(
                    source=source_node,
                    target=target_node,
                    type=self._rel_type(r.type),
                    properties=props,
                )
            )
        return GraphDocument(
            nodes=list(nodes.values()), relationships=rels, source=source
        )

    @staticmethod
    def _coerce(result: Any) -> _GraphSchema:
        if isinstance(result, _GraphSchema):
            return result
        if isinstance(result, dict):
            return _GraphSchema(**result)
        raise TypeError(f"Unexpected structured-output result: {type(result)!r}")

    # ---- public API ----

    def process_response(self, document: Document) -> GraphDocument:
        result = self._structured.invoke(self._messages(document.page_content))
        return self._to_graph_document(self._coerce(result), document)

    def convert_to_graph_documents(
        self, documents: Sequence[Document]
    ) -> List[GraphDocument]:
        return [self.process_response(d) for d in documents]

    async def aprocess_response(self, document: Document) -> GraphDocument:
        result = await self._structured.ainvoke(self._messages(document.page_content))
        return self._to_graph_document(self._coerce(result), document)

    async def aconvert_to_graph_documents(
        self, documents: Sequence[Document]
    ) -> List[GraphDocument]:
        """Convert documents concurrently, isolating per-document failures.

        Extractions run with ``return_exceptions=True`` so a single document that errors
        -- an answer that hit the output-token limit, say -- is logged and skipped rather
        than abandoning the whole batch. Returns the documents that were extracted, which
        may be fewer than were asked for.

        Cancellation is not one of those failures. ``CancelledError`` is a
        ``BaseException`` rather than an ``Exception``, so it arrives here like any other
        result, and it is raised on rather than logged: a caller that cancelled this is
        owed the cancellation, not a shorter list.
        """
        results = await asyncio.gather(
            *(self.aprocess_response(d) for d in documents),
            return_exceptions=True,
        )
        out: List[GraphDocument] = []
        for result in results:
            if isinstance(result, asyncio.CancelledError):
                raise result
            if isinstance(result, BaseException):
                logger.warning(
                    "LLMGraphTransformer: skipping a document after extraction "
                    "error: %s",
                    result,
                )
            else:
                out.append(result)
        return out


__all__ = ["LLMGraphTransformer"]
