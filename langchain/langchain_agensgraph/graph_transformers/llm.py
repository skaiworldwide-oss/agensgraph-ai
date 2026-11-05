"""Extract graph documents from text with a chat model.

:class:`LLMGraphTransformer` asks a model for the entities and relationships in each document and
returns :class:`~langchain_agensgraph.graphs.graph_document.GraphDocument` objects ready for
``AgensGraph.add_graph_documents``. A model with structured output answers through it; any other
model, or any model with ``ignore_tool_usage=True``, answers in the JSON the prompt describes.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from typing import Any, Literal

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda, ensure_config
from pydantic import BaseModel, Field

from ..graphs.graph_document import GraphDocument, Node, Relationship

logger = logging.getLogger(__name__)

Value = bool | int | float | str


class _Property(BaseModel):
    key: str = Field(description="property name")
    value: Value = Field(description="property value, as a number or boolean where it is one")


class _Node(BaseModel):
    id: str = Field(description="unique, human-readable identifier of the entity")
    type: str = Field(description="the entity type, e.g. Person, Company")
    properties: list[_Property] = Field(default_factory=list)


class _Relationship(BaseModel):
    source_id: str = Field(description="id of the source node")
    source_type: str = Field(description="type of the source node")
    target_id: str = Field(description="id of the target node")
    target_type: str = Field(description="type of the target node")
    type: str = Field(description="the relationship type, e.g. WORKS_AT")
    properties: list[_Property] = Field(default_factory=list)


class _Graph(BaseModel):
    """The entities and relationships in a piece of text."""

    nodes: list[_Node] = Field(default_factory=list)
    relationships: list[_Relationship] = Field(default_factory=list)


class LLMGraphTransformer:
    """Turn documents into graph documents with a chat model.

    ``allowed_nodes`` and ``allowed_relationships`` name the types to extract; a relationship may
    be a type or a ``(source type, type, target type)`` triple. With ``strict_mode`` anything
    else the model returns is dropped, and a triple's direction must match. A type the caller
    named is written as the caller spells it, whatever case the model used.

    ``node_properties`` and ``relationship_properties`` are True for any property, or a list of
    the property names to extract. ``additional_instructions`` is added to the prompt; ``prompt``
    replaces it. ``max_concurrency`` bounds the documents converted at once, and ``on_error``
    says what a failed document does: ``"raise"``, or ``"skip"`` to log it and leave it out.
    """

    def __init__(
        self,
        llm: Any,
        allowed_nodes: Sequence[str] = (),
        allowed_relationships: Sequence[str] | Sequence[tuple[str, str, str]] = (),
        *,
        node_properties: bool | Sequence[str] = False,
        relationship_properties: bool | Sequence[str] = False,
        strict_mode: bool = True,
        prompt: str | None = None,
        additional_instructions: str = "",
        ignore_tool_usage: bool = False,
        max_concurrency: int = 8,
        on_error: Literal["raise", "skip"] = "raise",
    ) -> None:
        if on_error not in ("raise", "skip"):
            raise ValueError(f"on_error is 'raise' or 'skip', not {on_error!r}")
        if max_concurrency < 1:
            raise ValueError(f"max_concurrency must be positive, not {max_concurrency}")
        self.allowed_nodes = list(allowed_nodes)
        self.allowed_relationships: list[str | tuple[str, str, str]] = list(allowed_relationships)
        self.node_properties = node_properties
        self.relationship_properties = relationship_properties
        self.strict_mode = strict_mode
        self.additional_instructions = additional_instructions
        self.max_concurrency = max_concurrency
        self.on_error = on_error
        self._node_case = {name.lower(): name for name in self.allowed_nodes}
        self._triples, self._rel_case = _relationships(self.allowed_relationships)
        if self._triples and self.allowed_nodes:
            unknown = {
                end
                for start, _, finish in self._triples
                for end in (start, finish)
                if end not in self._node_case
            }
            if unknown:
                raise ValueError(
                    f"relationship ends {sorted(unknown)} are not among the allowed nodes"
                )
        answer, instructions = _answering(llm, ignore_tool_usage)
        self._system = "\n".join(
            part
            for part in (prompt or self._default_prompt(), instructions, additional_instructions)
            if part
        )
        self._extract: Runnable[str, Any] = (RunnableLambda(self._messages) | answer).with_config(
            run_name="LLMGraphTransformer"
        )

    def _default_prompt(self) -> str:
        parts = [
            "You are an information extraction system. Extract the entities (nodes) in the "
            "user's text and the relationships between them.",
            "Use the entity's natural name as its id, and the same id each time the same entity "
            "appears, so the graph stays connected.",
        ]
        if self.allowed_nodes:
            parts.append("Extract only nodes of these types: " + ", ".join(self.allowed_nodes))
        if self._triples:
            parts.append(
                "Extract only these relationships, in this direction, as "
                "(source type, relationship type, target type): "
                + ", ".join(f"({', '.join(item)})" for item in self.allowed_relationships)
            )
        elif self.allowed_relationships:
            parts.append(
                "Extract only relationships of these types: "
                + ", ".join(map(str, self.allowed_relationships))
            )
        parts.append(_properties_rule("node", self.node_properties))
        parts.append(_properties_rule("relationship", self.relationship_properties))
        return "\n".join(parts)

    def _messages(self, text: str) -> list[SystemMessage | HumanMessage]:
        return [SystemMessage(content=self._system), HumanMessage(content=text)]

    # -- converting ------------------------------------------------------------------------

    def process_response(
        self, document: Document, config: RunnableConfig | None = None
    ) -> GraphDocument:
        """The graph document for one document."""
        return self._document(_graph(self._extract.invoke(document.page_content, config)), document)

    async def aprocess_response(
        self, document: Document, config: RunnableConfig | None = None
    ) -> GraphDocument:
        found = await self._extract.ainvoke(document.page_content, config)
        return self._document(_graph(found), document)

    def convert_to_graph_documents(
        self, documents: Sequence[Document], config: RunnableConfig | None = None
    ) -> list[GraphDocument]:
        """The graph documents for documents, in their order, converted ``max_concurrency`` at
        a time."""
        found = self._extract.batch(
            [document.page_content for document in documents],
            self._config(config),
            return_exceptions=self.on_error == "skip",
        )
        return self._documents(documents, found)

    async def aconvert_to_graph_documents(
        self, documents: Sequence[Document], config: RunnableConfig | None = None
    ) -> list[GraphDocument]:
        found = await self._extract.abatch(
            [document.page_content for document in documents],
            self._config(config),
            return_exceptions=self.on_error == "skip",
        )
        return self._documents(documents, found)

    def _config(self, config: RunnableConfig | None) -> RunnableConfig:
        given = ensure_config(config)
        if given.get("max_concurrency") is None:
            given["max_concurrency"] = self.max_concurrency
        return given

    def _documents(self, documents: Sequence[Document], found: list[Any]) -> list[GraphDocument]:
        """Each answer turned into a graph document. With ``on_error="skip"`` a document whose
        model call failed, or whose answer is no graph, is left out with a warning."""
        out = []
        for document, result in zip(documents, found, strict=True):
            if isinstance(result, asyncio.CancelledError):
                raise result
            try:
                if isinstance(result, BaseException):
                    raise result
                out.append(self._document(_graph(result), document))
            except Exception as exc:
                if self.on_error == "raise":
                    raise
                logger.warning("LLMGraphTransformer: a document was left out: %s", exc)
        return out

    # -- shaping the answer ----------------------------------------------------------------

    def _document(self, graph: _Graph, source: Document) -> GraphDocument:
        nodes: dict[tuple[str, str], Node] = {}
        for node in graph.nodes:
            kind = self._node_type(node.type)
            if self.strict_mode and not self._known_node(node.type):
                continue
            properties = _kept(node.properties, self.node_properties)
            nodes[(node.id, kind)] = Node(id=node.id, type=kind, properties=properties)
        relationships = []
        for edge in graph.relationships:
            if self.strict_mode and not self._known_relationship(edge):
                continue
            start = self._node_type(edge.source_type)
            end = self._node_type(edge.target_type)
            relationships.append(
                Relationship(
                    source=nodes.get((edge.source_id, start))
                    or Node(id=edge.source_id, type=start),
                    target=nodes.get((edge.target_id, end)) or Node(id=edge.target_id, type=end),
                    type=self._rel_case.get(edge.type.lower(), edge.type),
                    properties=_kept(edge.properties, self.relationship_properties),
                )
            )
        return GraphDocument(nodes=list(nodes.values()), relationships=relationships, source=source)

    def _node_type(self, kind: str) -> str:
        return self._node_case.get(kind.lower(), kind)

    def _known_node(self, kind: str) -> bool:
        return not self.allowed_nodes or kind.lower() in self._node_case

    def _known_relationship(self, found: _Relationship) -> bool:
        if not (self._known_node(found.source_type) and self._known_node(found.target_type)):
            return False
        if self._triples:
            triple = (found.source_type.lower(), found.type.lower(), found.target_type.lower())
            return triple in self._triples
        return not self._rel_case or found.type.lower() in self._rel_case


def _relationships(
    allowed: Sequence[str | tuple[str, str, str]],
) -> tuple[set[tuple[str, str, str]], dict[str, str]]:
    """The allowed triples, folded, and each allowed type by its folded spelling."""
    triples = [item for item in allowed if isinstance(item, tuple)]
    if triples and len(triples) != len(allowed):
        raise ValueError("allowed_relationships holds types or triples, not both")
    if any(len(item) != 3 for item in triples):
        raise ValueError("an allowed relationship triple is (source type, type, target type)")
    names = [item[1] if isinstance(item, tuple) else item for item in allowed]
    folded = {(s.lower(), r.lower(), t.lower()) for s, r, t in triples}
    return folded, {name.lower(): name for name in names}


def _answering(llm: Any, ignore_tool_usage: bool) -> tuple[Runnable[Any, Any], str]:
    """How the model answers, and what the prompt must then say about the form of the answer."""
    if not ignore_tool_usage and hasattr(llm, "with_structured_output"):
        try:
            return llm.with_structured_output(_Graph), ""
        except NotImplementedError:
            pass
    if not isinstance(llm, Runnable):
        raise ValueError("LLMGraphTransformer needs a chat model")
    parser = PydanticOutputParser(pydantic_object=_Graph)
    return llm | parser, parser.get_format_instructions()


def _properties_rule(what: str, wanted: bool | Sequence[str]) -> str:
    if wanted is True:
        return f"Extract {what} properties as key and value."
    if wanted:
        return f"Extract only these {what} properties, as key and value: " + ", ".join(wanted)
    return f"Extract no {what} properties."


def _kept(properties: Sequence[_Property], wanted: bool | Sequence[str]) -> dict[str, Value]:
    if wanted is True:
        return {item.key: item.value for item in properties}
    if not wanted:
        return {}
    names = {name.lower(): name for name in wanted}
    return {names[item.key.lower()]: item.value for item in properties if item.key.lower() in names}


def _graph(found: Any) -> _Graph:
    if isinstance(found, _Graph):
        return found
    if isinstance(found, dict):
        return _Graph.model_validate(found)
    raise TypeError(f"the model answered with {type(found).__name__}, not a graph")


__all__ = ["LLMGraphTransformer"]
