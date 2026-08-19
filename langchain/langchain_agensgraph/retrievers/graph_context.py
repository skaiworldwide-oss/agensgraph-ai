"""Retrieve documents with their bounded graph neighbourhood, in one statement.

The similarity search finds the seeds; what makes a graph store worth asking is
what sits *around* them. This retriever appends a variable-length expansion to
the same statement the seed search runs, so the seeds and every neighbour within
``expand_by_hops`` arrive in one server round trip -- the expansion runs per
seed, after the index has already cut the candidates to k, never as one query
per seed from the client.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from pydantic import Field, PrivateAttr, model_validator

from langchain_agensgraph.retrievers.vector import AgensVectorRetriever
from langchain_agensgraph.vectorstores.agensgraph_vector import IndexType, SearchType

_RELATIONSHIP_TYPE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Every neighbour within N hops is the far end of some walk of length 1..N, and
# every edge of the neighbourhood is the LAST edge of some walk of at most N --
# so collecting the walk ends and each walk's last edge, DISTINCT, yields the
# complete deduplicated context without unwinding paths into rows. OPTIONAL
# keeps a neighbourless seed alive: collect() skips its NULL, leaving [].
# The slice bounds are query parameters; the hop bound cannot be (a
# variable-length pattern takes only literal bounds), so it is inlined after
# validation. Braces other than the store's own format keys must not appear.
_EXPANSION = """
    OPTIONAL MATCH ({anchor})-[rels{rel_type}*1..{hops}]-(peer)
    WITH node, score, id({anchor}) AS seed_id, label({anchor}) AS seed_label,
         collect(DISTINCT CASE WHEN peer IS NULL THEN NULL ELSE
             jsonb_build_object('id', id(peer), 'label', label(peer),
                 'properties', properties(peer) ||
                 jsonb_build_object({embedding_property_literal}, Null))
         END)[0..%(max_context_nodes)s] AS ctx_nodes,
         collect(DISTINCT CASE WHEN rels IS NULL THEN NULL ELSE
             jsonb_build_object('type', label(rels[-1]),
                 'start', start_id(rels[-1]), 'end', end_id(rels[-1]),
                 'properties', properties(rels[-1]))
         END)[0..%(max_context_rels)s] AS ctx_rels
    RETURN node.{text_property} AS text, score, node.__id__ AS doc_id,
           node || jsonb_build_object({text_property_literal}, Null,
               {embedding_property_literal}, Null, '__id__', Null,
               '_seed_id_', seed_id, '_seed_label_', seed_label,
               '_context_nodes_', ctx_nodes, '_context_rels_', ctx_rels) AS metadata
"""

# After the server-side rank fusion `node` is a properties map, not a vertex, so
# it cannot anchor a pattern; the seed is found again by the identity property
# the store writes. OPTIONAL, so a row the re-match misses keeps its seed and
# carries an empty context rather than vanishing.
_HYBRID_ANCHOR = """
    OPTIONAL MATCH (seed:{label}) WHERE seed.__id__ = node.__id__
"""


def build_expansion_query(
    *, hops: int, hybrid: bool, relationship_type: Optional[str] = None
) -> str:
    """The retrieval query that fetches each seed's bounded neighbourhood.

    ``hops`` is inlined -- a variable-length bound must be a literal -- so it is
    range-checked here as well as by the retriever's field. ``relationship_type``
    is inlined into the pattern and therefore accepts only a plain identifier;
    one type at most, since the server takes no alternation in a
    variable-length pattern.
    """
    if not 1 <= int(hops) <= 3:
        raise ValueError("expand_by_hops must be between 1 and 3")
    if relationship_type is not None and not _RELATIONSHIP_TYPE.match(
        relationship_type
    ):
        raise ValueError(
            "relationship_type must be a plain identifier "
            "(letters, digits and underscores, not starting with a digit); "
            f"got {relationship_type!r}"
        )
    rel_type = f':"{relationship_type}"' if relationship_type else ""
    body = _EXPANSION.replace("{hops}", str(int(hops)))
    body = body.replace("{rel_type}", rel_type)
    if hybrid:
        return _HYBRID_ANCHOR + body.replace("{anchor}", "seed")
    return body.replace("{anchor}", "node")


class AgensGraphContextRetriever(AgensVectorRetriever):
    """Similarity seeds plus their neighbourhood, one round trip.

    Each returned document is a seed the vector (or hybrid) search chose, and
    its metadata carries the seed's neighbourhood within ``expand_by_hops``:
    ``_context_nodes_`` (id, label, properties per neighbour) and
    ``_context_rels_`` (type, start, end, properties per edge), both
    deduplicated per seed and cut to ``max_context_nodes`` /
    ``max_context_rels``. The caps bound the *payload*; the traversal itself is
    bounded by the hop count, which is why the hop bound stops at three and a
    ``relationship_type`` filter exists for densely connected graphs.

    This retriever builds its own retrieval query; pass a custom one to
    :class:`AgensVectorRetriever` instead.

    Example:
        .. code-block:: python

            retriever = AgensGraphContextRetriever(
                store=store,
                k=4,
                expand_by_hops=2,
                document_formatter=render_graph_context,
            )
            docs = retriever.invoke("who reviewed the failed deploy?")
    """

    expand_by_hops: int = Field(default=1, ge=1, le=3)
    max_context_nodes: int = Field(default=20, ge=1)
    max_context_rels: int = Field(default=20, ge=1)
    relationship_type: Optional[str] = None

    _expansion_query: str = PrivateAttr(default="")

    @model_validator(mode="after")
    def _build_own_query(self) -> "AgensGraphContextRetriever":
        if self.retrieval_query is not None:
            raise ValueError(
                "AgensGraphContextRetriever builds its own retrieval query; "
                "use AgensVectorRetriever for a custom one."
            )
        if self.store._index_type != IndexType.NODE:
            raise ValueError(
                "Graph context expands from vertices; this store searches a "
                "relationship index."
            )
        self._expansion_query = build_expansion_query(
            hops=self.expand_by_hops,
            hybrid=self.store.search_type == SearchType.HYBRID,
            relationship_type=self.relationship_type,
        )
        return self

    def _effective_retrieval_query(self) -> Optional[str]:
        return self._expansion_query

    def _search_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        if "retrieval_query" in kwargs:
            raise ValueError(
                "AgensGraphContextRetriever builds its own retrieval query; "
                "use AgensVectorRetriever for a custom one."
            )
        merged = super()._search_kwargs(kwargs)
        merged["params"] = {
            **(merged["params"] or {}),
            "max_context_nodes": self.max_context_nodes,
            "max_context_rels": self.max_context_rels,
        }
        return merged


def render_graph_context(doc: Document) -> Document:
    """A ready-made ``document_formatter``: the context, written into the text.

    The structured context stays in metadata; this renders a readable account of
    it under the seed's own text, for handing straight to a prompt. A neighbour
    is named by its ``name``, ``title`` or ``text`` property, whichever it has
    first, else by its id; an edge names both ends the same way.
    """
    nodes: List[Dict[str, Any]] = doc.metadata.get("_context_nodes_") or []
    rels: List[Dict[str, Any]] = doc.metadata.get("_context_rels_") or []
    if not nodes and not rels:
        return doc

    def name_of(props: Dict[str, Any], fallback: Any) -> str:
        for key in ("name", "title", "text"):
            value = props.get(key)
            if isinstance(value, str) and value:
                return value
        return str(fallback)

    named = {
        node["id"]: "{}:{}".format(
            node["label"], name_of(node.get("properties") or {}, node["id"])
        )
        for node in nodes
    }
    lines = [f"- {name}" for name in named.values()]
    # The seed is named for the edges that point back at it, and not listed among
    # them: it is the document the context belongs to, not one of its neighbours.
    # Without a name an edge into it would read as a bare graph id.
    seed_id = doc.metadata.get("_seed_id_")
    if seed_id is not None:
        own = name_of(doc.metadata, doc.id or "this document")
        seed_label = doc.metadata.get("_seed_label_")
        named.setdefault(seed_id, f"{seed_label}:{own}" if seed_label else own)
    for rel in rels:
        start = named.get(rel["start"], str(rel["start"]))
        end = named.get(rel["end"], str(rel["end"]))
        lines.append(f"- {start} -[{rel['type']}]-> {end}")
    rendered = doc.page_content + "\n\nGraph context:\n" + "\n".join(lines)
    return Document(id=doc.id, page_content=rendered, metadata=doc.metadata)
