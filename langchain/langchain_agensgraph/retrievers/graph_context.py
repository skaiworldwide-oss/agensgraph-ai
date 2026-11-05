"""Retrieve chunks with the part of the graph around them.

The similarity search finds the seeds; what makes a graph store worth asking is what sits around
them. After the search, one SQL statement walks out from every seed at once, up to ``hops`` edges
in either direction, reading at most a cap of edges from each node it reaches, so a hub cannot
flood the result. The walk reads the graph's ``ag_edge`` and ``ag_vertex``, which every edge and
vertex label inherits from, or one relationship type's own table.

Each document's metadata carries the context: ``_context_nodes_`` (id, label and properties of
each neighbour) and ``_context_rels_`` (type, start, end and properties of each edge), each
deduplicated and cut to its cap, beside ``_seed_id_`` and ``_seed_label_`` naming the chunk's own
vertex. :func:`render_graph_context` writes them into the text for a prompt.

From 2.18.6 a read through ``ag_vertex`` or ``ag_edge`` checks the reader's grants on every label,
so a role missing one is refused; the refusal names the labels to grant.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import psycopg
from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from pydantic import Field

from langchain_agensgraph._core.names import checked, ident, qualified
from langchain_agensgraph._core.program import Program, Statement
from langchain_agensgraph.errors import AgensGraphError
from langchain_agensgraph.retrievers.vector import AgensVectorRetriever


def neighbourhood(graph: str, relationship_type: str | None) -> str:
    """The walk from ``%s`` seed ids, at most ``%s`` edges per node, at most ``%s`` hops deep,
    cut to the first ``%s`` nodes and ``%s`` edges of each seed.

    One row per step: seed, node reached, its label and properties, the edge taken, its start,
    end, label and properties, in the order seed, depth, node, edge. A seed's first nodes and
    edges are those that order reaches first, which are the ones a reader keeping that many of
    each keeps; a step is returned only when its node or its edge is one of them, so the vertex
    of every other step is never read.
    """
    edges = (
        qualified(graph, relationship_type)
        if relationship_type is not None
        else f"{ident(graph)}.ag_edge"
    )
    return (
        "WITH RECURSIVE walk(seed, node, depth, edge, estart, eend, eoid, eprops) AS ("
        "SELECT s, s, 0, NULL::graphid, NULL::graphid, NULL::graphid, NULL::oid, NULL::jsonb "
        "FROM unnest(%s::graphid[]) AS s "
        "UNION ALL "
        'SELECT w.seed, n.other, w.depth + 1, n.id, n.start, n."end", n.tableoid, n.properties '
        "FROM walk w, LATERAL ("
        'SELECT e.id, e.start, e."end", e.tableoid, e.properties, '
        'CASE WHEN e.start = w.node THEN e."end" ELSE e.start END AS other '
        f'FROM {edges} e WHERE e.start = w.node OR e."end" = w.node '
        "ORDER BY e.id LIMIT %s) n "
        "WHERE w.depth < %s), "
        "steps AS (SELECT * FROM walk WHERE depth > 0), "
        "nodes AS (SELECT seed, node FROM ("
        "SELECT seed, node, row_number() OVER (PARTITION BY seed ORDER BY min(depth), node) AS n "
        "FROM steps WHERE node <> seed GROUP BY seed, node) ranked WHERE n <= %s), "
        "edges AS (SELECT seed, edge FROM ("
        "SELECT seed, edge, row_number() OVER (PARTITION BY seed ORDER BY depth, node, edge) AS n "
        "FROM (SELECT DISTINCT ON (seed, edge) seed, edge, depth, node FROM steps "
        "ORDER BY seed, edge, depth, node) first) ranked WHERE n <= %s), "
        "kept AS (SELECT s.* FROM steps s "
        "WHERE EXISTS (SELECT 1 FROM nodes k WHERE k.seed = s.seed AND k.node = s.node) "
        "OR EXISTS (SELECT 1 FROM edges k WHERE k.seed = s.seed AND k.edge = s.edge)) "
        "SELECT w.seed, w.node, vl.labname, v.properties, w.edge, w.estart, w.eend, "
        "el.labname, w.eprops "
        f"FROM kept w JOIN {ident(graph)}.ag_vertex v ON v.id = w.node "
        "JOIN ag_label vl ON vl.relid = v.tableoid "
        "LEFT JOIN ag_label el ON el.relid = w.eoid "
        "ORDER BY w.seed, w.depth, w.node, w.edge"
    )


UNGRANTED = (
    "SELECT l.labname FROM ag_label l JOIN ag_graph g ON g.oid = l.graphid "
    "WHERE g.graphname = %s AND NOT has_table_privilege(l.relid, 'SELECT') ORDER BY 1"
)


def walk(
    graph: str,
    seeds: Sequence[Any],
    *,
    hops: int,
    cap: int,
    nodes: int,
    rels: int,
    relationship_type: str | None,
) -> Program[list[tuple[Any, ...]]]:
    """The walk's steps (:func:`neighbourhood`); a role refused a label is told which."""
    if not seeds:
        return []
    try:
        result = yield Statement(
            neighbourhood(graph, relationship_type),
            (list(seeds), cap, hops, nodes, rels),
            may_fail=True,
        )
    except psycopg.errors.InsufficientPrivilege as refused:
        missing = yield Statement(UNGRANTED, (graph,))
        labels = ", ".join(str(row[0]) for row in missing.records)
        raise AgensGraphError(
            f"reading the graph around a chunk needs SELECT on every label of {graph!r}; "
            f"this role lacks it on: {labels or 'a label it cannot see'}"
        ) from refused
    return list(result.records)


class AgensGraphContextRetriever(AgensVectorRetriever):
    """Similarity seeds and their neighbourhood within ``hops`` edges.

    ``max_context_nodes`` and ``max_context_rels`` bound what each document carries, and also how
    many edges are read from any one node. ``relationship_type`` walks only edges of that type.
    This retriever reads the graph itself, and its seeds are the store's hits as they are, past
    any retrieval query the store holds; for a Cypher retrieval query of your own, use
    :class:`AgensVectorRetriever`.
    """

    hops: int = Field(default=1, ge=1, le=3)
    max_context_nodes: int = Field(default=20, ge=1)
    max_context_rels: int = Field(default=20, ge=1)
    relationship_type: str | None = None

    def _seeds(self, kwargs: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        if self.retrieval_query is not None or "retrieval_query" in kwargs:
            raise ValueError(
                "AgensGraphContextRetriever reads the graph itself; use AgensVectorRetriever "
                "with a retrieval query of your own"
            )
        if self.relationship_type is not None:
            checked(self.relationship_type, "relationship type")
        return self._arguments(kwargs)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun, **kwargs: Any
    ) -> list[Document]:
        k, search = self._seeds(kwargs)
        where = search.pop("filter", None)
        store = self.vectorstore
        # The seeds are the store's own hits, never read through a retrieval query it may hold.
        hits = store._search_hits(query, k, where, retrieval_query=None, **search)
        rows = store.engine.run(lambda: self._walk(hits), graph=store.graph_name, replay=True)
        return self._finalize(self._with_context(hits, rows))

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun, **kwargs: Any
    ) -> list[Document]:
        k, search = self._seeds(kwargs)
        where = search.pop("filter", None)
        store = self.vectorstore
        hits = await store._asearch_hits(query, k, where, retrieval_query=None, **search)
        rows = await store.engine.arun(
            lambda: self._walk(hits), graph=store.graph_name, replay=True
        )
        return self._finalize(self._with_context(hits, rows))

    def _walk(self, hits: list[tuple[Document, float, Any]]) -> Program[list[tuple[Any, ...]]]:
        return walk(
            self.vectorstore.graph_name,
            [gid for _, _, gid in hits],
            hops=self.hops,
            cap=self.max_context_rels,
            nodes=self.max_context_nodes,
            rels=self.max_context_rels,
            relationship_type=self.relationship_type,
        )

    def _with_context(
        self, hits: list[tuple[Document, float, Any]], rows: list[tuple[Any, ...]]
    ) -> list[tuple[Document, float]]:
        nodes: dict[str, dict[str, dict[str, Any]]] = {}
        rels: dict[str, dict[str, dict[str, Any]]] = {}
        for seed, node, label, properties, edge, start, end, edge_label, edge_props in rows:
            seed_nodes = nodes.setdefault(str(seed), {})
            seed_rels = rels.setdefault(str(seed), {})
            if str(node) != str(seed) and len(seed_nodes) < self.max_context_nodes:
                seed_nodes.setdefault(
                    str(node), {"id": str(node), "label": label, "properties": properties or {}}
                )
            if edge is not None and len(seed_rels) < self.max_context_rels:
                seed_rels.setdefault(
                    str(edge),
                    {
                        "type": edge_label,
                        "start": str(start),
                        "end": str(end),
                        "properties": edge_props or {},
                    },
                )
        out = []
        for document, score, gid in hits:
            seed = str(gid)
            metadata = {
                **document.metadata,
                "_seed_id_": seed,
                "_seed_label_": self.vectorstore.label,
                "_context_nodes_": list(nodes.get(seed, {}).values()),
                "_context_rels_": list(rels.get(seed, {}).values()),
            }
            out.append(
                (
                    Document(id=document.id, page_content=document.page_content, metadata=metadata),
                    score,
                )
            )
        return out


def render_graph_context(doc: Document) -> Document:
    """A ready-made ``document_formatter``: the context, written under the chunk's text.

    The structured context stays in the metadata. A neighbour is named by its ``name``, ``title``
    or ``text`` property, whichever it has first, else by its id; an edge names both ends the same
    way, and the chunk itself is named for the edges that reach it.
    """
    nodes: list[dict[str, Any]] = doc.metadata.get("_context_nodes_") or []
    rels: list[dict[str, Any]] = doc.metadata.get("_context_rels_") or []
    if not nodes and not rels:
        return doc

    def name_of(props: dict[str, Any], fallback: Any) -> str:
        for key in ("name", "title", "text"):
            value = props.get(key)
            if isinstance(value, str) and value:
                return value
        return str(fallback)

    named = {
        node["id"]: "{}:{}".format(node["label"], name_of(node.get("properties") or {}, node["id"]))
        for node in nodes
    }
    lines = [f"- {name}" for name in named.values()]
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
