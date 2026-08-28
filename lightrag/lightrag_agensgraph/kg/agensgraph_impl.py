# Copyright (c) 2025, SKAI Worldwide Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Graph storage: LightRAG's entities and relations as a property graph.

Every entity is a ``base`` vertex found by its ``entity_id``, every relation a
``DIRECTED`` edge. LightRAG treats a relation as undirected, so an edge is
stored once per pair with its endpoints in sorted order, and a unique index on
the pair keeps two writers from making two.

Writes are Cypher with bound parameters. Reads that touch edges are SQL over
the label tables, anchored by the graph ids the ``entity_id`` index resolves:
a Cypher pattern between two named nodes walks every edge of the first one and
tests the other end, where a probe of the edge table by the pair of ids is one
index read whatever the node's degree.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, final

from agensgraph import DesiredLabel, GraphId, Jsonb, Unique
from lightrag.base import BaseGraphStorage
from lightrag.constants import GRAPH_FIELD_SEP
from lightrag.types import KnowledgeGraph, KnowledgeGraphEdge, KnowledgeGraphNode
from lightrag.utils import logger
from psycopg import sql

from lightrag_agensgraph.kg._base import (
    DEFAULT_GRAPH,
    _AgensStorageBase,
    graph_name_for,
    resolve_workspace,
)

NODE = "base"
EDGE = "DIRECTED"
PAIR_INDEX = "DIRECTED_pair_key"
CHUNK_SIZE = 1000

# ---- statements ----
# SQL templates take the graph as {g}; a Cypher map's braces are doubled for format().
# ANCHORS resolves a bound list of names to graph ids through the entity_id index.
ANCHORS = (
    "v AS (SELECT name #>> '{{}}' AS name, gid FROM "
    "(UNWIND %(names)s AS name MATCH (n:base {{entity_id: name}}) RETURN name, id(n) AS gid) t)"
)
EDGES = '{g}."DIRECTED"'
NODES = "{g}.base"

HAS_NODE = "MATCH (n:base {entity_id: %(name)s}) RETURN true AS found LIMIT 1"
HAS_NODES = "UNWIND %(names)s AS name MATCH (n:base {entity_id: name}) RETURN n.entity_id AS name"
GET_NODE = "MATCH (n:base {entity_id: %(name)s}) RETURN properties(n) AS props"
GET_NODES = "UNWIND %(names)s AS name MATCH (n:base {entity_id: name}) RETURN name, properties(n) AS props"
UPSERT_NODE = "MERGE (n:base {entity_id: %(name)s}) SET n += %(props)s"
UPSERT_NODES = "UNWIND %(rows)s AS row MERGE (n:base {entity_id: row.id}) SET n += row.props"
UPSERT_EDGE = (
    "MERGE (a:base {entity_id: %(a)s}) MERGE (b:base {entity_id: %(b)s}) "
    'MERGE (a)-[r:"DIRECTED"]->(b) SET r += %(props)s'
)
UPSERT_EDGES = (
    "UNWIND %(rows)s AS row MERGE (a:base {entity_id: row.a}) MERGE (b:base {entity_id: row.b}) "
    'MERGE (a)-[r:"DIRECTED"]->(b) SET r += row.props'
)
DELETE_NODES = "UNWIND %(names)s AS name MATCH (n:base {entity_id: name}) DETACH DELETE n"
DROP = "MATCH (n) DETACH DELETE n"

DEGREES = (
    f"WITH {ANCHORS} SELECT v.name, "
    f"(SELECT count(*) FROM {EDGES} e WHERE e.start = v.gid) + "
    f'(SELECT count(*) FROM {EDGES} e WHERE e."end" = v.gid) AS degree FROM v'
)
# A single name is a scalar anchor: one index probe, whatever the server assumed when it
# planned the statement. A list of names goes through ANCHORS with sequential scans off.
PAIR = (
    "(SELECT gid FROM (MATCH (n:base {{entity_id: %(a)s}}) RETURN id(n) AS gid) t) a, "
    "(SELECT gid FROM (MATCH (n:base {{entity_id: %(b)s}}) RETURN id(n) AS gid) t) b"
)
SAME_PAIR = '((e.start = a.gid AND e."end" = b.gid) OR (e.start = b.gid AND e."end" = a.gid))'
HAS_EDGE = f"SELECT EXISTS (SELECT 1 FROM {EDGES} e, {PAIR} WHERE {SAME_PAIR}) AS found"
GET_EDGE = f"SELECT e.properties FROM {EDGES} e, {PAIR} WHERE {SAME_PAIR} LIMIT 1"
GET_EDGES = (
    f"WITH {ANCHORS}, p AS (SELECT * FROM unnest(%(src)s::text[], %(tgt)s::text[]) AS p(src, tgt)) "
    f"SELECT p.src, p.tgt, e.properties FROM p JOIN v a ON a.name = p.src JOIN v b ON b.name = p.tgt "
    f'JOIN {EDGES} e ON e.start = a.gid AND e."end" = b.gid '
    "UNION ALL "
    f"SELECT p.src, p.tgt, e.properties FROM p JOIN v a ON a.name = p.src JOIN v b ON b.name = p.tgt "
    f'JOIN {EDGES} e ON e.start = b.gid AND e."end" = a.gid'
)
# The neighbour's name is read by a correlated probe of the node table's primary key: as a
# join the planner merged the whole node table against a few hundred edges.
NODE_EDGES = (
    f"WITH {ANCHORS}, half AS ("
    f'SELECT v.name, e."end" AS other FROM v JOIN {EDGES} e ON e.start = v.gid '
    f'UNION ALL SELECT v.name, e.start FROM v JOIN {EDGES} e ON e."end" = v.gid) '
    f"SELECT v.name, (SELECT b.properties->>'entity_id' FROM {NODES} b WHERE b.id = half.other) AS other "
    "FROM v LEFT JOIN half ON half.name = v.name"
)
# A write to a label table is Cypher only, and a Cypher pattern between two nodes named by
# a property walks one node's edges however it is spelled. Named by graph id, the pattern
# is one probe of the pair index: the ids are resolved first, in SQL, then the edges go.
PAIR_IDS = (
    f"WITH {ANCHORS}, p AS (SELECT * FROM unnest(%(src)s::text[], %(tgt)s::text[]) AS p(src, tgt)) "
    "SELECT a.gid, b.gid FROM p JOIN v a ON a.name = p.src JOIN v b ON b.name = p.tgt"
)
DELETE_EDGES = (
    'UNWIND %(rows)s AS row MATCH (a)-[r:"DIRECTED"]->(b) '
    "WHERE id(a) = row.a::graphid AND id(b) = row.b::graphid DELETE r"
)
# Names sort in "C" collation: byte order, which is what LightRAG's own file-backed store
# sorts them in, and a fraction of the cost of a locale sort.
ALL_LABELS = (
    f"SELECT properties->>'entity_id' FROM {NODES} WHERE properties ? 'entity_id' "
    "ORDER BY (properties->>'entity_id') COLLATE \"C\""
)
ALL_NODES = f"SELECT properties FROM {NODES}"
ALL_EDGES = (
    f"SELECT s.properties->>'entity_id', t.properties->>'entity_id', e.properties FROM {EDGES} e "
    f'JOIN {NODES} s ON s.id = e.start JOIN {NODES} t ON t.id = e."end"'
)
DEGREE_TABLE = (
    "(SELECT id, count(*) AS degree FROM "
    f'(SELECT start AS id FROM {EDGES} UNION ALL SELECT "end" FROM {EDGES}) x GROUP BY id) d'
)
POPULAR_CONNECTED = (
    f"SELECT b.properties->>'entity_id' AS label FROM {DEGREE_TABLE} JOIN {NODES} b ON b.id = d.id "
    "WHERE b.properties ? 'entity_id' "
    "ORDER BY d.degree DESC, (b.properties->>'entity_id') COLLATE \"C\" ASC LIMIT %(lim)s"
)
POPULAR_ISOLATED = (
    f"SELECT properties->>'entity_id' AS label FROM {NODES} b WHERE b.properties ? 'entity_id' "
    f'AND NOT EXISTS (SELECT 1 FROM {EDGES} e WHERE e.start = b.id OR e."end" = b.id) '
    "ORDER BY (properties->>'entity_id') COLLATE \"C\" ASC LIMIT %(lim)s"
)
# Every label is read once: the name and its lower-case form come out of one pass over the
# nodes (OFFSET 0 keeps the subquery from being flattened into per-reference extraction),
# then matched by substring and ranked: exact, prefix, shorter first.
SEARCH_LABELS = (
    "SELECT label FROM "
    f"(SELECT properties->>'entity_id' AS label, lower(properties->>'entity_id') AS low "
    f"FROM {NODES} OFFSET 0) x "
    "WHERE strpos(low, %(q)s) > 0 "
    "ORDER BY CASE WHEN low = %(q)s THEN 1000 WHEN starts_with(low, %(q)s) THEN 500 "
    'ELSE 100 - length(label) END DESC, label COLLATE "C" ASC LIMIT %(lim)s'
)
COUNT_NODES = f"SELECT count(*) FROM {NODES} WHERE properties ? 'entity_id'"
TOP_NODES = (
    f"SELECT b.id, b.properties FROM {NODES} b LEFT JOIN {DEGREE_TABLE} ON d.id = b.id "
    "WHERE b.properties ? 'entity_id' "
    "ORDER BY COALESCE(d.degree, 0) DESC, b.properties->>'entity_id' ASC LIMIT %(lim)s"
)
SEED_NODE = "SELECT gid FROM (MATCH (n:base {{entity_id: %(name)s}}) RETURN id(n) AS gid) t"
EDGE_ROW = f'SELECT e.id, e.start, e."end", e.properties FROM {EDGES} e'
EDGES_AMONG = f'{EDGE_ROW} WHERE e.start = ANY(%(ids)s) AND e."end" = ANY(%(ids)s)'
EDGES_TOUCHING = f'{EDGE_ROW} WHERE e.start = ANY(%(ids)s) OR e."end" = ANY(%(ids)s)'
NODES_BY_ID = f"SELECT id, properties FROM {NODES} WHERE id = ANY(%(ids)s)"
NODES_BY_CHUNKS = (
    f"SELECT properties FROM {NODES} WHERE properties ? 'source_id' "
    "AND string_to_array(properties->>'source_id', %(sep)s) && %(ids)s::text[]"
)
EDGES_BY_CHUNKS = (
    f"SELECT s.properties->>'entity_id', t.properties->>'entity_id', e.properties FROM {EDGES} e "
    f'JOIN {NODES} s ON s.id = e.start JOIN {NODES} t ON t.id = e."end" '
    "WHERE e.properties ? 'source_id' "
    "AND string_to_array(e.properties->>'source_id', %(sep)s) && %(ids)s::text[]"
)


@final
@dataclass
class AgensgraphStorage(_AgensStorageBase, BaseGraphStorage):
    def __post_init__(self):
        # LightRAG names its graph namespace `chunk_entity_relation`; a workspace gets
        # a graph of its own, so two tenants' knowledge graphs never mix.
        self.workspace = resolve_workspace(self.workspace, self.global_config)
        self._engine = None

    def _graph_name(self) -> str:
        base = os.environ.get("AGENSGRAPH_GRAPHNAME") or self.namespace or DEFAULT_GRAPH
        return graph_name_for(base, self.workspace or "")

    @property
    def graph_name(self) -> str:
        return self._graph_name()

    def _sql(self, template: str) -> sql.Composed:
        return sql.SQL(template).format(g=sql.Identifier(self.graph_name))

    async def initialize(self):
        """Take the shared engine and declare the graph's labels and keys once."""
        await self._acquire_engine()

        async def declare(conn):
            # A node is found by its entity_id: the constraint gives MERGE and every lookup a
            # unique index and turns two writers racing on one name into an error the retry
            # handles. An edge is one per pair; that key is over the endpoint columns, which
            # a property index cannot express, so it is plain SQL.
            await conn.ensure_labels([DesiredLabel(NODE, "v"), DesiredLabel(EDGE, "e")])
            await conn.ensure_constraints([Unique(NODE, "entity_id")])
            await conn.execute(
                sql.SQL('CREATE UNIQUE INDEX IF NOT EXISTS {} ON {} (start, "end")').format(
                    sql.Identifier(PAIR_INDEX), sql.Identifier(self.graph_name, EDGE)
                )
            )

        await self._engine.setup_once(f"graph:{self.graph_name}", declare)
        logger.info("AgensGraph storage initialized for graph: %s", self.graph_name)

    async def finalize(self):
        await self._release_engine()

    async def index_done_callback(self) -> None:
        pass  # every write is already committed

    # ---- nodes ----

    async def has_node(self, node_id: str) -> bool:
        return bool(await self._fetch(HAS_NODE, {"name": node_id}))

    async def has_nodes_batch(self, node_ids: list[str]) -> set[str]:
        found: Set[str] = set()
        for start in range(0, len(node_ids), CHUNK_SIZE):
            rows = await self._fetch(HAS_NODES, {"names": Jsonb(node_ids[start : start + CHUNK_SIZE])})
            found.update(r["name"] for r in rows)
        return found

    async def get_node(self, node_id: str) -> dict[str, str] | None:
        rows = await self._fetch(GET_NODE, {"name": node_id})
        return rows[0]["props"] if rows else None

    async def get_nodes_batch(self, node_ids: list[str]) -> dict[str, dict]:
        nodes: Dict[str, dict] = {}
        for start in range(0, len(node_ids), CHUNK_SIZE):
            rows = await self._fetch(GET_NODES, {"names": Jsonb(node_ids[start : start + CHUNK_SIZE])})
            nodes.update((r["name"], r["props"]) for r in rows)
        return nodes

    async def node_degree(self, node_id: str) -> int:
        return (await self.node_degrees_batch([node_id]))[node_id]

    async def node_degrees_batch(self, node_ids: list[str]) -> dict[str, int]:
        degrees = {name: 0 for name in node_ids}
        for start in range(0, len(node_ids), CHUNK_SIZE):
            rows = await self._fetch_tuples(
                self._sql(DEGREES), {"names": Jsonb(node_ids[start : start + CHUNK_SIZE])}, by_index=True
            )
            degrees.update((name, int(degree)) for name, degree in rows)
        return degrees

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        degrees = await self.node_degrees_batch(list({src_id, tgt_id}))
        return degrees[src_id] + degrees[tgt_id]

    async def edge_degrees_batch(self, edge_pairs: list[tuple[str, str]]) -> dict[tuple[str, str], int]:
        names = list({name for pair in edge_pairs for name in pair})
        degrees = await self.node_degrees_batch(names)
        return {(src, tgt): degrees[src] + degrees[tgt] for src, tgt in edge_pairs}

    async def upsert_node(self, node_id: str, node_data: dict[str, str]) -> None:
        props = {**node_data, "entity_id": node_id}
        await self._fetch(UPSERT_NODE, {"name": node_id, "props": Jsonb(props)}, wrote=True)

    async def upsert_nodes_batch(self, nodes: list[tuple[str, dict[str, str]]]) -> None:
        # One row per id; the last data given for an id is what is written.
        rows = list(
            {
                node_id: {"id": node_id, "props": {**data, "entity_id": node_id}} for node_id, data in nodes
            }.values()
        )
        for start in range(0, len(rows), CHUNK_SIZE):
            await self._fetch(UPSERT_NODES, {"rows": Jsonb(rows[start : start + CHUNK_SIZE])}, wrote=True)

    async def delete_node(self, node_id: str) -> None:
        await self.remove_nodes([node_id])

    async def remove_nodes(self, nodes: list[str]):
        for start in range(0, len(nodes), CHUNK_SIZE):
            await self._fetch(DELETE_NODES, {"names": Jsonb(nodes[start : start + CHUNK_SIZE])}, wrote=True)

    # ---- edges ----

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        rows = await self._fetch_tuples(
            self._sql(HAS_EDGE),
            {"a": source_node_id, "b": target_node_id},
        )
        return bool(rows and rows[0][0])

    async def get_edge(self, source_node_id: str, target_node_id: str) -> dict[str, str] | None:
        rows = await self._fetch(
            self._sql(GET_EDGE),
            {"a": source_node_id, "b": target_node_id},
        )
        return rows[0]["properties"] if rows else None

    async def get_edges_batch(self, pairs: list[dict[str, str]]) -> dict[tuple[str, str], dict]:
        """The properties of each requested pair's edge, keyed by the pair as requested."""
        edges: Dict[Tuple[str, str], dict] = {}
        for start in range(0, len(pairs), CHUNK_SIZE):
            chunk = pairs[start : start + CHUNK_SIZE]
            src = [p["src"] for p in chunk]
            tgt = [p["tgt"] for p in chunk]
            rows = await self._fetch(
                self._sql(GET_EDGES),
                {"names": Jsonb(list({*src, *tgt})), "src": src, "tgt": tgt},
                by_index=True,
            )
            for r in rows:
                edges.setdefault((r["src"], r["tgt"]), r["properties"])
        return edges

    async def get_node_edges(self, source_node_id: str) -> list[tuple[str, str]] | None:
        """The node's edges as (node, neighbour) pairs; [] for a node with none, None for no node."""
        rows = await self._fetch_tuples(
            self._sql(NODE_EDGES), {"names": Jsonb([source_node_id])}, by_index=True
        )
        if not rows:
            return None
        return [(name, other) for name, other in rows if other is not None]

    async def get_nodes_edges_batch(self, node_ids: list[str]) -> dict[str, list[tuple[str, str]]]:
        edges: Dict[str, List[Tuple[str, str]]] = {name: [] for name in node_ids}
        for start in range(0, len(node_ids), CHUNK_SIZE):
            rows = await self._fetch_tuples(
                self._sql(NODE_EDGES), {"names": Jsonb(node_ids[start : start + CHUNK_SIZE])}, by_index=True
            )
            for name, other in rows:
                if other is not None:
                    edges[name].append((name, other))
        return edges

    async def upsert_edge(self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]) -> None:
        # A relation is undirected to LightRAG, so the pair is stored one way round; an
        # endpoint that is not there yet is made, rather than the edge silently not written.
        a, b = sorted((source_node_id, target_node_id))
        await self._fetch(UPSERT_EDGE, {"a": a, "b": b, "props": Jsonb(edge_data)}, wrote=True)

    async def upsert_edges_batch(self, edges: list[tuple[str, str, dict[str, str]]]) -> None:
        rows_by_pair: Dict[Tuple[str, str], dict] = {}
        for src, tgt, data in edges:
            a, b = sorted((src, tgt))
            rows_by_pair[(a, b)] = {"a": a, "b": b, "props": data}
        rows = list(rows_by_pair.values())
        for start in range(0, len(rows), CHUNK_SIZE):
            await self._fetch(UPSERT_EDGES, {"rows": Jsonb(rows[start : start + CHUNK_SIZE])}, wrote=True)

    async def remove_edges(self, edges: list[tuple[str, str]]):
        for start in range(0, len(edges), CHUNK_SIZE):
            chunk = [tuple(sorted(pair)) for pair in edges[start : start + CHUNK_SIZE]]
            src = [a for a, _ in chunk]
            tgt = [b for _, b in chunk]
            found = await self._fetch_tuples(
                self._sql(PAIR_IDS),
                {"names": Jsonb(list({*src, *tgt})), "src": src, "tgt": tgt},
                by_index=True,
            )
            if found:
                rows = [{"a": str(a), "b": str(b)} for a, b in found]
                await self._fetch(DELETE_EDGES, {"rows": Jsonb(rows)}, wrote=True)

    # ---- the whole graph ----

    async def get_all_labels(self) -> list[str]:
        return [r[0] for r in await self._fetch_tuples(self._sql(ALL_LABELS))]

    async def get_all_nodes(self) -> list[dict]:
        return [r[0] for r in await self._fetch_tuples(self._sql(ALL_NODES))]

    async def get_all_edges(self) -> list[dict]:
        return [
            {**props, "source": source, "target": target}
            for source, target, props in await self._fetch_tuples(self._sql(ALL_EDGES))
        ]

    async def get_popular_labels(self, limit: int = 300) -> list[str]:
        """Entity names by degree, most connected first; isolated entities last."""
        labels = [r[0] for r in await self._fetch_tuples(self._sql(POPULAR_CONNECTED), {"lim": limit})]
        if len(labels) < limit:
            rows = await self._fetch_tuples(self._sql(POPULAR_ISOLATED), {"lim": limit - len(labels)})
            labels.extend(r[0] for r in rows)
        return labels

    async def search_labels(self, query: str, limit: int = 50) -> list[str]:
        """Entity names containing the text, exact matches first, then prefixes, then shorter names."""
        query = (query or "").strip().lower()
        if not query:
            return []
        rows = await self._fetch_tuples(self._sql(SEARCH_LABELS), {"q": query, "lim": limit})
        return [r[0] for r in rows]

    async def drop(self) -> dict[str, str]:
        try:
            await self._fetch(DROP, wrote=True)
            logger.info("Successfully dropped all data from graph %s", self.graph_name)
            return {"status": "success", "message": "graph data dropped"}
        except Exception as e:
            logger.error("Error dropping graph %s: %s", self.graph_name, e)
            return {"status": "error", "message": str(e)}

    # ---- subgraphs ----

    async def get_knowledge_graph(
        self, node_label: str, max_depth: int = 3, max_nodes: int = 1000
    ) -> KnowledgeGraph:
        """A connected subgraph: the densest nodes for ``*``, else a breadth-first walk from a node.

        Nodes carry the entity name as id and first label. The walk is one statement per
        level over the edge table by graph id, and stops adding nodes at ``max_nodes``.
        """
        result = KnowledgeGraph()
        nodes: Dict[GraphId, dict] = {}
        edges: Dict[GraphId, tuple] = {}

        if node_label == "*":
            total = (await self._fetch_tuples(self._sql(COUNT_NODES)))[0][0]
            for gid, props in await self._fetch_tuples(self._sql(TOP_NODES), {"lim": max_nodes}):
                nodes[gid] = props
            result.is_truncated = int(total) > max_nodes
            if nodes:
                for eid, start, end, props in await self._fetch_tuples(
                    self._sql(EDGES_AMONG), {"ids": list(nodes)}
                ):
                    edges[eid] = (start, end, props)
        else:
            seed = await self._fetch_tuples(self._sql(SEED_NODE), {"name": node_label})
            if seed:
                for gid, props in await self._fetch_tuples(self._sql(NODES_BY_ID), {"ids": [seed[0][0]]}):
                    nodes[gid] = props
            frontier = list(nodes)
            depth = 0
            while frontier and depth < max_depth and len(nodes) < max_nodes:
                touching = await self._fetch_tuples(self._sql(EDGES_TOUCHING), {"ids": frontier})
                new_ids = {x for _, start, end, _ in touching for x in (start, end) if x not in nodes}
                if new_ids:
                    room = max_nodes - len(nodes)
                    if len(new_ids) > room:
                        result.is_truncated = True
                    fetched = await self._fetch_tuples(self._sql(NODES_BY_ID), {"ids": list(new_ids)})
                    fetched.sort(key=lambda row: row[1].get("entity_id") or "")
                    for gid, props in fetched[:room]:
                        nodes[gid] = props
                for eid, start, end, props in touching:
                    if start in nodes and end in nodes:
                        edges[eid] = (start, end, props)
                frontier = [gid for gid, _ in fetched[:room]] if new_ids else []
                depth += 1

        names = {gid: props.get("entity_id") for gid, props in nodes.items()}
        result.nodes = [
            KnowledgeGraphNode(id=str(name), labels=[str(name)], properties=props)
            for gid, props in nodes.items()
            if (name := names[gid]) is not None
        ]
        result.edges = [
            KnowledgeGraphEdge(
                id=str(eid),
                type=EDGE,
                source=str(names[start]),
                target=str(names[end]),
                properties=props or {},
            )
            for eid, (start, end, props) in edges.items()
            if names.get(start) is not None and names.get(end) is not None
        ]
        return result

    async def get_nodes_by_chunk_ids(self, chunk_ids: list[str]) -> list[dict]:
        rows = await self._fetch_tuples(
            self._sql(NODES_BY_CHUNKS), {"sep": GRAPH_FIELD_SEP, "ids": list(chunk_ids)}
        )
        return [{**props, "id": props.get("entity_id")} for (props,) in rows]

    async def get_edges_by_chunk_ids(self, chunk_ids: list[str]) -> list[dict]:
        rows = await self._fetch_tuples(
            self._sql(EDGES_BY_CHUNKS), {"sep": GRAPH_FIELD_SEP, "ids": list(chunk_ids)}
        )
        return [{**props, "source": source, "target": target} for source, target, props in rows]
