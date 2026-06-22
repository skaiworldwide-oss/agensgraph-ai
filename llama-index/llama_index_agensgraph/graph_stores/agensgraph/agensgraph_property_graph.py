'''
Copyright (c) 2025, SKAI Worldwide Co., Ltd.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

from typing import (
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Pattern,
    Tuple,
)
import re, json
import logging
from contextlib import asynccontextmanager, contextmanager

from llama_index.core.graph_stores.prompts import DEFAULT_CYPHER_TEMPALTE
from llama_index.core.graph_stores.types import (
    PropertyGraphStore,
    Triplet,
    LabelledNode,
    Relation,
    EntityNode,
    ChunkNode,
)
from llama_index.core.graph_stores.utils import value_sanitize
from llama_index_agensgraph.engine import AgensEngine
from llama_index_agensgraph.filters import metadata_filters_to_cypher
from llama_index_agensgraph.graph_stores.agensgraph.utils import *
from llama_index.core.prompts import PromptTemplate
from llama_index.core.vector_stores.types import VectorStoreQuery
import psycopg
from psycopg import sql
from psycopg.types.json import Jsonb

BASE_ENTITY_LABEL = "__Entity__"
BASE_NODE_LABEL = "__Node__"
EXHAUSTIVE_SEARCH_LIMIT = 10000
# Threshold for returning all available prop values in graph schema
DISTINCT_VALUE_LIMIT = 10
CHUNK_SIZE = 1000
# Max example values kept per property in the enhanced schema.
ENHANCED_MAX_EXAMPLES = 5
VECTOR_INDEX_NAME = "entity"
LONG_TEXT_THRESHOLD = 52

# Since we do not support multiple labels, we will maintain the extra labels as a list
# This function will be used in queries to append new labels to the existing list
# and ensure that the labels are unique
append_label_function = """
    CREATE OR REPLACE FUNCTION append_label(labels jsonb, new_label text) 
    RETURNS jsonb AS $$
    BEGIN
        IF labels IS NULL OR jsonb_typeof(labels) <> 'array' THEN
            labels := '[]'::jsonb;
        END IF;

        IF NOT labels @> to_jsonb(new_label) THEN
            RETURN labels || jsonb_build_array(new_label);
        ELSE
            RETURN labels;
        END IF;
    END;
    $$ LANGUAGE plpgsql;

"""

label_catalog = """
CREATE TABLE IF NOT EXISTS label_catalog (
    graph_id oid PRIMARY KEY,
    labels jsonb DEFAULT '[]'::jsonb
);

"""

track_labels = """
CREATE OR REPLACE FUNCTION track_labels()
RETURNS TRIGGER AS $$
DECLARE
    graphid OID := {}::oid;
    new_labels JSONB;
BEGIN
    INSERT INTO label_catalog (graph_id, labels)
    VALUES (graphid, '[]'::jsonb)
    ON CONFLICT (graph_id) DO NOTHING;

    IF NEW.properties ? 'labels' THEN
        new_labels := NEW.properties->'labels';
        new_labels := (
            SELECT jsonb_agg(elems)
            FROM jsonb_array_elements_text(new_labels) AS elems
            WHERE elems NOT IN ('__Node__', '__Entity__')
        );
    ELSE
        new_labels := '[]'::jsonb;
    END IF;

    UPDATE label_catalog
    SET labels = (
        SELECT jsonb_agg(DISTINCT elems)
        FROM jsonb_array_elements(COALESCE(labels, '[]'::jsonb) || COALESCE(new_labels, '[]'::jsonb)) AS elems
    )
    WHERE graph_id = graphid;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

"""

# Collect the DISTINCT *types* per property rather than the DISTINCT *values*.
# Deduplicating values forced the server to materialize (and ship) every
# distinct value of every property on each schema refresh — including full
# embedding vectors — which scales with the graph size for no benefit, since
# only the property name and type are used (example values are sampled
# separately, with a bound, when ``enhanced_schema`` is enabled).
node_properties_query = f"""
    MATCH (a:"{BASE_NODE_LABEL}")
    UNWIND a.labels AS label
    UNWIND keys(properties(a)) AS prop
    WITH label, prop, typeof(properties(a)[prop]) AS vtype
    WHERE prop != 'labels' AND label != '{BASE_ENTITY_LABEL}'
    WITH label, prop AS property, COLLECT(DISTINCT vtype) AS types
    RETURN label, COLLECT({{'property': property, 'type': types[0]}}) as props;
"""

edge_properties_query = f"""
    MATCH ()-[e]->()
    WITH type(e) as label, properties(e) as properties
    UNWIND keys(properties) AS prop
    WITH label, prop, typeof(properties[prop]) AS vtype
    WITH label, prop AS property, COLLECT(DISTINCT vtype) AS types
    RETURN label, COLLECT(DISTINCT {{'property': property, type: types[0]}}) as props;
"""

rel_query = f"""
    MATCH (start_node)-[r]->(end_node)
    WITH DISTINCT start_node.labels AS start_labels, type(r) AS relationship_type, end_node.labels AS end_labels
    UNWIND start_labels AS start_label
    UNWIND end_labels AS end_label
    WITH DISTINCT start_label, relationship_type, end_label
    WHERE start_label != '{BASE_ENTITY_LABEL}' AND end_label != '{BASE_ENTITY_LABEL}'
    RETURN {{start: start_label, type: relationship_type, end: end_label}} AS output
"""

constraint_wrapper = """
    DO
    $$BEGIN
        {}
    EXCEPTION
        WHEN others THEN
            NULL;
    END;$$;
"""

typeof_function = r"""
    CREATE OR REPLACE FUNCTION typeof(element jsonb)
    RETURNS text AS $$
    DECLARE
        elem_type text;
    BEGIN
        elem_type := jsonb_typeof(element);
        
        IF elem_type = 'number' THEN
            IF element::text ~ '^\d+$' THEN
                RETURN 'INTEGER';
            ELSIF element::text ~ '^\d+\.\d+$' THEN
                RETURN 'FLOAT';
            ELSE               
                RETURN 'NUMBER';
            END IF;
        ELSE
            CASE UPPER(elem_type)
                WHEN 'OBJECT' THEN RETURN 'MAP';
                WHEN 'ARRAY' THEN RETURN 'LIST';
                ELSE RETURN UPPER(elem_type);
            END CASE;
        END IF;
    END;
    $$ LANGUAGE plpgsql IMMUTABLE;
"""

logger = logging.getLogger(__name__)

class AgensPropertyGraphStore(PropertyGraphStore):
    """
    AgensGraph Property Graph Store.

    This class implements a AgensGraph property graph store.
    """

    vertex_regex: Pattern = re.compile(r"(\w+)\[(\d+\.\d+)\](\{.*\})")
    edge_regex: Pattern = re.compile(r"(\w+)\[(\d+\.\d+)\]\[(\d+\.\d+),\s*(\d+\.\d+)\](\{.*\})")

    supports_structured_queries: bool = True
    supports_vector_queries: bool = True
    text_to_cypher_template: PromptTemplate = DEFAULT_CYPHER_TEMPALTE

    @require_psycopg
    def __init__(
        self,
        graph_name: str,
        conf: Dict[str, Any],
        vector_dimension: int = None,
        sanitize_query_output: bool = True,
        enhanced_schema: bool = False,
        create_indexes: bool = True,
        create: bool = True,
        refresh_schema: bool = True,
        engine: Optional[AgensEngine] = None,
    ) -> None:
        """Create a new Agensgraph Graph instance."""

        self.graph_name = graph_name
        self.sanitize_query_output = sanitize_query_output
        self.enhanced_schema = enhanced_schema
        self.create_indexes = create_indexes
        self.connection = psycopg.connect(**conf)
        self.vector_dimension = vector_dimension
        # The engine (pool) is wired in only after setup completes: graph/index
        # creation must run on the dedicated connection before the graph exists
        # (a pooled checkout would try to `SET graph_path` to a missing graph).
        self._conf = conf
        self._engine = None
        self._aconn = None

        with self._get_cursor() as curs:
            graphid = get_graph_id(curs, graph_name)
            if graphid is None:
                if create:
                    create_graph(curs, graph_name)
                    self.connection.commit()
                else:
                    raise Exception(
                        (
                            'Graph "{}" does not exist in the database '
                            + 'and "create" is set to False'
                        ).format(graph_name)
                    )
                graphid = get_graph_id(curs, graph_name)

            self.graphid = graphid
            set_graph_path(curs, graph_name)

            # Create functions, triggers and catalog to handle multiple labels
            execute_query(curs, append_label_function)
            execute_query(curs, label_catalog)
            execute_query(curs, track_labels.format(self.graphid))
            execute_query(curs, typeof_function)
            execute_query(curs, sql.SQL("CREATE VLABEL IF NOT EXISTS {};").format(
                sql.Identifier(BASE_NODE_LABEL)
            ))
            self.connection.commit()

        # Schema introspection scans every node's properties (including full
        # embedding vectors), so it is O(N) and slow on large graphs — yet it ran
        # on every construction. Make it optional: with refresh_schema=False it is
        # deferred and computed lazily on the first get_schema()/get_schema_str().
        self._schema_refreshed = False
        self.structured_schema = {
            "node_props": {}, "rel_props": {}, "relationships": {}, "metadata": {},
        }
        if refresh_schema:
            self.refresh_schema()
        self.verify_vector_support()
        if create_indexes and self._supports_vector_store and not self.vector_dimension:
            logger.warning(
                "vector_dimension was not provided; the HNSW vector index will "
                "not be created. Pass vector_dimension=<N> to enable indexed "
                "vector search (search still works without it, but unindexed)."
            )
        if create_indexes:
            self.structured_query(
                constraint_wrapper.format(
                    f"""CREATE CONSTRAINT unique_id 
                        ON "{BASE_NODE_LABEL}" 
                        ASSERT id IS UNIQUE;"""
                )
            )
            if self._supports_vector_index:
                # Create the HNSW index with a property-index expression that
                # matches what the Cypher vector query generates
                # (``n.embedding::vector(N)``), so the planner uses the index
                # (an index on ``properties->>'embedding'`` would NOT match a
                # Cypher property cast and would fall back to a seq scan).
                self.structured_query(
                    sql.SQL(
                        "CREATE PROPERTY INDEX IF NOT EXISTS {name} ON {label} "
                        "USING hnsw ((embedding::vector("
                        + str(int(self.vector_dimension))
                        + ")) vector_cosine_ops)"
                    ).format(
                        name=sql.Identifier(VECTOR_INDEX_NAME),
                        label=sql.Identifier(BASE_NODE_LABEL),
                    )
                )
                self.structured_query(
                    constraint_wrapper.format(
                        f"""CREATE CONSTRAINT embedding_length   
                        ON "{BASE_NODE_LABEL}" 
                        ASSERT jsonb_typeof(embedding) = 'array' AND 
                               jsonb_array_length(embedding) = {self.vector_dimension};"""
                    )
                )

        # Also add constraint to ensure that labels property is always a jsonb array
        self.structured_query(
            constraint_wrapper.format(
                f"""CREATE CONSTRAINT labels_array
                    ON "{BASE_NODE_LABEL}"
                    ASSERT jsonb_typeof(properties->'labels') = 'array';"""
            )
        )

        # Setup is done; runtime queries may now use the pool.
        self._engine = engine

    @require_psycopg
    def _get_cursor(self) -> psycopg.Cursor:
        cursor = self.connection.cursor(row_factory=psycopg.rows.namedtuple_row)
        return cursor

    @contextmanager
    def _acquire(self) -> "Iterator[psycopg.Connection]":
        """Yield the connection ``structured_query`` should run on.

        Uses a pooled connection from the engine when one is configured; falls
        back to the dedicated connection otherwise (the pre-engine behavior).
        """
        if self._engine is not None:
            with self._engine.connection(graph_path=self.graph_name) as conn:
                yield conn
        else:
            yield self.connection

    @asynccontextmanager
    async def _aacquire(self) -> "AsyncIterator[psycopg.AsyncConnection]":
        """Async sibling of :meth:`_acquire`."""
        if self._engine is not None:
            async with self._engine.aconnection(graph_path=self.graph_name) as conn:
                yield conn
        else:
            if self._aconn is None or self._aconn.closed:
                self._aconn = await psycopg.AsyncConnection.connect(**self._conf)
                async with self._aconn.cursor() as cur:
                    await cur.execute(
                        sql.SQL("SET graph_path = {n}").format(
                            n=sql.Identifier(self.graph_name)
                        )
                    )
                await self._aconn.commit()
            yield self._aconn

    @property
    def client(self) -> Any:
        return self.connection

    @require_psycopg
    def verify_vector_support(self) -> None:
        """
        Verify if the graph store supports vector operations
        """
        # check if the vector index is supported
        self._supports_vector_index = False
        self._supports_vector_store = False
        with self._get_cursor() as curs:
            try:
                curs.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                self.connection.commit()
                self._supports_vector_store = True
                if self.vector_dimension:
                    self._supports_vector_index = True
            except psycopg.Error:
                self.connection.rollback()
                logger.log(logging.WARNING, """Vector extension not supported\nUnable to install pg_vector extension""")
                pass

    def create_property_index(self, property_name: str) -> None:
        """
        Create a btree property index on ``property_name`` of the base node
        label.

        A metadata-filtered ``vector_query`` cannot use the HNSW index for the
        filter, so without a property index the filter degrades to a sequential
        scan over all embedded nodes. Indexing the keys you filter on lets the
        planner pre-select matching rows via an index/bitmap scan.
        """
        self.structured_query(
            sql.SQL(
                "CREATE PROPERTY INDEX IF NOT EXISTS {index_name} ON {label} ({prop})"
            ).format(
                index_name=sql.Identifier(f"{BASE_NODE_LABEL}_{property_name}_idx"),
                label=sql.Identifier(BASE_NODE_LABEL),
                prop=sql.Identifier(property_name),
            )
        )

    def refresh_schema(self) -> None:
        """
        Refresh the graph schema information by updating the available
        labels, relationships, and properties
        """

        self.structured_schema = {
            "node_props": self._get_node_properties(),
            "rel_props": self._get_edge_properties(),
            "relationships": self._get_triples(),
            "metadata": {},
        }

        # The schema-introspection helpers run SELECTs on the dedicated
        # connection without committing, which would otherwise leave it
        # idle-in-transaction holding an AccessShareLock on the label tables --
        # enough to block a second store's CREATE CONSTRAINT (AccessExclusiveLock)
        # on the same graph. Commit to release those locks.
        self.connection.commit()

        if self.enhanced_schema:
            self._enhance_schema()

        self._schema_refreshed = True

    def get_schema(self, refresh: bool = False) -> Any:
        # Lazily run the (O(N)) introspection on first access if it was deferred
        # at construction (refresh_schema=False).
        if refresh or not self._schema_refreshed:
            self.refresh_schema()

        return self.structured_schema

    def get_schema_str(
        self,
        refresh: bool = False,
        exclude_types: List[str] = [],
        include_types: List[str] = [],
    ) -> str:
        schema = self.get_schema(refresh=refresh)
        def filter_func(x: str) -> bool:
            return x in include_types if include_types else x not in exclude_types

        filtered_schema: Dict[str, Any] = {
            "node_props": {
                k: v for k, v in schema.get("node_props", {}).items() if filter_func(k)
            },
            "rel_props": {
                k: v for k, v in schema.get("rel_props", {}).items() if filter_func(k)
            },
            "relationships": [
                r
                for r in schema.get("relationships", [])
                if all(filter_func(r[t]) for t in ["start", "end", "type"])
            ],
        }

        formatted_node_props = []
        formatted_rel_props = []
        # Format node properties
        for label, props in filtered_schema["node_props"].items():
            prop_strs = []
            for prop in props:
                prop_str = f"{prop['property']}: {prop['type']}"
                # Statistics are only present when ``enhanced_schema`` is on.
                if "min" in prop and "max" in prop:
                    prop_str += f" (min: {prop['min']}, max: {prop['max']})"
                elif "min_size" in prop and "max_size" in prop:
                    prop_str += (
                        f" (list size min: {prop['min_size']}, max: {prop['max_size']})"
                    )
                elif prop.get("values"):
                    examples = ", ".join(str(v) for v in prop["values"])
                    prop_str += f" (e.g. {examples})"
                prop_strs.append(prop_str)
            props_str = ", ".join(prop_strs)
            formatted_node_props.append(f"{label} {{{props_str}}}")

        # Format relationship properties using structured_schema
        for type, props in filtered_schema["rel_props"].items():
            props_str = ", ".join(
                [f"{prop['property']}: {prop['type']}" for prop in props]
            )
            formatted_rel_props.append(f"{type} {{{props_str}}}")

        # Format relationships
        formatted_rels = [
            f"(:{el['start']})-[:{el['type']}]->(:{el['end']})"
            for el in filtered_schema["relationships"]
        ]

        return "\n".join(
            [
                "Node properties:",
                "\n".join(formatted_node_props),
                "Relationship properties:",
                "\n".join(formatted_rel_props),
                "The relationships:",
                "\n".join(formatted_rels),
            ]
        )

    def _build_upsert_nodes_ops(
        self, nodes: List[LabelledNode]
    ) -> List[Tuple[sql.Composed, Dict[str, Any]]]:
        """Build the ordered (query, params) operations for ``upsert_nodes``."""
        # Lists to hold separated types
        entity_dicts: List[dict] = []
        chunk_dicts: List[dict] = []

        # Sort by type
        for item in nodes:
            if isinstance(item, EntityNode):
                entity_dicts.append({**item.model_dump(), "id": item.id})
            elif isinstance(item, ChunkNode):
                chunk_dicts.append({**item.model_dump(), "id": item.id})
            else:
                # Log that we do not support these types of nodes
                # Or raise an error?
                pass

        ops: List[Tuple[sql.Composed, Dict[str, Any]]] = []

        if chunk_dicts:
            for index in range(0, len(chunk_dicts), CHUNK_SIZE):
                chunked_params = chunk_dicts[index : index + CHUNK_SIZE]
                ops.append((
                    sql.SQL("""
                    UNWIND %(chunked_params)s AS row
                    MERGE (c:{BASE_NODE_LABEL} {{id: row.id}})
                    SET c.text = row.text, c.labels = append_label(c.labels, 'Chunk')
                    WITH c, row
                    SET c += row.properties, c.embedding = row.embedding
                    RETURN count(*)
                    """).format(
                        BASE_NODE_LABEL=sql.Identifier(BASE_NODE_LABEL)
                    ), {"chunked_params": Jsonb(chunked_params)}
                ))

        if entity_dicts:
            for index in range(0, len(entity_dicts), CHUNK_SIZE):
                chunked_params = entity_dicts[index : index + CHUNK_SIZE]
                ops.append((
                    sql.SQL("""
                    UNWIND %(chunked_params)s AS row
                    MERGE (e:{BASE_NODE_LABEL} {{id: row.id}})
                    SET e += CASE WHEN row.properties IS NOT NULL THEN row.properties ELSE properties(e) END
                    SET e.name = CASE WHEN row.name IS NOT NULL THEN row.name ELSE e.name END,
                        e.labels = append_label(e.labels, {BASE_ENTITY_LABEL})
                    WITH e, row
                    SET e.labels = append_label(e.labels, row.label)
                    """).format(
                        BASE_NODE_LABEL=sql.Identifier(BASE_NODE_LABEL),
                        BASE_ENTITY_LABEL=sql.Literal(BASE_ENTITY_LABEL)
                    ), {"chunked_params": Jsonb(chunked_params)}
                ))
                # Write embeddings for every entity that carries one. This is a
                # SEPARATE statement from the MENTIONS link below on purpose:
                # AgensGraph does not persist an earlier SET when a subsequent
                # `WITH ... WHERE` filters out every row ahead of a MERGE, so folding
                # the embedding write into the triplet_source_id branch silently
                # dropped embeddings for any entity that has no source chunk (e.g. a
                # structured / non-LLM-extracted graph).
                ops.append((
                    sql.SQL("""
                    UNWIND %(chunked_params)s AS row
                    MATCH (e:{BASE_NODE_LABEL} {{id: row.id}})
                    WHERE row.embedding IS NOT NULL
                    SET e.embedding = row.embedding
                    """).format(
                        BASE_NODE_LABEL=sql.Identifier(BASE_NODE_LABEL)
                    ), {"chunked_params": Jsonb(chunked_params)}
                ))
                # Link each entity to its source chunk via MENTIONS, for the rows
                # that carry a triplet_source_id.
                ops.append((
                    sql.SQL("""
                    UNWIND %(chunked_params)s AS row
                    WITH row WHERE row.properties.triplet_source_id IS NOT NULL
                    MATCH (e:{BASE_NODE_LABEL} {{id: row.id}})
                    MERGE (c:{BASE_NODE_LABEL} {{id: row.properties.triplet_source_id}})
                    MERGE (e)<-[:"MENTIONS"]-(c)
                    """).format(
                        BASE_NODE_LABEL=sql.Identifier(BASE_NODE_LABEL)
                    ), {"chunked_params": Jsonb(chunked_params)}
                ))

        return ops

    def upsert_nodes(self, nodes: List[LabelledNode]) -> None:
        for query, params in self._build_upsert_nodes_ops(nodes):
            self.structured_query(query, params)

    async def aupsert_nodes(self, nodes: List[LabelledNode]) -> None:
        """True-async counterpart of :meth:`upsert_nodes`."""
        for query, params in self._build_upsert_nodes_ops(nodes):
            await self.astructured_query(query, params)

    def _build_upsert_relations_ops(
        self, relations: List[Relation]
    ) -> List[Tuple[sql.Composed, Dict[str, Any]]]:
        """Build the ordered (query, params) operations for ``upsert_relations``.

        Relations are grouped by label and each group is UNWIND-batched in
        CHUNK_SIZE rows (the relationship type must be a literal in MERGE, so a
        batch can only span one label). This replaces the previous
        one-query-per-relation behavior; the batched ``MERGE (n {id: row.id})``
        is still index-backed.
        """
        by_label: Dict[str, List[dict]] = {}
        for r in relations:
            d = r.model_dump()
            by_label.setdefault(d["label"], []).append(d)

        ops: List[Tuple[sql.Composed, Dict[str, Any]]] = []
        for label, rels in by_label.items():
            for index in range(0, len(rels), CHUNK_SIZE):
                chunk = rels[index : index + CHUNK_SIZE]
                rows = [
                    {
                        "source_id": p["source_id"],
                        "target_id": p["target_id"],
                        "properties": p["properties"],
                    }
                    for p in chunk
                ]
                ops.append((
                    sql.SQL("""
                    UNWIND %(rows)s AS row
                    MERGE (source: {BASE_NODE_LABEL} {{id: row.source_id}})
                    ON CREATE SET source.labels = append_label(source.labels, 'Chunk')
                    MERGE (target: {BASE_NODE_LABEL} {{id: row.target_id}})
                    ON CREATE SET target.labels = append_label(target.labels, 'Chunk')
                    WITH source, target, row
                    MERGE (source)-[r:{label}]->(target)
                    SET r += row.properties
                    RETURN count(*)
                    """).format(
                        BASE_NODE_LABEL=sql.Identifier(BASE_NODE_LABEL),
                        label=sql.Identifier(label),
                    ), {"rows": Jsonb(rows)}
                ))
        return ops

    def upsert_relations(self, relations: List[Relation]) -> None:
        """Add relations."""
        for query, params in self._build_upsert_relations_ops(relations):
            self.structured_query(query, params)

    async def aupsert_relations(self, relations: List[Relation]) -> None:
        """True-async counterpart of :meth:`upsert_relations`."""
        for query, params in self._build_upsert_relations_ops(relations):
            await self.astructured_query(query, params)

    @staticmethod
    def _or_equalities(
        field: str, values: List[str], prefix: str
    ) -> Tuple[str, Dict[str, Any]]:
        """Build an index-friendly ``(field = v0 OR field = v1 ...)`` fragment.

        The ``<@`` / ``IN`` containment forms never use the btree index; an
        OR-of-equalities is served via a BitmapOr index scan when ``field`` is
        indexed (``id`` always is; other properties when explicitly indexed).
        """
        terms = []
        params: Dict[str, Any] = {}
        for i, value in enumerate(values):
            pname = f"{prefix}_{i}"
            params[pname] = Jsonb(value)
            terms.append(f"{field} = %({pname})s")
        return "(" + " OR ".join(terms) + ")", params

    def _build_get(
        self,
        properties: Optional[dict] = None,
        ids: Optional[List[str]] = None,
    ) -> Tuple[sql.Composed, Dict[str, Any]]:
        """Build the (query, params) for :meth:`get`."""
        query = """SELECT t.name,
                            t.type,
                            (t.properties - 'labels') || '{{"embedding": null, "id": null}}'::jsonb AS properties
                     FROM ("""
        params: Dict[str, Any] = {}
        query += 'MATCH (e:{BASE_NODE_LABEL}) '
        query += "WHERE e.id IS NOT NULL "
        if ids is not None and len(ids) == 0:
            # An explicit empty id list means "no nodes". Without this guard the
            # `if ids:` below would skip the filter entirely and the query would
            # match the ENTIRE graph — e.g. get_llama_nodes([]) (no source docs)
            # fetched all 130k+ nodes, making include_text retrieval take ~29s.
            query += "AND false "
        elif ids:
            frag, id_params = self._or_equalities("e.id", ids, "get_id")
            query += "AND " + frag + " "
            params.update(id_params)

        if properties:
            prop_params = [f'e."{prop}" = %({prop})s' for prop in properties]
            query += "AND " + " AND ".join(prop_params)
            params.update(
                {f"{prop}": Jsonb(properties[prop]) for prop in properties}
            )

        query += """
            WITH e, e.labels as labels
            RETURN
            e.id AS name,
            CASE
                WHEN {BASE_ENTITY_LABEL} IN labels THEN
                    CASE
                        WHEN length(labels) > 2 THEN labels[2]
                        WHEN length(labels) > 1 THEN labels[1]
                        ELSE NULL
                    END
                ELSE labels[0]
            END AS type,
            properties(e) AS properties
        """
        query += ")t"
        return (
            sql.SQL(query).format(
                BASE_NODE_LABEL=sql.Identifier(BASE_NODE_LABEL),
                BASE_ENTITY_LABEL=sql.Literal(BASE_ENTITY_LABEL)
            ),
            params,
        )

    @staticmethod
    def _get_response_to_nodes(
        response: Optional[List[Dict[str, Any]]]
    ) -> List[LabelledNode]:
        response = response if response else []
        nodes: List[LabelledNode] = []
        for record in response:
            if "text" in record["properties"] or record["type"] is None:
                text = record["properties"].pop("text", "")
                nodes.append(
                    ChunkNode(
                        id_=record["name"],
                        text=text,
                        properties=record["properties"],
                    )
                )
            else:
                nodes.append(
                    EntityNode(
                        name=record["name"],
                        label=record["type"],
                        properties=record["properties"],
                    )
                )

        return nodes

    def get(
        self,
        properties: Optional[dict] = None,
        ids: Optional[List[str]] = None,
    ) -> List[LabelledNode]:
        """Get nodes."""
        query, params = self._build_get(properties, ids)
        response = self.structured_query(query, params=params)
        return self._get_response_to_nodes(response)

    async def aget(
        self,
        properties: Optional[dict] = None,
        ids: Optional[List[str]] = None,
    ) -> List[LabelledNode]:
        """True-async counterpart of :meth:`get`."""
        query, params = self._build_get(properties, ids)
        response = await self.astructured_query(query, params=params)
        return self._get_response_to_nodes(response)

    def get_triplets(
        self,
        entity_names: Optional[List[str]] = None,
        relation_names: Optional[List[str]] = None,
        properties: Optional[dict] = None,
        ids: Optional[List[str]] = None,
    ) -> List[Triplet]:
        params = {}

        query = """
                SELECT t.type,
                        t.rel_prop,
                        t.source_id,
                        t.source_type,
                        (t.source_properties - 'labels') || '{{"embedding": null, "name": null}}'::jsonb AS source_properties,
                        t.target_id,
                        t.target_type,
                        (t.target_properties - 'labels') || '{{"embedding": null, "name": null}}'::jsonb AS target_properties
                FROM ("""
        query += "MATCH (e)-[r]->(t) "
        query += "WHERE {BASE_ENTITY_LABEL} IN e.labels "

        if entity_names or relation_names or properties or ids:
            query += "AND "

        if entity_names:
            frag, p = self._or_equalities("e.name", entity_names, "etn")
            query += frag + " "
            params.update(p)

        if relation_names and entity_names:
            query += "AND "

        if relation_names:
            # type(r) is the edge label, not a property, so it can't use a
            # property index; the containment form is fine here.
            query += "type(r) <@ %(relation_names)s "
            params["relation_names"] = Jsonb(relation_names)

        if ids:
            frag, p = self._or_equalities("e.id", ids, "gtid")
            query += frag + " "
            params.update(p)

        if properties:
            prop_params = [f'e."{prop}" = %({prop})s' for prop in properties]
            query += "AND " + " AND ".join(prop_params)
            params.update(
                {f"{prop}": Jsonb(properties[prop]) for prop in properties}
            )

        query += """
        AND NOT ANY(label IN e.labels WHERE label = 'Chunk')
            WITH *, e.labels as e_labels, t.labels as t_labels
            RETURN type(r) as type, properties(r) as rel_prop, e.id as source_id,
            CASE
                WHEN {BASE_ENTITY_LABEL} IN e_labels THEN
                    CASE
                        WHEN length(e_labels) > 2 THEN e_labels[2]
                        WHEN length(e_labels) > 1 THEN e_labels[1]
                        ELSE NULL
                    END
                ELSE e_labels[0]
            END AS source_type,
            properties(e) AS source_properties,
            t.id as target_id,
            CASE
                WHEN {BASE_ENTITY_LABEL} IN t_labels THEN
                    CASE
                        WHEN length(t_labels) > 2 THEN t_labels[2]
                        WHEN length(t_labels) > 1 THEN t_labels[1]
                        ELSE NULL
                    END
                ELSE t_labels[0]
            END AS target_type, properties(t) AS target_properties LIMIT 100
        """

        query += ")t"
        data = self.structured_query(
            sql.SQL(query).format(
                BASE_ENTITY_LABEL=sql.Literal(BASE_ENTITY_LABEL),
                BASE_NODE_LABEL=sql.Identifier(BASE_NODE_LABEL)
            ), params=params
        )
        data = data if data else []

        triplets = []
        for record in data:
            source = EntityNode(
                name=record["source_id"],
                label=record["source_type"],
                properties=record["source_properties"],
            )
            target = EntityNode(
                name=record["target_id"],
                label=record["target_type"],
                properties=record["target_properties"],
            )
            rel = Relation(
                source_id=record["source_id"],
                target_id=record["target_id"],
                label=record["type"],
                properties=record["rel_prop"],
            )
            triplets.append([source, rel, target])
        return triplets

    def get_rel_map(
        self,
        graph_nodes: List[LabelledNode],
        depth: int = 2,
        limit: int = 30,
        ignore_rels: Optional[List[str]] = None,
    ) -> List[Triplet]:
        """Get depth-aware rel map."""
        triples = []

        ids = [node.id for node in graph_nodes]
        if not ids:
            return triples
        query = """SELECT t.source_id,
                            t.source_type,
                            (t.source_properties - 'labels') || '{{"embedding": null, "id": null}}'::jsonb AS source_properties,
                            t.type,
                            t.rel_properties,
                            t.target_id,
                            t.target_type,
                            (t.target_properties - 'labels') || '{{"embedding": null, "id": null}}'::jsonb AS target_properties
                      FROM (
                """
        # OR-of-equalities seed match uses the id index (BitmapOr), whereas the
        # previous `UNWIND idx ... WHERE e.id = ids[idx]` dynamic subscript
        # forced a sequential scan of the whole node set per call.
        seed_frag, seed_params = self._or_equalities("e.id", ids, "rmid")
        # AgensGraph's variable-length-edge engine is pathologically slow even at
        # depth 1 -- a plain 1-hop match returns the same rows ~1000x faster
        # (~12 ms vs ~17 s for depth-1 over a few seeds on a 50k-node graph). Use a
        # fixed pattern for the common depth<=1 case; only fall back to the
        # *1..depth path for genuine multi-hop maps.
        if depth <= 1:
            traversal = (
                """
            MATCH (e:{BASE_NODE_LABEL})
            WHERE """ + seed_frag + """
            MATCH (e)-[rel]-(other)
            WHERE type(rel) <> 'MENTIONS'
                """
            )
        else:
            traversal = (
                """
            MATCH (e:{BASE_NODE_LABEL})
            WHERE """ + seed_frag + """
            MATCH p=(e)-[r*1..{depth}]-(other)
            UNWIND relationships(p) AS rel
            WITH DISTINCT rel, collect(type(rel)) AS types
            WHERE all(x IN types WHERE x <> 'MENTIONS')
                """
            )
        query += traversal + """
            WITH startNode(rel) AS source,
                type(rel) AS type,
                rel AS rel_properties,
                endNode(rel) AS endNode,
                startNode(rel).labels AS source_labels,
                endNode(rel).labels AS target_labels
            LIMIT %(limit)s
            RETURN source.id AS source_id,
                CASE
                    WHEN {BASE_ENTITY_LABEL} IN source_labels THEN
                        CASE
                            WHEN length(source_labels) > 2 THEN source_labels[2]
                            WHEN length(source_labels) > 1 THEN source_labels[1]
                            ELSE NULL
                        END
                    ELSE source_labels[0]
                END AS source_type,
                properties(source) AS source_properties,
                type,
                properties(rel_properties) as rel_properties,
                endNode.id AS target_id,
                CASE
                    WHEN {BASE_ENTITY_LABEL} IN target_labels THEN
                        CASE
                            WHEN length(target_labels) > 2 THEN target_labels[2]
                            WHEN length(target_labels) > 1 THEN target_labels[1] ELSE NULL
                        END
                    ELSE target_labels[0]
                END AS target_type,
                properties(endNode) AS target_properties
            LIMIT %(limit)s
            """
        query += ")t"
        response = self.structured_query(
            sql.SQL(query).format(
                BASE_NODE_LABEL=sql.Identifier(BASE_NODE_LABEL),
                BASE_ENTITY_LABEL=sql.Literal(BASE_ENTITY_LABEL),
                depth=depth
            ), {**seed_params, "limit": limit}
        )
        response = response if response else []

        ignore_rels = ignore_rels or []
        for record in response:
            if record["type"] in ignore_rels:
                continue

            source = EntityNode(
                name=record["source_id"],
                label=record["source_type"],
                properties=record["source_properties"],
            )
            target = EntityNode(
                name=record["target_id"],
                label=record["target_type"],
                properties=record["target_properties"],
            )
            rel = Relation(
                source_id=record["source_id"],
                target_id=record["target_id"],
                label=record["type"],
                properties=record["rel_properties"],
            )
            triples.append([source, rel, target])

        return triples
    
    def delete(
        self,
        entity_names: Optional[List[str]] = None,
        relation_names: Optional[List[str]] = None,
        properties: Optional[dict] = None,
        ids: Optional[List[str]] = None,
    ) -> None:
        """Delete matching data."""
        if entity_names:
            frag, p = self._or_equalities("n.name", entity_names, "etn")
            self.structured_query(
                'MATCH (n:"__Node__") WHERE ' + frag + " DETACH DELETE n", p
            )

        if ids:
            frag, p = self._or_equalities("n.id", ids, "delid")
            self.structured_query(
                'MATCH (n:"__Node__") WHERE ' + frag + " DETACH DELETE n", p
            )

        if relation_names:
            for rel in relation_names:
                self.structured_query(
                    sql.SQL(
                        'MATCH ()-[r:{rel}]->() DELETE r'
                    ).format(
                        rel=sql.Identifier(rel)
                    )
                )

        if properties:
            cypher = "MATCH (e) WHERE "
            props = [f'e."{prop}" = %({prop})s' for prop in properties]
            cypher += " AND ".join(props)
            cypher += " DETACH DELETE e"
            params = {f"{prop}": Jsonb(properties[prop]) for prop in properties}
            self.structured_query(cypher, params=params)

    def _build_vector_query(
        self, query: VectorStoreQuery
    ) -> Optional[Tuple[sql.Composed, Dict[str, Any]]]:
        """Build the (query, params) for :meth:`vector_query`, or None if vector
        operations are unsupported."""
        # Translate metadata filters into a parameterized WHERE fragment so the
        # ANN search can be scoped (mirrors the filtered-vector-search feature of
        # other graph integrations, but injection-safe).
        filter_clause: sql.Composed = sql.SQL("")
        filter_params: Dict[str, Any] = {}
        if query.filters:
            snippet, filter_params = metadata_filters_to_cypher(
                query.filters, alias="n"
            )
            filter_clause = sql.SQL("AND (") + snippet + sql.SQL(")")

        if self._supports_vector_index:
            # The nearest-neighbour ORDER BY + LIMIT live INSIDE the Cypher
            # sub-query against the actual query embedding, so AgensGraph can
            # use the HNSW index; the outer SQL only reshapes the properties.
            # Dimension is the store's configured dimension (a pgvector typmod
            # must be a literal, so it is interpolated, not bound).
            vector_query = (
                """
                SELECT
                    t.name,
                    t.type,
                    (t.properties - 'labels') || '{{"embedding": null, "name": null, "id": null}}'::jsonb AS properties,
                    t.similarity
                FROM (
                    MATCH (n: {BASE_NODE_LABEL})
                    WHERE n.embedding IS NOT NULL {filter_clause}
                    WITH n, n.labels AS labels,
                         (n.embedding::vector({dim}) <=> %(query_embedding)s::vector({dim})) AS dist
                    ORDER BY dist
                    LIMIT %(top_k)s
                    RETURN n.id as name,
                           properties(n) AS properties,
                           (1 - dist) AS similarity,
                           CASE
                                WHEN {BASE_ENTITY_LABEL} IN labels THEN
                                    CASE
                                        WHEN length(labels) > 2 THEN labels[2]
                                        WHEN length(labels) > 1 THEN labels[1]
                                        ELSE NULL
                                    END
                                ELSE labels[0]
                           END AS type
                )t;
                """
            )
            return (
                sql.SQL(vector_query).format(
                    BASE_NODE_LABEL=sql.Identifier(BASE_NODE_LABEL),
                    BASE_ENTITY_LABEL=sql.Literal(BASE_ENTITY_LABEL),
                    dim=sql.SQL(str(int(self.vector_dimension))),
                    filter_clause=filter_clause,
                ),
                {
                    "query_embedding": Jsonb(query.query_embedding),
                    "top_k": query.similarity_top_k,
                    **filter_params,
                },
            )
        elif self._supports_vector_store:
            vector_query = """SELECT t.name,
                                t.type,
                                t.similarity,
                                (t.properties - 'labels') || '{{"embedding": null, "name": null, "id": null}}'::jsonb AS properties
                            FROM (
                            """
            vector_query += """
                            MATCH (n: {BASE_NODE_LABEL})
                            WHERE n.embedding IS NOT NULL {filter_clause}
                            WITH n,
                                n.labels AS labels,
                                %(query_embedding)s::vector <=> n.embedding::vector AS cos_d
                            ORDER BY cos_d
                            LIMIT %(top_k)s
                            RETURN n.id as name,
                                properties(n) AS properties,
                                1-cos_d as similarity,
                                CASE
                                    WHEN {BASE_ENTITY_LABEL} IN labels THEN
                                        CASE
                                            WHEN length(labels) > 2 THEN labels[2]
                                            WHEN length(labels) > 1 THEN labels[1]
                                            ELSE NULL
                                        END
                                    ELSE labels[0]
                                END AS type
                            """
            vector_query += ")t"
            return (
                sql.SQL(vector_query).format(
                    BASE_NODE_LABEL=sql.Identifier(BASE_NODE_LABEL),
                    BASE_ENTITY_LABEL=sql.Literal(BASE_ENTITY_LABEL),
                    filter_clause=filter_clause,
                ),
                {
                    "query_embedding": Jsonb(query.query_embedding),
                    "top_k": query.similarity_top_k,
                    **filter_params,
                },
            )
        else:
            return None

    @staticmethod
    def _vector_data_to_result(
        data: Optional[List[Dict[str, Any]]]
    ) -> Tuple[List[LabelledNode], List[float]]:
        data = data if data else []
        nodes: List[LabelledNode] = []
        scores: List[float] = []
        for record in data:
            node = EntityNode(
                name=record["name"],
                label=record["type"],
                properties=record["properties"],
            )
            nodes.append(node)
            scores.append(record["similarity"])

        return (nodes, scores)

    def vector_query(
        self, query: VectorStoreQuery, **kwargs: Any
    ) -> Tuple[List[LabelledNode], List[float]]:
        """Query the graph store with a vector store query."""
        built = self._build_vector_query(query)
        data = self.structured_query(*built) if built is not None else []
        return self._vector_data_to_result(data)

    async def avector_query(
        self, query: VectorStoreQuery, **kwargs: Any
    ) -> Tuple[List[LabelledNode], List[float]]:
        """True-async counterpart of :meth:`vector_query`."""
        built = self._build_vector_query(query)
        data = await self.astructured_query(*built) if built is not None else []
        return self._vector_data_to_result(data)

    @staticmethod
    def _record_to_dict(record: NamedTuple) -> Dict[str, Any]:
        """
        Convert a record returned from an agensgraph query to a dictionary

        Args:
            record (): a record from an agensgraph query result

        Returns:
            Dict[str, Any]: a dictionary representation of the record where
                the dictionary key is the field name and the value is the
                value converted to a python type
        """
        # result holder
        d = {}

        # prebuild a mapping of vertex_id to vertex mappings to be used
        # later to build edges
        vertices = {}
        for k in record._fields:
            v = getattr(record, k)

            # records comes back label[id]{properties} which must be parsed
            if isinstance(v, str):
                vertex = AgensPropertyGraphStore.vertex_regex.match(v)
                if vertex:
                    label, vertex_id, properties = vertex.groups()
                    properties = json.loads(properties)
                    vertices[str(vertex_id)] = properties

        # iterate returned fields and parse appropriately
        for k in record._fields:
            v = getattr(record, k)

            if isinstance(v, str):
                vertex = AgensPropertyGraphStore.vertex_regex.match(v)
                edge = AgensPropertyGraphStore.edge_regex.match(v)

                if vertex:
                    d[k] = json.loads(vertex.group(3))
                elif edge:
                    elabel, edge_id, start_id, end_id, properties = edge.groups()
                    d[k] = (
                        vertices.get(start_id, {}),
                        elabel,
                        vertices.get(end_id, {}),
                    )
                else:
                    d[k] = v

            else:
                d[k] = v

        return d

    @require_psycopg
    def structured_query(self, query: str, params: dict = {}) -> List[Dict[str, Any]]:
        """
        Query the graph by taking a cypher query, executing it and
        converting the result

        Args:
            query (str): a cypher query to be executed
            params (dict): parameters for the query (not used in this implementation)

        Returns:
            List[Dict[str, Any]]: a list of dictionaries containing the result set
        """

        # execute the query, rolling back on an error
        with self._acquire() as conn:
            with conn.cursor(row_factory=psycopg.rows.namedtuple_row) as curs:
                try:
                    curs.execute(query, params)
                    conn.commit()
                except psycopg.Error as e:
                    conn.rollback()
                    raise AgensQueryException(
                        {
                            "message": "Error executing graph query: {}".format(query),
                            "detail": str(e),
                        }
                    )
                try:
                    data = curs.fetchall()
                except psycopg.ProgrammingError:
                    data = []  # Handle queries that don’t return data

                if data is None:
                    result = []
                # convert to dictionaries
                else:
                    result = [self._record_to_dict(d) for d in data]

                if self.sanitize_query_output:
                    result = [value_sanitize(el) for el in result]

                return result

    async def astructured_query(
        self, query: str, params: dict = {}
    ) -> List[Dict[str, Any]]:
        """Async counterpart of :meth:`structured_query` (true async I/O)."""
        async with self._aacquire() as conn:
            async with conn.cursor(row_factory=psycopg.rows.namedtuple_row) as curs:
                try:
                    await curs.execute(query, params)
                    await conn.commit()
                except psycopg.Error as e:
                    await conn.rollback()
                    raise AgensQueryException(
                        {
                            "message": "Error executing graph query: {}".format(query),
                            "detail": str(e),
                        }
                    )
                try:
                    data = await curs.fetchall()
                except psycopg.ProgrammingError:
                    data = []

                if data is None:
                    result = []
                else:
                    result = [self._record_to_dict(d) for d in data]

                if self.sanitize_query_output:
                    result = [value_sanitize(el) for el in result]

                return result

    @require_psycopg
    def _get_node_properties(self) -> Dict[str, Any]:
        node_properties = {}
        with self._get_cursor() as curs:
            execute_query(curs, node_properties_query)
            rows = curs.fetchall()

            for row in rows:                
                node_properties[row.label] = row.props

        return node_properties

    @require_psycopg
    def _get_edge_properties(self) -> Dict[str, Any]:
        edge_properties = {}
        with self._get_cursor() as curs:
            execute_query(curs, edge_properties_query)
            rows = curs.fetchall()

            for row in rows:                
                edge_properties[row.label] = row.props

        return edge_properties

    def _get_triples(self) -> List[Dict[str, str]]:
        """
        Get a set of distinct relationship types (as a list of dicts) in the graph
        to be used as context by an llm.

        Returns:
            List[Dict[str, str]]: relationships as a list of dicts in the format
                "{'start':<from_label>, 'type':<edge_label>, 'end':<from_label>}"
        """

        triple_schema = []
        with self._get_cursor() as curs:
            execute_query(curs, rel_query)
            rows = curs.fetchall()
            triple_schema = [row.output for row in rows]
        
        return triple_schema

    def _get_triples_str(self) -> List[str]:
        """
        Get a set of distinct relationship types (as a list of strings) in the graph
        to be used as context by an llm.

        Returns:
            List[str]: relationships as a list of strings in the format
                "(:"<from_label>")-[:"<edge_label>"]->(:"<to_label>")"
        """

        triples = self._get_triples()
        return format_triples(triples)

    # Property type groups for enhanced-schema statistics.
    _NUMERIC_TYPES = {"INTEGER", "FLOAT", "NUMBER"}

    def _label_count(self, label: str) -> int:
        rows = self.structured_query(
            sql.SQL(
                "MATCH (a:{base_label}) WHERE %(label)s IN a.labels RETURN count(a) AS c"
            ).format(base_label=sql.Identifier(BASE_NODE_LABEL)),
            {"label": Jsonb(label)},
        )
        return int(rows[0]["c"]) if rows else 0

    def _enhance_schema(self) -> None:
        """
        Enrich each node-label property in ``structured_schema`` with concrete
        statistics for text-to-Cypher prompting:

        - numeric properties get ``min`` / ``max`` / ``distinct_count``,
        - list properties get ``min_size`` / ``max_size``,
        - everything else (strings, etc.) gets example ``values`` +
          ``distinct_count``.

        Stats are computed exhaustively when the label has at most
        ``EXHAUSTIVE_SEARCH_LIMIT`` nodes, otherwise over a bounded sample of
        that many nodes. The embedding/labels properties are always skipped, so
        this never materializes vectors or grows unbounded with the graph.
        """
        node_props = self.structured_schema.get("node_props", {})
        for label, props in node_props.items():
            if label == BASE_ENTITY_LABEL:
                continue
            try:
                count = self._label_count(label)
            except Exception as exc:  # pragma: no cover - best-effort enrichment
                logger.warning("Enhanced schema count failed for %s: %s", label, exc)
                continue
            if count == 0:
                continue
            # None => exhaustive (no LIMIT); else cap the scan to a sample.
            sample = None if count <= EXHAUSTIVE_SEARCH_LIMIT else EXHAUSTIVE_SEARCH_LIMIT

            for prop in props:
                name = prop["property"]
                if name in ("embedding", "labels"):
                    continue
                try:
                    if prop.get("type") in self._NUMERIC_TYPES:
                        prop.update(self._numeric_stats(label, name, sample))
                    elif prop.get("type") == "LIST":
                        prop.update(self._list_stats(label, name, sample))
                    else:
                        prop.update(self._value_stats(label, name, sample))
                except Exception as exc:  # pragma: no cover - best-effort enrichment
                    logger.warning(
                        "Enhanced schema sampling failed for %s.%s: %s",
                        label,
                        name,
                        exc,
                    )

    def _stat_subquery(
        self, prop: str, sample: Optional[int]
    ) -> Tuple[sql.Composed, sql.Composed]:
        """Build the shared Cypher subquery returning a property's values and a
        LIMIT clause (empty when exhaustive)."""
        limit = (
            sql.SQL("LIMIT {n}").format(n=sql.SQL(str(int(sample))))
            if sample is not None
            else sql.SQL("")
        )
        subquery = sql.SQL(
            "MATCH (a:{base_label}) WHERE %(label)s IN a.labels AND a.{prop} IS NOT NULL "
            "RETURN a.{prop} AS v {limit}"
        ).format(
            base_label=sql.Identifier(BASE_NODE_LABEL),
            prop=sql.Identifier(prop),
            limit=limit,
        )
        return subquery, limit

    def _numeric_stats(
        self, label: str, prop: str, sample: Optional[int]
    ) -> Dict[str, Any]:
        subquery, _ = self._stat_subquery(prop, sample)
        rows = self.structured_query(
            sql.SQL(
                "SELECT min((t.v #>> '{{}}')::numeric) AS min, "
                "max((t.v #>> '{{}}')::numeric) AS max, "
                "count(DISTINCT t.v) AS distinct_count FROM ({sub})t"
            ).format(sub=subquery),
            {"label": Jsonb(label)},
        )
        if not rows or rows[0]["min"] is None:
            return {}
        row = rows[0]
        return {
            "min": float(row["min"]),
            "max": float(row["max"]),
            "distinct_count": row["distinct_count"],
        }

    def _list_stats(
        self, label: str, prop: str, sample: Optional[int]
    ) -> Dict[str, Any]:
        subquery, _ = self._stat_subquery(prop, sample)
        rows = self.structured_query(
            sql.SQL(
                "SELECT min(jsonb_array_length(t.v)) AS min_size, "
                "max(jsonb_array_length(t.v)) AS max_size FROM ({sub})t"
            ).format(sub=subquery),
            {"label": Jsonb(label)},
        )
        if not rows or rows[0]["min_size"] is None:
            return {}
        return {"min_size": rows[0]["min_size"], "max_size": rows[0]["max_size"]}

    def _value_stats(
        self, label: str, prop: str, sample: Optional[int]
    ) -> Dict[str, Any]:
        subquery, _ = self._stat_subquery(prop, sample)
        rows = self.structured_query(
            sql.SQL(
                "SELECT (array_agg(DISTINCT t.v))[1:{max_examples}] AS examples, "
                "count(DISTINCT t.v) AS distinct_count FROM ({sub})t"
            ).format(
                max_examples=sql.SQL(str(ENHANCED_MAX_EXAMPLES)),
                sub=subquery,
            ),
            {"label": Jsonb(label)},
        )
        if not rows or not rows[0].get("examples"):
            return {}
        return {
            "values": rows[0]["examples"],
            "distinct_count": rows[0]["distinct_count"],
        }
