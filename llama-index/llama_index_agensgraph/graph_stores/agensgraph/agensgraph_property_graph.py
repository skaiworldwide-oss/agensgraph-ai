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
# Bounds for enhanced-schema example sampling (keeps it independent of graph size)
ENHANCED_SAMPLE_SIZE = 1000
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

    def get_schema(self, refresh: bool = False) -> Any:
        if refresh:
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
                # Example values are only present when ``enhanced_schema`` is on.
                if prop.get("values"):
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
                ops.append((
                    sql.SQL("""
                    UNWIND %(chunked_params)s AS row
                    MATCH (e:{BASE_NODE_LABEL} {{id: row.id}})
                    WHERE row.embedding IS NOT NULL
                    SET e.embedding = row.embedding
                    WITH e, row
                    WHERE row.properties.triplet_source_id IS NOT NULL
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
        """Build the ordered (query, params) operations for ``upsert_relations``."""
        params = [r.model_dump() for r in relations]
        ops: List[Tuple[sql.Composed, Dict[str, Any]]] = []
        for index in range(0, len(params), CHUNK_SIZE):
            chunked_params = params[index : index + CHUNK_SIZE]
            for param in chunked_params:
                ops.append((
                    sql.SQL("""
                    MERGE (source: {BASE_NODE_LABEL} {{id: %(source_id)s}})
                    ON CREATE SET source.labels = append_label(source.labels, 'Chunk')
                    MERGE (target: {BASE_NODE_LABEL} {{id: %(target_id)s}})
                    ON CREATE SET target.labels = append_label(target.labels, 'Chunk')
                    WITH source, target
                    MERGE (source)-[r:{label}]->(target)
                    SET r += %(properties)s
                    RETURN count(*)
                    """).format(
                        BASE_NODE_LABEL=sql.Identifier(BASE_NODE_LABEL),
                        label=sql.Identifier(param["label"])
                    ), {
                        "source_id": Jsonb(param["source_id"]),
                        "target_id": Jsonb(param["target_id"]),
                        "properties": Jsonb(param["properties"])
                    }
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
        if ids:
            query += "AND e.id <@ %(ids)s "
            params["ids"] = Jsonb(ids)

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
            query += "e.name <@ %(entity_names)s "
            params["entity_names"] = Jsonb(entity_names)

        if relation_names and entity_names:
            query += "AND "

        if relation_names:
            query += "type(r) <@ %(relation_names)s "
            params["relation_names"] = Jsonb(relation_names)

        if ids:
            query += "e.id <@ %(ids)s "
            params["ids"] = Jsonb(ids)

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
        query += f"""
            UNWIND {[0] if len(ids) == 1 else f'range(0, {len(ids)} - 1)::jsonb'} AS idx
            """
        query += """
            MATCH (e:{BASE_NODE_LABEL})
            WHERE e.id = %(ids)s[idx]
            MATCH p=(e)-[r*1..{depth}]-(other)
            UNWIND relationships(p) AS rel
            WITH DISTINCT rel, idx, collect(type(rel)) AS types
            WHERE all(x IN types WHERE x <> 'MENTIONS')
            WITH startNode(rel) AS source,
                type(rel) AS type,
                rel AS rel_properties,
                endNode(rel) AS endNode,
                idx,
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
                properties(endNode) AS target_properties,
                idx
            ORDER BY idx
            LIMIT %(limit)s
            """
        query += ")t"
        response = self.structured_query(
            sql.SQL(query).format(
                BASE_NODE_LABEL=sql.Identifier(BASE_NODE_LABEL),
                BASE_ENTITY_LABEL=sql.Literal(BASE_ENTITY_LABEL),
                depth=depth
            ), {"ids": Jsonb(ids), "limit": limit}
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
            self.structured_query(
                "MATCH (n) WHERE n.name <@ %(entity_names)s DETACH DELETE n",
                {"entity_names": Jsonb(entity_names)}
            )

        if ids:
            self.structured_query(
                "MATCH (n) WHERE n.id <@ %(ids)s DETACH DELETE n",
                {"ids": Jsonb(ids)}
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
                    WHERE n.embedding IS NOT NULL
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
                ),
                {
                    "query_embedding": Jsonb(query.query_embedding),
                    "top_k": query.similarity_top_k,
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
                    BASE_ENTITY_LABEL=sql.Literal(BASE_ENTITY_LABEL)
                ),
                {
                    "query_embedding": Jsonb(query.query_embedding),
                    "top_k": query.similarity_top_k,
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

    def _enhance_schema(self) -> None:
        """
        Attach a small sample of example values to each node-label property in
        ``structured_schema`` (an enhanced schema), so that a
        text-to-Cypher prompt can show concrete examples.

        The sampling is bounded — at most ``ENHANCED_SAMPLE_SIZE`` nodes are
        scanned per label and at most ``ENHANCED_MAX_EXAMPLES`` distinct values
        are kept per property — and the embedding property is skipped, so this
        never materializes vectors or grows with the graph.
        """
        node_props = self.structured_schema.get("node_props", {})
        for label, props in node_props.items():
            if label == BASE_ENTITY_LABEL:
                continue
            try:
                rows = self.structured_query(
                    sql.SQL(
                        """
                        MATCH (a:{base_label})
                        WHERE %(label)s IN a.labels
                        WITH a LIMIT {sample_size}
                        UNWIND keys(properties(a)) AS prop
                        WITH prop, properties(a)[prop] AS value
                        WHERE prop <> 'labels' AND prop <> 'embedding'
                        WITH prop AS property, COLLECT(DISTINCT value) AS examples
                        RETURN property, examples[0..{max_examples}] AS examples;
                        """
                    ).format(
                        base_label=sql.Identifier(BASE_NODE_LABEL),
                        sample_size=sql.SQL(str(ENHANCED_SAMPLE_SIZE)),
                        max_examples=sql.SQL(str(ENHANCED_MAX_EXAMPLES)),
                    ),
                    {"label": Jsonb(label)},
                )
            except Exception as exc:  # pragma: no cover - best-effort enrichment
                logger.warning("Enhanced schema sampling failed for %s: %s", label, exc)
                continue

            examples_by_prop = {r["property"]: r.get("examples") or [] for r in rows}
            for prop in props:
                examples = examples_by_prop.get(prop["property"])
                if examples:
                    prop["values"] = examples
