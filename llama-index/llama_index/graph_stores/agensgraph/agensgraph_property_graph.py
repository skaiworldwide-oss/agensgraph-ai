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

from typing import Any, List, Dict, Optional, Tuple, NamedTuple, Pattern
import re, json
import logging

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
from llama_index.graph_stores.agensgraph.utils import *
from llama_index.core.prompts import PromptTemplate
from llama_index.core.vector_stores.types import VectorStoreQuery
import psycopg2
import psycopg2.extras


def remove_empty_values(input_dict):
    """
    Remove entries with empty values from the dictionary.

    Parameters:
    input_dict (dict): The dictionary from which empty values need to be removed.

    Returns:
    dict: A new dictionary with all empty values removed.
    """
    # Create a new dictionary excluding empty values
    return {key: value for key, value in input_dict.items() if value}

def remove_nones(input_dict):
    """
    Remove entries with None values from the dictionary.

    Parameters:
    input_dict (dict): The dictionary from which None values need to be removed.

    Returns:
    dict: A new dictionary with all None values removed.
    """
    return {key: value for key, value in input_dict.items() if value is not None}


BASE_ENTITY_LABEL = "__Entity__"
BASE_NODE_LABEL = "__Node__"
EXHAUSTIVE_SEARCH_LIMIT = 10000
# Threshold for returning all available prop values in graph schema
DISTINCT_VALUE_LIMIT = 10
CHUNK_SIZE = 1000
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

node_properties_query = f"""
    MATCH (a:"{BASE_NODE_LABEL}")
    UNWIND a.labels AS label
    UNWIND keys(properties(a)) AS prop
    WITH label, prop, properties(a)[prop] AS value 
    WHERE prop != 'labels'
    WITH                
        label,
        prop AS property,
        COLLECT(DISTINCT value) AS values,
        COUNT(DISTINCT value) AS distinct_count
    WHERE label != '{BASE_ENTITY_LABEL}' 
    RETURN label, COLLECT({{'property': property, 'values':values, 'distinct_count': distinct_count, 'type': typeof(values[0])}}) as props;
"""

edge_properties_query = f"""
    MATCH ()-[e]->()
    WITH type(e) as label, properties(e) as properties
    UNWIND keys(properties) AS prop
    WITH label, prop, properties[prop] AS value 
    WITH                
        label,
        prop AS property,
        COLLECT(DISTINCT value) AS values
    RETURN label, COLLECT(DISTINCT {{'property': property, type: typeof(values[0])}}) as props;
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

typeof_function = f"""
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

    @require_psycopg2
    def __init__(
        self,
        graph_name: str,
        conf: Dict[str, Any],
        vector_dimension: int = None,
        sanitize_query_output: bool = True,
        enhanced_schema: bool = False,
        create_indexes: bool = True,
        create: bool = True
    ) -> None:
        """Create a new Agensgraph Graph instance."""

        self.graph_name = graph_name
        self.sanitize_query_output = sanitize_query_output
        self.enhanced_schema = enhanced_schema
        self.create_indexes = create_indexes
        self.connection = psycopg2.connect(**conf)
        self.vector_dimension = vector_dimension

        with self._get_cursor() as curs:
            graph_id_query = (
                """SELECT oid as graphid FROM ag_graph WHERE graphname = '{}';""".format(
                    graph_name
                )
            )
            execute_query(curs, graph_id_query)
            data = curs.fetchone()

            if data is None:
                if create:
                    create_statement = """
                        CREATE GRAPH {};
                    """.format(graph_name)
                    execute_query(curs, create_statement, "Error creating graph")
                else:
                    raise Exception(
                        (
                            'Graph "{}" does not exist in the database '
                            + 'and "create" is set to False'
                        ).format(graph_name)
                    )

                curs.execute(graph_id_query)
                data = curs.fetchone()

            self.graphid = data.graphid

            graph_path = """SET graph_path = '{}';""".format(self.graph_name)
            execute_query(curs, graph_path)

            # Create functions, triggers and catalog to handle multiple labels
            execute_query(curs, append_label_function)
            execute_query(curs, label_catalog)
            execute_query(curs, track_labels.format(self.graphid))
            execute_query(curs, typeof_function)
            execute_query(curs, f'CREATE VLABEL IF NOT EXISTS "{BASE_NODE_LABEL}"')
            self.connection.commit()

        self.refresh_schema()
        self.verify_vector_support()
        if create_indexes:
            self.structured_query(
                constraint_wrapper.format(
                    f"""CREATE CONSTRAINT unique_id 
                        ON "{BASE_NODE_LABEL}" 
                        ASSERT id IS UNIQUE;"""
                )
            )
            if self._supports_vector_index:
                self.structured_query(
                        f"""CREATE INDEX IF NOT EXISTS {VECTOR_INDEX_NAME}
                        ON {self.graph_name}."{BASE_NODE_LABEL}" USING hnsw 
                        (((properties->>'embedding')::vector({self.vector_dimension})) vector_cosine_ops)"""
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

    @require_psycopg2
    def _get_cursor(self) -> psycopg2.extras.NamedTupleCursor:
        cursor = self.connection.cursor(cursor_factory=psycopg2.extras.NamedTupleCursor)
        return cursor

    @property
    def client(self) -> Any:
        return self.connection

    @require_psycopg2
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
            except psycopg2.Error:
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
            props_str = ", ".join(
                [f"{prop['property']}: {prop['type']}" for prop in props]
            )
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

    def upsert_nodes(self, nodes: List[LabelledNode]) -> None:
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

        if chunk_dicts:
            for index in range(0, len(chunk_dicts), CHUNK_SIZE):
                chunked_params = chunk_dicts[index : index + CHUNK_SIZE]
                chunked_params = [remove_nones(row) for row in chunked_params]
                self.structured_query(
                    """
                    UNWIND {} AS row
                    MERGE (c:"{BASE_NODE_LABEL}" {{id: row.id}})
                    SET c.text = row.text, c.labels = append_label(c.labels, 'Chunk')
                    WITH c, row
                    SET c += row.properties, c.embedding = row.embedding
                    RETURN count(*)
                    """.format(chunked_params, BASE_NODE_LABEL=BASE_NODE_LABEL),
                )

        if entity_dicts:
            for index in range(0, len(entity_dicts), CHUNK_SIZE):
                chunked_params = entity_dicts[index : index + CHUNK_SIZE]
                chunked_params = [remove_nones(row) for row in chunked_params]
                self.structured_query(
                    """
                    UNWIND {chunked_params} AS row
                    MERGE (e:"{BASE_NODE_LABEL}" {{id: row.id}})
                    SET e += CASE WHEN row.properties IS NOT NULL THEN row.properties ELSE properties(e) END
                    SET e.name = CASE WHEN row.name IS NOT NULL THEN row.name ELSE e.name END,
                        e.labels = append_label(e.labels, '{BASE_ENTITY_LABEL}')
                    WITH e, row
                    SET e.labels = append_label(e.labels, row.label);

                    UNWIND {chunked_params} AS row
                    MATCH (e:"{BASE_NODE_LABEL}" {{id: row.id}})
                    WHERE row.embedding IS NOT NULL
                    SET e.embedding = row.embedding
                    WITH e, row
                    WHERE row.properties.triplet_source_id IS NOT NULL
                    MERGE (c:"{BASE_NODE_LABEL}" {{id: row.properties.triplet_source_id}})
                    MERGE (e)<-[:"MENTIONS"]-(c)
                    """.format(chunked_params=chunked_params,
                               BASE_NODE_LABEL=BASE_NODE_LABEL, 
                               BASE_ENTITY_LABEL=BASE_ENTITY_LABEL),
                )

    def upsert_relations(self, relations: List[Relation]) -> None:
        """Add relations."""
        params = [r.model_dump() for r in relations]
        for index in range(0, len(params), CHUNK_SIZE):
            chunked_params = params[index : index + CHUNK_SIZE]
            for param in chunked_params:
                formatted_properties = ", ".join(
                    [f"{key}: {value!r}" for key, value in param["properties"].items()]
                )
                self.structured_query(
                    f"""
                    MERGE (source: "{BASE_NODE_LABEL}" {{id: '{param["source_id"]}'}})
                    ON CREATE SET source.labels = append_label(source.labels, 'Chunk')
                    MERGE (target: "{BASE_NODE_LABEL}" {{id: '{param["target_id"]}'}})
                    ON CREATE SET target.labels = append_label(target.labels, 'Chunk')
                    WITH source, target
                    MERGE (source)-[r:"{param["label"]}"]->(target)
                    SET r += {{{formatted_properties}}}
                    RETURN count(*)
                    """
                )

    def get(
        self,
        properties: Optional[dict] = None,
        ids: Optional[List[str]] = None,
    ) -> List[LabelledNode]:
        """Get nodes."""
        wrapper = """SELECT t.name,
                            t.type,
                            (t.properties - 'labels') || '{{"embedding": null, "id": null}}'::jsonb AS properties
                     FROM ({})t"""
        cypher_statement = f'MATCH (e:"{BASE_NODE_LABEL}") '

        cypher_statement += "WHERE e.id IS NOT NULL "

        if ids:
            cypher_statement += "AND e.id IN {} ".format(ids)

        if properties:
            prop_list = []
            for i, prop in enumerate(properties):
                property = properties[prop] if not isinstance(properties[prop], str) else f"'{properties[prop]}'"
                prop_list.append(f'e."{prop}" = {property}')
            cypher_statement += " AND " + " AND ".join(prop_list)

        return_statement = f"""
            WITH e, e.labels as labels
            RETURN
            e.id AS name,
            CASE
                WHEN '{BASE_ENTITY_LABEL}' IN labels THEN
                    CASE
                        WHEN length(labels) > 2 THEN labels[2]
                        WHEN length(labels) > 1 THEN labels[1]
                        ELSE NULL
                    END
                ELSE labels[0]
            END AS type,
            properties(e) AS properties
        """
        cypher_statement += return_statement
        response = self.structured_query(wrapper.format(cypher_statement))
        response = response if response else []

        nodes = []
        for record in response:
            if "text" in record["properties"] or record["type"] is None:
                text = record["properties"].pop("text", "")
                nodes.append(
                    ChunkNode(
                        id_=record["name"],
                        text=text,
                        properties=remove_empty_values(record["properties"]),
                    )
                )
            else:
                nodes.append(
                    EntityNode(
                        name=record["name"],
                        label=record["type"],
                        properties=remove_empty_values(record["properties"]),
                    )
                )

        return nodes

    def get_triplets(
        self,
        entity_names: Optional[List[str]] = None,
        relation_names: Optional[List[str]] = None,
        properties: Optional[dict] = None,
        ids: Optional[List[str]] = None,
    ) -> List[Triplet]:
        cypher_statement = "MATCH (e)-[r]->(t) "
        cypher_statement += f"WHERE '{BASE_ENTITY_LABEL}' IN e.labels "

        if entity_names or relation_names or properties or ids:
            cypher_statement += "AND "

        if entity_names:
            cypher_statement += "e.name IN {} ".format(entity_names)

        if relation_names and entity_names:
            cypher_statement += "AND "

        if relation_names:
            cypher_statement += "type(r) IN {} ".format(relation_names)

        if ids:
            cypher_statement += "e.id IN {} ".format(ids)

        if properties:
            prop_list = []
            for i, prop in enumerate(properties):
                property = properties[prop] if not isinstance(properties[prop], str) else f"'{properties[prop]}'"
                prop_list.append(f'e."{prop}" = {property}')
            cypher_statement += " AND ".join(prop_list)

        return_statement = f"""
        AND NOT ANY(label IN e.labels WHERE label = 'Chunk')
            WITH *, e.labels as e_labels, t.labels as t_labels
            RETURN type(r) as type, properties(r) as rel_prop, e.id as source_id,
            CASE
                WHEN '{BASE_ENTITY_LABEL}' IN e_labels THEN
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
                WHEN '{BASE_ENTITY_LABEL}' IN t_labels THEN
                    CASE
                        WHEN length(t_labels) > 2 THEN t_labels[2]
                        WHEN length(t_labels) > 1 THEN t_labels[1]
                        ELSE NULL
                    END
                ELSE t_labels[0]
            END AS target_type, properties(t) AS target_properties LIMIT 100
        """

        cypher_statement += return_statement
        wrapper = """
                    SELECT t.type,
                           t.rel_prop,
                           t.source_id,
                           t.source_type,
                           (t.source_properties - 'labels') || '{{"embedding": null, "name": null}}'::jsonb AS source_properties,
                           t.target_id,
                           t.target_type,
                           (t.target_properties - 'labels') || '{{"embedding": null, "name": null}}'::jsonb AS target_properties
                    FROM ({})t;
        """
        data = self.structured_query(wrapper.format(cypher_statement))
        data = data if data else []

        triplets = []
        for record in data:
            source = EntityNode(
                name=record["source_id"],
                label=record["source_type"],
                properties=remove_empty_values(record["source_properties"]),
            )
            target = EntityNode(
                name=record["target_id"],
                label=record["target_type"],
                properties=remove_empty_values(record["target_properties"]),
            )
            rel = Relation(
                source_id=record["source_id"],
                target_id=record["target_id"],
                label=record["type"],
                properties=remove_empty_values(record["rel_prop"]),
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
        cypher_statement = f"""
            UNWIND {[0] if len(ids) == 1 else f'range(0, {len(ids)} - 1)::jsonb'} AS idx
            MATCH (e:"{BASE_NODE_LABEL}")
            WHERE e.id = {ids}[idx]
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
            LIMIT {limit}
            RETURN source.id AS source_id,
                CASE
                    WHEN '{BASE_ENTITY_LABEL}' IN source_labels THEN
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
                    WHEN '{BASE_ENTITY_LABEL}' IN target_labels THEN
                        CASE
                            WHEN length(target_labels) > 2 THEN target_labels[2]
                            WHEN length(target_labels) > 1 THEN target_labels[1] ELSE NULL
                        END
                    ELSE target_labels[0]
                END AS target_type,
                properties(endNode) AS target_properties,
                idx
            ORDER BY idx
            LIMIT {limit}
            """
        wrapper = """SELECT t.source_id,
                            t.source_type,
                            (t.source_properties - 'labels') || '{{"embedding": null, "id": null}}'::jsonb AS source_properties,
                            t.type,
                            t.rel_properties,
                            t.target_id,
                            t.target_type,
                            (t.target_properties - 'labels') || '{{"embedding": null, "id": null}}'::jsonb AS target_properties
                      FROM ({})t;
          """
        response = self.structured_query(wrapper.format(cypher_statement))
        response = response if response else []

        ignore_rels = ignore_rels or []
        for record in response:
            if record["type"] in ignore_rels:
                continue

            source = EntityNode(
                name=record["source_id"],
                label=record["source_type"],
                properties=remove_empty_values(record["source_properties"]),
            )
            target = EntityNode(
                name=record["target_id"],
                label=record["target_type"],
                properties=remove_empty_values(record["target_properties"]),
            )
            rel = Relation(
                source_id=record["source_id"],
                target_id=record["target_id"],
                label=record["type"],
                properties=remove_empty_values(record["rel_properties"]),
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
                f"MATCH (n) WHERE n.name IN {entity_names} DETACH DELETE n"
            )

        if ids:
            self.structured_query(
                f"MATCH (n) WHERE n.id IN {ids} DETACH DELETE n"
            )

        if relation_names:
            for rel in relation_names:
                self.structured_query(f'MATCH ()-[r:"{rel}"]->() DELETE r')

        if properties:
            cypher = "MATCH (e) WHERE "
            prop_list = []
            for i, prop in enumerate(properties):
                property = properties[prop] if not isinstance(properties[prop], str) else f"'{properties[prop]}'"
                prop_list.append(f'e."{prop}" = {property}')
            cypher += " AND ".join(prop_list)
            self.structured_query(cypher + " DETACH DELETE e")

    def vector_query(
        self, query: VectorStoreQuery, **kwargs: Any
    ) -> Tuple[List[LabelledNode], List[float]]:
        """Query the graph store with a vector store query."""
        if self._supports_vector_index:
            vector_query = f"""
                            SELECT
                                t.name,
                                t.type,
                                (t.properties - 'labels') || '{{"embedding": null, "name": null, "id": null}}'::jsonb AS properties,
                                1 - ((t.properties->>'embedding')::vector(3) <=> '{query.query_embedding}'::vector({self.vector_dimension})) AS similarity
                            FROM (
                                MATCH (n: "__Node__")
                                WITH n, n.labels AS labels
                                RETURN n.id as name,
                                       properties(n) AS properties,
                                       CASE
                                            WHEN '__Entity__' IN labels THEN
                                                CASE
                                                    WHEN length(labels) > 2 THEN labels[2]
                                                    WHEN length(labels) > 1 THEN labels[1]
                                                    ELSE NULL
                                                END
                                            ELSE labels[0]
                                       END AS type   
                            )t ORDER BY
                                (properties->>'embedding')::vector(3) <=> '[0.1, 0.2, 0.31]'::vector(3)
                            LIMIT {query.similarity_top_k};
                            """
            data = self.structured_query(vector_query)
        elif self._supports_vector_store:
            wrapper = """SELECT t.name,
                                t.type,
                                t.similarity,
                                (t.properties - 'labels') || '{{"embedding": null, "name": null, "id": null}}'::jsonb AS properties
                        FROM ({})t
                        """
            vector_query = f"""
                            MATCH (n: "{BASE_NODE_LABEL}")
                            WITH n,
                                n.labels AS labels,
                                {query.query_embedding}::vector <=> n.embedding::vector AS cos_d
                            ORDER BY cos_d
                            LIMIT {query.similarity_top_k}
                            RETURN n.id as name,
                                properties(n) AS properties,
                                1-cos_d as similarity,
                                CASE
                                        WHEN '{BASE_ENTITY_LABEL}' IN labels THEN
                                            CASE
                                                WHEN length(labels) > 2 THEN labels[2]
                                                WHEN length(labels) > 1 THEN labels[1]
                                                ELSE NULL
                                            END
                                        ELSE labels[0]
                                END AS type
                            """
            data = self.structured_query(wrapper.format(vector_query))
        else:
            data = []
        data = data if data else []

        nodes = []
        scores = []
        for record in data:
            node = EntityNode(
                name=record["name"],
                label=record["type"],
                properties=remove_empty_values(record["properties"]),
            )
            nodes.append(node)
            scores.append(record["similarity"])

        return (nodes, scores)

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

    @require_psycopg2
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
        with self._get_cursor() as curs:
            try:
                curs.execute(query)
                self.connection.commit()
            except psycopg2.Error as e:
                self.connection.rollback()
                raise AgensQueryException(
                    {
                        "message": "Error executing graph query: {}".format(query),
                        "detail": str(e),
                    }
                )
            try:
                data = curs.fetchall()
            except psycopg2.ProgrammingError:
                data = []  # Handle queries that don’t return data

            if data is None:
                result = []
            # convert to dictionaries
            else:
                result = [self._record_to_dict(d) for d in data]

            if self.sanitize_query_output:
                result = [value_sanitize(el) for el in result]

            return result

    @require_psycopg2
    def _get_node_properties(self) -> Dict[str, Any]:
        node_properties = {}
        with self._get_cursor() as curs:
            execute_query(curs, node_properties_query)
            rows = curs.fetchall()

            for row in rows:                
                node_properties[row.label] = row.props

        return node_properties

    @require_psycopg2
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
