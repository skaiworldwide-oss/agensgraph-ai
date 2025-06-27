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

from __future__ import annotations

import json
import re
from hashlib import md5
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Pattern, Tuple, Union

from langchain_community.graphs.graph_document import GraphDocument
from langchain_community.graphs.graph_store import GraphStore
from functools import wraps

import psycopg2
import psycopg2.extras

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

node_properties_query = f"""
    MATCH (a)
    UNWIND keys(properties(a)) AS prop
    WITH label(a) as label, prop, properties(a)[prop] AS value 
    WITH                
        label,
        prop AS property,
        COLLECT(DISTINCT value) AS values
    RETURN label, COLLECT(DISTINCT {{'property': property, type: typeof(values[0])}}) as props;
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

triple_query = """
    MATCH (start_node)-[r]->(end_node)
    WITH labels(start_node) AS start, type(r) AS relationship_type, labels(end_node) AS endd, keys(r) AS relationship_properties
    UNWIND endd as end_label
    RETURN DISTINCT {start: start[0], type: relationship_type, end: end_label} AS output;
"""

def require_psycopg2(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            import psycopg2
        except ImportError:
            raise ImportError(
                "Could not import psycopg2 python package. "
                "Please install it with `pip install psycopg2`."
            )
        return func(*args, **kwargs)
    return wrapper

def execute_query(cursor, query, error_message = "Error executing graph query"):
    try:
        cursor.execute(query)
    except psycopg2.Error as e:
        raise AgensQueryException(
            {
                "message": error_message,
                "detail": str(e),
            }
        )

class AgensQueryException(Exception):
    """Exception for the Agensgraph queries."""

    def __init__(self, exception: Union[str, Dict]) -> None:
        if isinstance(exception, dict):
            self.message = exception["message"] if "message" in exception else "unknown"
            self.details = exception["details"] if "details" in exception else "unknown"
        else:
            self.message = exception
            self.details = "unknown"

    def get_message(self) -> str:
        return self.message

    def get_details(self) -> Any:
        return self.details


class AgensGraph(GraphStore):
    """
    Agensgraph wrapper for graph operations.

    Args:
        graph_name (str): the name of the graph to connect to or create
        conf (Dict[str, Any]): the pgsql connection config passed directly
            to psycopg2.connect
        create (bool): if True and graph doesn't exist, attempt to create it

    *Security note*: Make sure that the database connection uses credentials
        that are narrowly-scoped to only include necessary permissions.
        Failure to do so may result in data corruption or loss, since the calling
        code may attempt commands that would result in deletion, mutation
        of data if appropriately prompted or reading sensitive data if such
        data is present in the database.
        The best way to guard against such negative outcomes is to (as appropriate)
        limit the permissions granted to the credentials used with this tool.

        See https://python.langchain.com/docs/security for more information.
    """

    # precompiled regex for checking chars in graph labels and
    # identifying record as vertex or edge
    label_regex: Pattern = re.compile("[^0-9a-zA-Z]+")
    vertex_regex: Pattern = re.compile(r"(\w+)\[(\d+\.\d+)\](\{.*\})")
    edge_regex: Pattern = re.compile(r"(\w+)\[(\d+\.\d+)\]\[(\d+\.\d+),\s*(\d+\.\d+)\](\{.*\})")

    @require_psycopg2
    def __init__(
        self, graph_name: str, conf: Dict[str, Any], create: bool = False
    ) -> None:
        """Create a new Agensgraph Graph instance."""

        self.graph_name = graph_name
        self.connection = psycopg2.connect(**conf)

        with self._get_cursor() as curs:
            # check if graph with name graph_name exists
            graph_id_query = (
                """SELECT oid as graphid FROM ag_graph WHERE graphname = '{}';""".format(
                    graph_name
                )
            )

            execute_query(curs, graph_id_query)
            data = curs.fetchone()

            # if graph doesn't exist and create is True, create it
            if data is None:
                if create:
                    create_statement = """
                        CREATE GRAPH {};
                    """.format(graph_name)
                    execute_query(curs, create_statement)
                    self.connection.commit()
                else:
                    raise Exception(
                        (
                            'Graph "{}" does not exist in the database '
                            + 'and "create" is set to False'
                        ).format(graph_name)
                    )

                execute_query(curs, graph_id_query)
                data = curs.fetchone()

            # store graph id and refresh the schema
            self.graphid = data.graphid

            # set the graph path to the current graph and declare some functions
            graph_path = """SET graph_path = '{}';""".format(self.graph_name)
            execute_query(curs, graph_path)
            execute_query(curs, typeof_function)
            self.connection.commit()

        self.refresh_schema()

    @require_psycopg2
    def _get_cursor(self) -> psycopg2.extras.NamedTupleCursor:
        cursor = self.connection.cursor(cursor_factory=psycopg2.extras.NamedTupleCursor)
        return cursor

    @require_psycopg2
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
            execute_query(curs, triple_query)
            rows = curs.fetchall()
            triple_schema = [row.output for row in rows]
        
        return triple_schema

    def _get_triples_str(self, e_labels: List[str]) -> List[str]:
        """
        Get a set of distinct relationship types (as a list of strings) in the graph
        to be used as context by an llm.

        Args:
            e_labels (List[str]): a list of edge labels to filter for

        Returns:
            List[str]: relationships as a list of strings in the format
                "(:"<from_label>")-[:"<edge_label>"]->(:"<to_label>")"
        """

        triples = self._get_triples(e_labels)

        return self._format_triples(triples)

    @staticmethod
    def _format_triples(triples: List[Dict[str, str]]) -> List[str]:
        """
        Convert a list of relationships from dictionaries to formatted strings
        to be better readable by an llm

        Args:
            triples (List[Dict[str,str]]): a list relationships in the form
                {'start':<from_label>, 'type':<edge_label>, 'end':<from_label>}

        Returns:
            List[str]: a list of relationships in the form
                "(:"<from_label>")-[:"<edge_label>"]->(:"<to_label>")"
        """
        triple_template = '(:"{start}")-[:"{type}"]->(:"{end}")'
        triple_schema = [triple_template.format(**triple) for triple in triples]

        return triple_schema

    @require_psycopg2
    def _get_node_properties(self) -> List[Dict[str, Any]]:
        """
        Fetch a list of available node properties by node label to be used
        as context for an llm

        Args:
            n_labels (List[str]): a list of node labels to filter for

        Returns:
            List[Dict[str, Any]]: a list of node labels and
                their corresponding properties in the form
                "{
                    'labels': <node_label>,
                    'properties': [
                        {
                            'property': <property_name>,
                            'type': <property_type>
                        },...
                        ]
                }"
        """

        node_properties = []
        with self._get_cursor() as curs:
            execute_query(curs, node_properties_query)
            rows = curs.fetchall()
            
            for row in rows:
                node_properties.append(
                    {
                        "labels": row.label,
                        "properties": row.props
                    }
                )

        return node_properties

    @require_psycopg2
    def _get_edge_properties(self) -> List[Dict[str, Any]]:
        """
        Fetch a list of available edge properties by edge label to be used
        as context for an llm

        Args:
            e_labels (List[str]): a list of edge labels to filter for

        Returns:
            List[Dict[str, Any]]: a list of edge labels
                and their corresponding properties in the form
                "{
                    'labels': <edge_label>,
                    'properties': [
                        {
                            'property': <property_name>,
                            'type': <property_type>
                        },...
                        ]
                }"
        """
        edge_properties = []
        with self._get_cursor() as curs:
            execute_query(curs, edge_properties_query)
            rows = curs.fetchall()
            
            for row in rows:
                edge_properties.append(
                    {
                        "type": row.label,
                        "properties": row.props
                    }
                )

        return edge_properties

    def refresh_schema(self) -> None:
        """
        Refresh the graph schema information by updating the available
        labels, relationships, and properties
        """

        # fetch graph schema information
        node_properties = self._get_node_properties()
        edge_properties = self._get_edge_properties()
        triple_schema = self._get_triples()

        # update the formatted string representation
        self.schema = f"""
        Node properties are the following:
        {node_properties}
        Relationship properties are the following:
        {edge_properties}
        The relationships are the following:
        {self._format_triples(triple_schema)}
        """

        # update the dictionary representation
        self.structured_schema = {
            "node_props": {el["labels"]: el["properties"] for el in node_properties},
            "rel_props": {el["type"]: el["properties"] for el in edge_properties},
            "relationships": triple_schema,
            "metadata": {},
        }

    @property
    def get_schema(self) -> str:
        """Returns the schema of the Graph"""
        return self.schema

    @property
    def get_structured_schema(self) -> Dict[str, Any]:
        """Returns the structured schema of the Graph"""
        return self.structured_schema

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
                vertex = AgensGraph.vertex_regex.match(v)
                if vertex:
                    label, vertex_id, properties = vertex.groups()
                    properties = json.loads(properties)
                    vertices[str(vertex_id)] = properties

        # iterate returned fields and parse appropriately
        for k in record._fields:
            v = getattr(record, k)

            if isinstance(v, str):
                vertex = AgensGraph.vertex_regex.match(v)
                edge = AgensGraph.edge_regex.match(v)

                if vertex:
                    d[k] = json.loads(vertex.group(3))

                # convert edge from id-label->id by replacing id with node information
                # we only do this if the vertex was also returned in the query
                # this is an attempt to be consistent with neo4j implementation
                elif edge:
                    elabel, edge_id, start_id, end_id, properties = edge.groups()
                    d[k] = (
                        vertices.get(start_id, {}),
                        elabel,
                        vertices.get(end_id, {}),
                    )
                else:
                    try:
                        d[k] = json.loads(v)
                    except json.JSONDecodeError:
                        d[k] = v

            else:
                d[k] = v

        return d

    @require_psycopg2
    def query(self, query: str, params: dict = {}) -> List[Dict[str, Any]]:
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

            return result

    @staticmethod
    def _format_properties(
        properties: Dict[str, Any], id: Union[str, None] = None
    ) -> str:
        """
        Convert a dictionary of properties to a string representation that
        can be used in a cypher query insert/merge statement.

        Args:
            properties (Dict[str,str]): a dictionary containing node/edge properties
            id (Union[str, None]): the id of the node or None if none exists

        Returns:
            str: the properties dictionary as a properly formatted string
        """
        props = []
        # wrap property key in double quotes to escape
        for k, v in properties.items():
            prop = f'"{k}": \'{v}\'' if isinstance(v, str) else f'"{k}": {v}'
            props.append(prop)
        if id is not None and "id" not in properties:
            props.append(
                f"id: '{id}'" if isinstance(id, str) else f"id: {id}"
            )
        return "{" + ", ".join(props) + "}"

    @staticmethod
    def clean_graph_labels(label: str) -> str:
        """
        remove any disallowed characters from a label and replace with '_'

        Args:
            label (str): the original label

        Returns:
            str: the sanitized version of the label
        """
        return re.sub(AgensGraph.label_regex, "_", label)

    def add_graph_documents(
        self, graph_documents: List[GraphDocument], include_source: bool = False
    ) -> None:
        """
        insert a list of graph documents into the graph

        Args:
            graph_documents (List[GraphDocument]): the list of documents to be inserted
            include_source (bool): if True add nodes for the sources
                with MENTIONS edges to the entities they mention

        Returns:
            None
        """
        # Ensure that the label used in merge exists (due to bug in agensgraph)
        # query for inserting nodes
        node_insert_query = (
            """
            CREATE VLABEL IF NOT EXISTS "{label}";
            MERGE (n:"{label}" {{id: '{id}'}})
            SET n = {properties};
            """
            if not include_source
            else """
            CREATE VLABEL IF NOT EXISTS "{label}";
            CREATE VLABEL IF NOT EXISTS "Document";
            CREATE ELABEL IF NOT EXISTS "MENTIONS";
            MERGE (n:"{label}" {properties})
            MERGE (d:"Document" {d_properties})
            MERGE (d)-[:"MENTIONS"]->(n)
        """
        )

        # query for inserting edges
        edge_insert_query = """
            CREATE VLABEL IF NOT EXISTS "{f_label}";
            CREATE VLABEL IF NOT EXISTS "{t_label}";
            CREATE ELABEL IF NOT EXISTS "{r_label}";
            MERGE ("from":"{f_label}" {f_properties})
            MERGE ("to":"{t_label}" {t_properties})
            MERGE ("from")-[:"{r_label}" {r_properties}]->("to")
        """
        # iterate docs and insert them
        for doc in graph_documents:
            # if we are adding sources, create an id for the source
            if include_source:
                if not doc.source.metadata.get("id"):
                    doc.source.metadata["id"] = md5(
                        doc.source.page_content.encode("utf-8")
                    ).hexdigest()

            # insert entity nodes
            for node in doc.nodes:
                node.properties["id"] = node.id
                if include_source:
                    query = node_insert_query.format(
                        label=node.type,
                        properties=self._format_properties(node.properties),
                        d_properties=self._format_properties(doc.source.metadata),
                    )
                else:
                    query = node_insert_query.format(
                        label=AgensGraph.clean_graph_labels(node.type),
                        properties=self._format_properties(node.properties),
                        id=node.id,
                    )
                self.query(query)

            # insert relationships
            for edge in doc.relationships:
                edge.source.properties["id"] = edge.source.id
                edge.target.properties["id"] = edge.target.id
                inputs = {
                    "f_label": AgensGraph.clean_graph_labels(edge.source.type),
                    "f_properties": self._format_properties(edge.source.properties),
                    "t_label": AgensGraph.clean_graph_labels(edge.target.type),
                    "t_properties": self._format_properties(edge.target.properties),
                    "r_label": AgensGraph.clean_graph_labels(edge.type).upper(),
                    "r_properties": self._format_properties(edge.properties),
                }

                query = edge_insert_query.format(**inputs)
                self.query(query)
