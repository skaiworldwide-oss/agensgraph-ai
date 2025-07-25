from __future__ import annotations

import enum
import logging
import os
from hashlib import md5
import re, json
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    NamedTuple,
    Pattern,
    Union
)

import psycopg
from psycopg import sql
from psycopg.types.json import Jsonb

import numpy as np
from langchain_core._api.deprecation import deprecated
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.utils import get_from_dict_or_env
from langchain_core.vectorstores import VectorStore

from langchain_agensgraph.graphs.agensgraph import AgensGraph
from langchain_community.vectorstores.utils import (
    DistanceStrategy,
    maximal_marginal_relevance,
)

DEFAULT_DISTANCE_STRATEGY = DistanceStrategy.COSINE
DISTANCE_STRATEGY_OPS = {
    DistanceStrategy.EUCLIDEAN_DISTANCE: "vector_l2_ops",
    DistanceStrategy.COSINE: "vector_cosine_ops",
    DistanceStrategy.MAX_INNER_PRODUCT: "vector_ip_ops",
    DistanceStrategy.DOT_PRODUCT: "vector_ip_ops",
    DistanceStrategy.JACCARD: "vector_jaccard_ops",
}

DISTANCE_OPERATOR_MAPPING = {
    DistanceStrategy.EUCLIDEAN_DISTANCE: "<->",
    DistanceStrategy.COSINE: "<=>",
    DistanceStrategy.MAX_INNER_PRODUCT: "<#>",
    DistanceStrategy.DOT_PRODUCT: "<#>",
    DistanceStrategy.JACCARD: "<%>",
}

COMPARISONS_TO_NATIVE = {
    "$eq": "=",
    "$ne": "<>",
    "$lt": "<",
    "$lte": "<=",
    "$gt": ">",
    "$gte": ">=",
}

SPECIAL_CASED_OPERATORS = {
    "$in",
    "$nin",
    "$between",
}

TEXT_OPERATORS = {
    "$like",
    "$ilike",
}

LOGICAL_OPERATORS = {"$and", "$or"}

SUPPORTED_OPERATORS = (
    set(COMPARISONS_TO_NATIVE)
    .union(TEXT_OPERATORS)
    .union(LOGICAL_OPERATORS)
    .union(SPECIAL_CASED_OPERATORS)
)

get_vector_index_info_function = r"""
    CREATE OR REPLACE FUNCTION ag_list_vector_indexes(
        index_name text DEFAULT NULL,
        node_label text DEFAULT NULL,
        embedding_node_property text DEFAULT NULL
    )
    RETURNS TABLE (
        name text,
        labelortype text,
        property text,
        entitytype text,
        dimensions int
    )
    LANGUAGE sql
    AS $$
        SELECT
            c.relname AS name,
            l.labname AS labelOrType,
            CASE
                WHEN indexdef ~ '\(+([a-zA-Z_][a-zA-Z0-9_]*)\)+::' THEN regexp_replace(indexdef, '.*\(+([a-zA-Z_][a-zA-Z0-9_]*)\)+::.*', '\1')
                ELSE NULL
            END AS property,
            CASE
                WHEN l.labkind = 'v' THEN 'NODE'
                WHEN l.labkind = 'e' THEN 'RELATIONSHIP'
                ELSE 'UNKNOWN'
            END AS entityType,
            CASE
                WHEN indexdef ~ 'vector\((\d+)\)' THEN (regexp_match(indexdef, 'vector\((\d+)\)'))[1]::int
                ELSE NULL
            END AS dimensions
        FROM
            pg_catalog.pg_index i
        JOIN pg_catalog.pg_class c ON c.oid = i.indexrelid
        JOIN pg_catalog.ag_label l ON i.indrelid = l.relid
        JOIN pg_catalog.ag_graph g ON l.graphid = g.oid
        JOIN LATERAL pg_catalog.ag_get_propindexdef(c.oid) AS indexdef ON true
        WHERE
            g.graphname = current_setting('graph_path')
            AND i.indexprs IS NOT NULL
            AND indexdef ~ '::vector\(\d+\)'
            AND (
                index_name IS NULL AND node_label IS NULL AND embedding_node_property IS NULL
                OR (
                    (index_name IS NOT NULL AND c.relname = index_name)
                    OR (
                        node_label IS NOT NULL
                        AND embedding_node_property IS NOT NULL
                        AND l.labname = node_label
                        AND CASE
                            WHEN indexdef ~ '\(\(\(([^)]+)\)::' THEN regexp_replace(indexdef, '.*\(\(\(([^)]+)\)::.*', '\1')
                            ELSE NULL
                        END = embedding_node_property
                    )
                )
            )
    $$;


"""

get_keyword_index_info_function = r"""
    CREATE OR REPLACE FUNCTION ag_list_text_indexes(
        index_name text DEFAULT NULL,
        node_label text DEFAULT NULL,
        text_node_properties text[] DEFAULT NULL
    )
    RETURNS TABLE (
        name text,
        labelortype text,
        properties text[],
        entitytype text
    )
    LANGUAGE sql
    AS $$
    WITH extracted_props AS (
        SELECT
            c.relname AS name,
            l.labname AS labelOrType,
            ARRAY(
                SELECT
                    trim(both '"' from trim(m[1]))
                FROM
                    regexp_matches(indexdef, 'to_tsvector\((?:[^,]+),\s*([^)]+)\)', 'g') AS m
            ) AS props,
            CASE
                WHEN l.labkind = 'v' THEN 'NODE'
                WHEN l.labkind = 'e' THEN 'RELATIONSHIP'
                ELSE 'UNKNOWN'
            END AS entityType
        FROM
            pg_catalog.pg_index i
        JOIN pg_catalog.pg_class c ON c.oid = i.indexrelid
        JOIN pg_catalog.ag_label l ON i.indrelid = l.relid
        JOIN pg_catalog.ag_graph g ON l.graphid = g.oid
        JOIN LATERAL pg_catalog.ag_get_propindexdef(c.oid) AS indexdef ON true
        WHERE
            g.graphname = current_setting('graph_path')
            AND i.indexprs IS NOT NULL
            AND indexdef ~ 'to_tsvector\('
    )
    SELECT
        name,
        labelOrType,
        props AS properties,
        entityType
    FROM
        extracted_props
    WHERE
        (
            (index_name IS NULL OR name = index_name)
            AND (node_label IS NULL OR labelOrType = node_label)
            AND (
                text_node_properties IS NULL OR
                array(SELECT unnest(props) ORDER BY 1) = array(SELECT unnest(text_node_properties) ORDER BY 1)
            )
        );
    $$;

"""

class SearchType(str, enum.Enum):
    """Enumerator of the Distance strategies."""

    VECTOR = "vector"
    HYBRID = "hybrid"


DEFAULT_SEARCH_TYPE = SearchType.VECTOR

class IndexType(str, enum.Enum):
    """Enumerator of the index types."""

    NODE = "NODE"
    RELATIONSHIP = "RELATIONSHIP"

class VectorIndexAM(str, enum.Enum):
    """Enumerator of the vector index access methods."""

    HNSW = "HNSW"
    IVFFLAT = "IVFLLAT"
    DISKANN = "DISKANN"

class FullTextIndexAM(str, enum.Enum):
    """Enumerator of the full text index access methods."""

    GIN = "GIN"
    GIST = "GIST"
    SPGIST = "SPGIST"

DEFAULT_INDEX_TYPE = IndexType.NODE
DEFAULT_VECTOR_INDEX_AM = VectorIndexAM.HNSW
DEFAULT_FULLTEXT_INDEX_AM = FullTextIndexAM.GIN

def check_if_not_null(props: List[str], values: List[Any]) -> None:
    """Check if the values are not None or empty string"""
    for prop, value in zip(props, values):
        if not value:
            raise ValueError(f"Parameter `{prop}` must not be None or empty string")

def sort_by_index_name(
    lst: List[Dict[str, Any]], index_name: str
) -> List[Dict[str, Any]]:
    """Sort first element to match the index_name if exists"""
    return sorted(lst, key=lambda x: x.get("name") != index_name)

def remove_lucene_chars(text: str) -> str:
    """Remove Lucene special characters"""
    special_chars = [
        "+",
        "-",
        "&",
        "|",
        "!",
        "(",
        ")",
        "{",
        "}",
        "[",
        "]",
        "^",
        '"',
        "~",
        "*",
        "?",
        ":",
        "\\",
    ]
    for char in special_chars:
        if char in text:
            text = text.replace(char, " ")
    return text.strip()

def dict_to_yaml_str(input_dict: Dict, indent: int = 0) -> str:
    """
    Convert a dictionary to a YAML-like string without using external libraries.

    Parameters:
    - input_dict (dict): The dictionary to convert.
    - indent (int): The current indentation level.

    Returns:
    - str: The YAML-like string representation of the input dictionary.
    """
    yaml_str = ""
    for key, value in input_dict.items():
        padding = "  " * indent
        if isinstance(value, dict):
            yaml_str += f"{padding}{key}:\n{dict_to_yaml_str(value, indent + 1)}"
        elif isinstance(value, list):
            yaml_str += f"{padding}{key}:\n"
            for item in value:
                yaml_str += f"{padding}- {item}\n"
        else:
            yaml_str += f"{padding}{key}: {value}\n"
    return yaml_str

def combine_queries(
    input_queries: List[Tuple[str, Dict[str, Any]]], operator: str
) -> Tuple[str, Dict[str, Any]]:
    """Combine multiple queries with an operator."""

    # Initialize variables to hold the combined query and parameters
    combined_query: str = ""
    combined_params: Dict = {}
    param_counter: Dict = {}

    for query, params in input_queries:
        # Process each query fragment and its parameters
        new_query = query
        for param, value in params.items():
            # Update the parameter name to ensure uniqueness
            if param in param_counter:
                param_counter[param] += 1
            else:
                param_counter[param] = 1
            new_param_name = f"{param}_{param_counter[param]}"

            # Replace the parameter in the query fragment
            new_query = new_query.replace(f"%({param})s", f"%({new_param_name})s")
            # Add the parameter to the combined parameters dictionary
            combined_params[new_param_name] = value

        # Combine the query fragments with an AND operator
        if combined_query:
            combined_query += f" {operator} "
        combined_query += f"({new_query})"

    return combined_query, combined_params

def collect_params(
    input_data: List[Tuple[str, Dict[str, str]]],
) -> Tuple[List[str], Dict[str, Any]]:
    """Transform the input data into the desired format.

    Args:
    - input_data (list of tuples): Input data to transform.
      Each tuple contains a string and a dictionary.

    Returns:
    - tuple: A tuple containing a list of strings and a dictionary.
    """
    # Initialize variables to hold the output parts
    query_parts = []
    params = {}

    # Loop through each item in the input data
    for query_part, param in input_data:
        # Append the query part to the list
        query_parts.append(query_part)
        # Update the params dictionary with the param dictionary
        params.update(param)

    # Return the transformed data
    return (query_parts, params)

def _handle_field_filter(
    field: str, value: Any, param_number: int = 1
) -> Tuple[str, Dict]:
    """Create a filter for a specific field.

    Args:
        field: name of field
        value: value to filter
            If provided as is then this will be an equality filter
            If provided as a dictionary then this will be a filter, the key
            will be the operator and the value will be the value to filter by
        param_number: sequence number of parameters used to map between param
           dict and Cypher snippet

    Returns a tuple of
        - Cypher filter snippet
        - Dictionary with parameters used in filter snippet
    """
    if not isinstance(field, str):
        raise ValueError(
            f"field should be a string but got: {type(field)} with value: {field}"
        )

    if field.startswith("$"):
        raise ValueError(
            f"Invalid filter condition. Expected a field but got an operator: {field}"
        )

    # Allow [a-zA-Z0-9_], disallow $ for now until we support escape characters
    if not field.isidentifier():
        raise ValueError(f"Invalid field name: {field}. Expected a valid identifier.")

    if isinstance(value, dict):
        # This is a filter specification
        if len(value) != 1:
            raise ValueError(
                "Invalid filter condition. Expected a value which "
                "is a dictionary with a single key that corresponds to an operator "
                f"but got a dictionary with {len(value)} keys. The first few "
                f"keys are: {list(value.keys())[:3]}"
            )
        operator, filter_value = list(value.items())[0]
        # Verify that that operator is an operator
        if operator not in SUPPORTED_OPERATORS:
            raise ValueError(
                f"Invalid operator: {operator}. Expected one of {SUPPORTED_OPERATORS}"
            )
    else:  # Then we assume an equality operator
        operator = "$eq"
        filter_value = value

    if isinstance(filter_value, (list, dict, str)) and \
       not operator in {"$in", "$nin"}:
        filter_value = Jsonb(filter_value)

    if operator in COMPARISONS_TO_NATIVE:
        # Then we implement an equality filter
        # native is trusted input
        native = COMPARISONS_TO_NATIVE[operator]
        query_snippet = f'n."{field}" {native} %(param_{param_number})s'
        query_param = {f"param_{param_number}": filter_value}
        return (query_snippet, query_param)
    elif operator == "$between":
        low, high = filter_value
        query_snippet = (
            f'%(param_{param_number}_low)s <= n."{field}" AND n."{field}" <= %(param_{param_number}_high)s'
        )
        query_param = {
            f"param_{param_number}_low": low,
            f"param_{param_number}_high": high,
        }
        return (query_snippet, query_param)

    elif operator in {"$in", "$nin", "$like", "$ilike"}:
        # We'll do force coercion to text
        if operator in {"$in", "$nin"}:
            for val in filter_value:
                if not isinstance(val, (str, int, float)):
                    raise NotImplementedError(
                        f"Unsupported type: {type(val)} for value: {val}"
                    )
            filter_value = Jsonb(filter_value)
        if operator in {"$in"}:
            query_snippet = f'n."{field}" <@ %(param_{param_number})s'
            query_param = {f"param_{param_number}": filter_value}
            return (query_snippet, query_param)
        elif operator in {"$nin"}:
            query_snippet = f'NOT (n."{field}" NOT IN %(param_{param_number})s)'
            query_param = {f"param_{param_number}": filter_value}
            return (query_snippet, query_param)
        elif operator in {"$like"}:
            query_snippet = f'n."{field}" CONTAINS %(param_{param_number})s'
            query_param = {f"param_{param_number}": filter_value.rstrip("%")}
            return (query_snippet, query_param)
        elif operator in {"$ilike"}:
            query_snippet = f'toLower(n."{field}") CONTAINS %(param_{param_number})s'
            query_param = {f"param_{param_number}": filter_value.rstrip("%")}
            return (query_snippet, query_param)
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()

def construct_metadata_filter(filter: Dict[str, Any]) -> Tuple[str, Dict]:
    """Construct a metadata filter.

    Args:
        filter: A dictionary representing the filter condition.

    Returns:
        Tuple[str, Dict]
    """

    if isinstance(filter, dict):
        if len(filter) == 1:
            # The only operators allowed at the top level are $AND and $OR
            # First check if an operator or a field
            key, value = list(filter.items())[0]
            if key.startswith("$"):
                # Then it's an operator
                if key.lower() not in ["$and", "$or"]:
                    raise ValueError(
                        f"Invalid filter condition. Expected $and or $or but got: {key}"
                    )
            else:
                # Then it's a field
                return _handle_field_filter(key, filter[key])

            # Here we handle the $and and $or operators
            if not isinstance(value, list):
                raise ValueError(
                    f"Expected a list, but got {type(value)} for value: {value}"
                )
            if key.lower() == "$and":
                and_ = combine_queries(
                    [construct_metadata_filter(el) for el in value], "AND"
                )
                if len(and_) >= 1:
                    return and_
                else:
                    raise ValueError(
                        "Invalid filter condition. Expected a dictionary "
                        "but got an empty dictionary"
                    )
            elif key.lower() == "$or":
                or_ = combine_queries(
                    [construct_metadata_filter(el) for el in value], "OR"
                )
                if len(or_) >= 1:
                    return or_
                else:
                    raise ValueError(
                        "Invalid filter condition. Expected a dictionary "
                        "but got an empty dictionary"
                    )
            else:
                raise ValueError(
                    f"Invalid filter condition. Expected $and or $or but got: {key}"
                )
        elif len(filter) > 1:
            # Then all keys have to be fields (they cannot be operators)
            for key in filter.keys():
                if key.startswith("$"):
                    raise ValueError(
                        f"Invalid filter condition. Expected a field but got: {key}"
                    )
            # These should all be fields and combined using an $and operator
            and_multiple = collect_params(
                [
                    _handle_field_filter(k, v, index)
                    for index, (k, v) in enumerate(filter.items())
                ]
            )
            if len(and_multiple) >= 1:
                return " AND ".join(and_multiple[0]), and_multiple[1]
            else:
                raise ValueError(
                    "Invalid filter condition. Expected a dictionary "
                    "but got an empty dictionary"
                )
        else:
            raise ValueError("Got an empty dictionary for filters.")
        
def construct_vector_expression(variable: str, property: str,
                                embedding: List[float],
                                distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY) -> str:
    """Construct a vector expression for the Cypher query"""

    return f"""{variable}."{property}"::vector({len(embedding)}) {DISTANCE_OPERATOR_MAPPING[distance_strategy]} {embedding}::vector({len(embedding)})"""

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

class AgensgraphVector(VectorStore):
    """`AgensGraph` vector index.

    To use, you should have the ``psycopg`` python package installed.

    Args:
        url: AgensGraph connection url
        graph_name: The name of the graph to use in Agensgraph. Defaults to "vector_store".
        embedding: Any embedding function implementing
            `langchain.embeddings.base.Embeddings` interface.
        distance_strategy: The distance strategy to use. (default: COSINE)
        search_type: The type of search to be performed, either
            'vector' or 'hybrid'
        node_label: The label used for nodes in the AgensGraph database.
            (default: "Chunk")
        embedding_node_property: The property name in AgensGraph to store embeddings(powered by pgvector).
            (default: "embedding")
        text_node_property: The property name in AgensGraph to store the text.
            (default: "text")
        retrieval_query: The Cypher query to be used for customizing retrieval.
            If empty, a default query will be used.
        index_type: The type of index to be used, either
            'NODE' or 'RELATIONSHIP'
        pre_delete_collection: If True, will delete existing data if it exists.
            (default: False). Useful for testing.

    Example:
        .. code-block:: python

            from langchain_agensgraph.vectorstores.agensgraph_vector import AgensgraphVector
            from langchain_community.embeddings.openai import OpenAIEmbeddings

            url="postgresql://username:password@host:port/dbname"
            graph_name="my_graph"
            embeddings = OpenAIEmbeddings()
            vectorestore = AgensgraphVector.from_documents(
                embedding=embeddings,
                documents=docs,
                url=url
            )


    """

    vertex_regex: Pattern = re.compile(r"(\w+)\[(\d+\.\d+)\](\{.*\})")
    edge_regex: Pattern = re.compile(r"(\w+)\[(\d+\.\d+)\]\[(\d+\.\d+),\s*(\d+\.\d+)\](\{.*\})")

    def __init__(
        self,
        embedding: Embeddings,
        *,
        search_type: SearchType = SearchType.VECTOR,
        url: Optional[str] = None,
        graph_name: Optional[str] = "vector_store",
        keyword_index_name: Optional[str] = "keyword",
        index_name: str = "vector",
        node_label: str = "Chunk",
        embedding_node_property: str = "embedding",
        text_node_property: str = "text",
        text_node_properties: Optional[List[str]] = None,
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        logger: Optional[logging.Logger] = None,
        pre_delete_collection: bool = False,
        retrieval_query: str = "",
        relevance_score_fn: Optional[Callable[[float], float]] = None,
        index_type: IndexType = DEFAULT_INDEX_TYPE,
        vector_index_am: VectorIndexAM = DEFAULT_VECTOR_INDEX_AM,
        fulltext_index_am: FullTextIndexAM = DEFAULT_FULLTEXT_INDEX_AM,
        graph: Optional[AgensGraph] = None,
    ) -> None:
        # Allow only cosine and euclidean distance strategies
        if distance_strategy not in [
            DistanceStrategy.EUCLIDEAN_DISTANCE,
            DistanceStrategy.COSINE,
        ]:
            raise ValueError(
                "distance_strategy must be either 'EUCLIDEAN_DISTANCE' or 'COSINE'"
            )

        # Graph object takes precedent over env or input params
        if graph:
            self.connection = graph.connection
            self.graph_name = graph.graph_name
        else:
            url = get_from_dict_or_env({"url": url}, "url", "AGENSGRAPH_URL")

            # If graph is not provided, create a new one
            self.connection = psycopg.connect(url)
            self.graph_name = graph_name

        self.schema = ""

        # Verify if the version support vector index
        self.verify_vector_support()

        # Verify that required values are not null
        check_if_not_null(
            [
                "index_name",
                "node_label",
                "embedding_node_property",
                "text_node_property",
            ],
            [index_name, node_label, embedding_node_property, text_node_property],
        )

        self.embedding = embedding
        self._distance_strategy = distance_strategy
        self.index_name = index_name
        self.keyword_index_name = keyword_index_name
        self.node_label = node_label
        self.embedding_node_property = embedding_node_property
        self.text_node_property = text_node_property
        self.text_node_properties = text_node_properties
        self.logger = logger or logging.getLogger(__name__)
        self.override_relevance_score_fn = relevance_score_fn
        self.retrieval_query = retrieval_query
        self.search_type = search_type
        self._index_type = index_type
        self._vector_index_am = vector_index_am
        self._fulltext_index_am = fulltext_index_am
        # Calculate embedding dimension
        self.embedding_dimension = len(embedding.embed_query("foo"))

        # Create graph and set graph_path
        self.query(sql.SQL("CREATE GRAPH IF NOT EXISTS {}").format(
            sql.Identifier(self.graph_name)
        ))
        self.query(sql.SQL("SET graph_path = {}").format(
            sql.Identifier(self.graph_name)
        ))

        # Create some helper functions
        self.query(get_vector_index_info_function)
        self.query(get_keyword_index_info_function)

        # Delete existing data if flagged
        if pre_delete_collection:
            self.query(sql.SQL(
                "MATCH (n: {label}) DETACH DELETE n"
            ).format(label=sql.Identifier(self.node_label)))
            # Delete index
            self.query(sql.SQL(
                "DROP PROPERTY INDEX IF EXISTS {index}"
            ).format(index=sql.Identifier(self.index_name)))

    def retrieve_existing_index(self) -> Tuple[Optional[int], Optional[str]]:
        """
        Check if the vector index exists in the AgensGraph database
        and returns its embedding dimension.

        This method queries the AgensGraph database for existing indexes
        and attempts to retrieve the dimension of the vector index
        with the specified name. If the index exists, its dimension is returned.
        If the index doesn't exist, `None` is returned.

        Returns:
            int or None: The embedding dimension of the existing index if found.
        """
        index_information = self.query(
            """SELECT * FROM ag_list_vector_indexes(index_name => %(index_name)s,
                                                    node_label => %(node_label)s,
                                                    embedding_node_property => %(embedding_node_property)s)
            """,
            params={
                "index_name": self.index_name,
                "node_label": self.node_label,
                "embedding_node_property": self.embedding_node_property,
            },
        )
        # sort by index_name
        index_information = sort_by_index_name(index_information, self.index_name)
        try:
            print("DEBUG: Index information:", index_information[0])
            self.index_name = index_information[0]["name"]
            self.node_label = index_information[0]["labelortype"]
            self.embedding_node_property = index_information[0]["property"]
            self._index_type = index_information[0]["entitytype"]
            embedding_dimension = index_information[0]["dimensions"]

            return embedding_dimension, index_information[0]["entitytype"]
        except IndexError:
            return None, None

    def retrieve_existing_fts_index(self) -> Optional[str]:
        """
        Check if the fulltext index exists in the AgensGraph database

        This method queries the AgensGraph database for existing fts indexes
        with the specified name.

        Returns:
            (Tuple): keyword index information
        """

        index_information = self.query(
            """SELECT * FROM ag_list_text_indexes(index_name => %(index_name)s,
                                                  node_label => %(node_label)s,
                                                  text_node_properties => %(text_node_properties)s)
            """,
            params={
                "index_name": self.keyword_index_name,
                "node_label": self.node_label,
                "text_node_properties": self.text_node_properties or [self.text_node_property],
            },
        )
        # sort by index_name
        index_information = sort_by_index_name(index_information, self.index_name)
        try:
            self.keyword_index_name = index_information[0]["name"]
            self.text_node_property = index_information[0]["properties"][0]
            node_label = index_information[0]["labelortype"]

            print("DEBUG: Keyword index information:", index_information[0])
            return node_label
        except IndexError:
            return None

    def verify_label_existence(self) -> None:
        """Create label if it does not exist."""
        if self._index_type == IndexType.RELATIONSHIP:
            self.query(
                sql.SQL("CREATE ELABEL IF NOT EXISTS {}").format(
                    sql.Identifier(self.node_label)
                )
            )
        else:
            self.query(
                sql.SQL("CREATE VLABEL IF NOT EXISTS {}").format(
                    sql.Identifier(self.node_label)
                )
            )


    def create_new_index(self, vector_index_am: VectorIndexAM = None) -> None:
        """
        This method constructs a Cypher query and executes it
        to create a new vector index in AgensGraph.
        """
        if vector_index_am is None:
            vector_index_am = self._vector_index_am

        # make sure label exists
        self.verify_label_existence()
        index_query = """CREATE PROPERTY INDEX IF NOT EXISTS {index_name}
            ON {node_label} USING {vector_index_am}
            (({embedding_node_property}::vector({embedding_dimension})) {similarity_metric})"""

        self.query(
            sql.SQL(index_query).format(
                index_name=sql.Identifier(self.index_name),
                node_label=sql.Identifier(self.node_label),
                vector_index_am=sql.SQL(vector_index_am.value),
                embedding_node_property=sql.Identifier(self.embedding_node_property),
                embedding_dimension=self.embedding_dimension,
                similarity_metric=sql.SQL(
                    DISTANCE_STRATEGY_OPS[self._distance_strategy]
                )
            )
        )

    def create_new_keyword_index(self, fulltext_index_am: FullTextIndexAM = None) -> None:
        """
        This method constructs a Cypher query and executes it
        to create a new full text index in AgensGraph.
        """
        if fulltext_index_am is None:
            fulltext_index_am = self._fulltext_index_am
        # make sure label exists
        self.verify_label_existence()
        node_props = self.text_node_properties or [self.text_node_property]

        fts_parts = [sql.SQL('(to_tsvector(\'english\', {}))').format(sql.Identifier(el)) for el in node_props]
        fts_index_query = """CREATE PROPERTY INDEX IF NOT EXISTS {index_name}
                             ON {node_label} USING {fulltext_index_am} ({expr})"""

        self.query(
            sql.SQL(fts_index_query).format(
                index_name=sql.Identifier(self.keyword_index_name),
                node_label=sql.Identifier(self.node_label),
                fulltext_index_am=sql.SQL(fulltext_index_am.value),
                expr=sql.SQL(", ").join(fts_parts)
            )
        )

    @property
    def embeddings(self) -> Embeddings:
        return self.embedding

    @classmethod
    def __from(
        cls,
        texts: List[str],
        embeddings: List[List[float]],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        create_id_index: bool = True,
        search_type: SearchType = SearchType.VECTOR,
        **kwargs: Any,
    ) -> AgensgraphVector:
        if ids is None:
            ids = [md5(text.encode("utf-8")).hexdigest() for text in texts]

        if not metadatas:
            metadatas = [{} for _ in texts]

        store = cls(
            embedding=embedding,
            search_type=search_type,
            **kwargs,
        )
        # Check if the vector index already exists
        embedding_dimension, index_type = store.retrieve_existing_index()

        # Raise error if relationship index type
        if index_type == "RELATIONSHIP":
            raise ValueError(
                "Data ingestion is not supported with relationship vector index."
            )

        # If the vector index doesn't exist yet
        if not index_type:
            store.create_new_index()
        # If the index already exists, check if embedding dimensions match
        elif (
            embedding_dimension and not store.embedding_dimension == embedding_dimension
        ):
            raise ValueError(
                f"Index with name {store.index_name} already exists."
                "The provided embedding function and vector index "
                "dimensions do not match.\n"
                f"Embedding function dimension: {store.embedding_dimension}\n"
                f"Vector index dimension: {embedding_dimension}"
            )

        if search_type == SearchType.HYBRID:
            fts_node_label = store.retrieve_existing_fts_index()
            # If the FTS index doesn't exist yet
            if not fts_node_label:
                store.create_new_keyword_index()
            else:  # Validate that FTS and Vector index use the same information
                if not fts_node_label == store.node_label:
                    raise ValueError(
                        "Vector and keyword index don't index the same node label"
                    )

        # Create unique constraint for faster import
        if create_id_index:
            query = """CREATE UNIQUE PROPERTY INDEX IF NOT EXISTS {index_name}
                       ON {node_label} ({id_property})"""
            store.query(
                sql.SQL(query).format(
                    index_name=sql.Identifier(f"{store.node_label}_id_index"),
                    node_label=sql.Identifier(store.node_label),
                    id_property=sql.Identifier("id")
            )
        )

        store.add_embeddings(
            texts=texts, embeddings=embeddings, metadatas=metadatas, ids=ids, **kwargs
        )

        return store

    def add_embeddings(
        self,
        texts: Iterable[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add embeddings to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            embeddings: List of list of embedding vectors.
            metadatas: List of metadatas associated with the texts.
            kwargs: vectorstore specific parameters
        """
        if ids is None:
            ids = [md5(text.encode("utf-8")).hexdigest() for text in texts]

        if not metadatas:
            metadatas = [{} for _ in texts]

        parameters = {
            "data": Jsonb([
                {"text": text, "metadata": metadata, "embedding": embedding, "id": id}
                for text, metadata, embedding, id in zip(
                    texts, metadatas, embeddings, ids
                )
            ])
        }

        import_query = sql.SQL(
             """UNWIND %(data)s AS row 
                MERGE (c:{label} {{id: row.id}}) 
                WITH c, row 
                SET c.{embedding_property} = row.embedding 
                SET c.{text_property} = row.text 
                SET c += row.metadata """
        ).format(
            label=sql.Identifier(self.node_label),
            embedding_property=sql.Identifier(self.embedding_node_property),
            text_property=sql.Identifier(self.text_node_property),
        )
        self.query(import_query, params=parameters)

        return ids

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            kwargs: vectorstore specific parameters

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        embeddings = self.embedding.embed_documents(list(texts))
        return self.add_embeddings(
            texts=texts, embeddings=embeddings, metadatas=metadatas, ids=ids, **kwargs
        )

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        params: Dict[str, Any] = {},
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Run similarity search with AgensgraphVector.

        Args:
            query (str): Query text to search for.
            k (int): Number of results to return. Defaults to 4.
            params (Dict[str, Any]): The search params for the index type.
                Defaults to empty dict.
            filter (Optional[Dict[str, Any]]): Dictionary of argument(s) to
                    filter on metadata.
                Defaults to None.

        Returns:
            List of Documents most similar to the query.
        """
        embedding = self.embedding.embed_query(text=query)
        return self.similarity_search_by_vector(
            embedding=embedding,
            k=k,
            query=query,
            params=params,
            filter=filter,
            **kwargs,
        )

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        params: Dict[str, Any] = {},
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            params (Dict[str, Any]): The search params for the index type.
                Defaults to empty dict.
            filter (Optional[Dict[str, Any]]): Dictionary of argument(s) to
                    filter on metadata.
                Defaults to None.

        Returns:
            List of Documents most similar to the query and score for each
        """
        embedding = self.embedding.embed_query(query)
        docs = self.similarity_search_with_score_by_vector(
            embedding=embedding,
            k=k,
            query=query,
            params=params,
            filter=filter,
            **kwargs,
        )
        return docs

    def similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        params: Dict[str, Any] = {},
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """
        Perform a similarity search in the AgensGraph database using a
        given vector and return the top k similar documents with their scores.

        This method uses a Cypher query to find the top k documents that
        are most similar to a given embedding. The similarity is measured
        using a vector index in the AgensGraph database. The results are returned
        as a list of tuples, each containing a Document object and
        its similarity score.

        Args:
            embedding (List[float]): The embedding vector to compare against.
            k (int, optional): The number of top similar documents to retrieve.
            filter (Optional[Dict[str, Any]]): Dictionary of argument(s) to
                    filter on metadata.
                Defaults to None.
            params (Dict[str, Any]): The search params for the index type.
                Defaults to empty dict.

        Returns:
            List[Tuple[Document, float]]: A list of tuples, each containing
                                a Document object and its similarity score.
        """
        filter_params = {}

        if self._index_type == IndexType.RELATIONSHIP:
            base_index_query = (
                 """MATCH ()-[n:{label}]->() 
                    WHERE n.{embedding_property} IS NOT NULL AND 
                    array_size(n.{embedding_property}) = {embedding_dimension} """
            )
        else:
            base_index_query = (
                 """MATCH (n:{label}) 
                    WHERE n.{embedding_property} IS NOT NULL AND 
                    array_size(n.{embedding_property}) = {embedding_dimension} """
            )

        base_cosine_query = """
            WITH n, n.{embedding_property}::vector({embedding_dimension}) <=> %(embedding)s::vector({embedding_dimension}) AS inv_score
            ORDER BY inv_score
            LIMIT %(k)s
            WITH n, 1 - inv_score AS score 
            """

        if filter:
            # Metadata filtering and hybrid doesn't work
            if self.search_type == SearchType.HYBRID:
                raise ValueError(
                    "Metadata filtering can't be use in combination with "
                    "a hybrid search approach"
                )

            filter_snippets, filter_params = construct_metadata_filter(filter)
            index_query = base_index_query + 'AND ' + filter_snippets + base_cosine_query
        else:
            if self.search_type == SearchType.HYBRID:
                index_query = (
                    """
                        UNWIND [1] as a
                        WITH (
                            SELECT jsonb_agg(jsonb_build_object('n', n, 'score', score))
                            FROM (
                                WITH semantic_search AS (
                                    SELECT n, RANK () OVER (ORDER BY inv_score) AS rank
                                    FROM (""" +
                                        base_index_query +
                                     """RETURN properties(n) as n,
                                               n.{embedding_property}::vector({embedding_dimension}) <=> %(embedding)s::vector({embedding_dimension}) AS inv_score
                                        ORDER BY inv_score
                                        LIMIT %(k)s
                                    )t
                                ),
                                keyword_search AS (
                                    SELECT n, RANK () OVER (ORDER BY score DESC) AS rank
                                    FROM (
                                        MATCH (n:{label})
                                        WHERE n.{text_property} IS NOT NULL AND
                                              to_tsvector('english', n.{text_property}) @@ plainto_tsquery('english', %(query)s)
                                        RETURN properties(n) as n, ts_rank_cd(to_tsvector('english', n.{text_property}), plainto_tsquery('english', %(query)s)) AS score
                                        ORDER BY score DESC
                                        LIMIT %(k)s
                                    )t
                                )
                                SELECT
                                    COALESCE(semantic_search.n, keyword_search.n) AS n,
                                    COALESCE(1.0 / (60 + semantic_search.rank), 0.0) +
                                    COALESCE(1.0 / (60 + keyword_search.rank), 0.0) AS score
                                FROM semantic_search
                                FULL OUTER JOIN keyword_search ON semantic_search.n->'id' = keyword_search.n->'id'
                                ORDER BY score DESC
                            )
                        ) AS outputs
                        UNWIND outputs as output
                        WITH output.score AS score,
                             output.n as n
                    """
                )
            else:
                index_query = base_index_query + base_cosine_query

        if self._index_type == IndexType.RELATIONSHIP:
            var = "relationship"
            index_query = index_query + " WITH *, n as relationship "
        else:
            var = "node"
            index_query = index_query + " WITH *, n as node "

        if not self.retrieval_query:
            if kwargs.get("return_embeddings"):
                retrieval_query = (
                    """RETURN {var}.{text_property} AS text, score, 
                    {var} || jsonb_build_object({text_property_literal}, Null, 
                    {embedding_property_literal}, Null, 'id', Null, 
                    '_embedding_', {var}.{embedding_property}) AS metadata"""
                ).replace("{var}", var)
            else:
                retrieval_query = (
                    """RETURN {var}.{text_property} AS text, score, 
                    {var} || jsonb_build_object({text_property_literal}, Null, 
                    {embedding_property_literal}, Null, 'id', Null) AS metadata"""
                ).replace("{var}", var)
            read_query = index_query + retrieval_query
        else:
            retrieval_query = self.retrieval_query

        read_query = index_query + retrieval_query

        parameters = {
            "k": k,
            "embedding": embedding,
            "query": remove_lucene_chars(kwargs["query"]),
            "embedding_property": self.embedding_node_property,
            "text_property": self.text_node_property,
            "text_node_properties": Jsonb(self.text_node_properties),
            **params,
            **filter_params,
        }

        results = self.query(
            sql.SQL(read_query).format(
                label=sql.Identifier(self.node_label),
                embedding_property=sql.Identifier(self.embedding_node_property),
                text_property=sql.Identifier(self.text_node_property),
                embedding_dimension=self.embedding_dimension,
                text_property_literal=sql.Literal(self.text_node_property),
                embedding_property_literal=sql.Literal(self.embedding_node_property),
            ), params=parameters)

        if any(result["text"] is None for result in results):
            if not self.retrieval_query:
                raise ValueError(
                    f"Make sure that none of the `{self.text_node_property}` "
                    f"properties on nodes with label `{self.node_label}` "
                    "are missing or empty"
                )
            else:
                raise ValueError(
                    "Inspect the `retrieval_query` and ensure it doesn't "
                    "return None for the `text` column"
                )
        if kwargs.get("return_embeddings") and any(
            result["metadata"]["_embedding_"] is None for result in results
        ):
            if not self.retrieval_query:
                raise ValueError(
                    f"Make sure that none of the `{self.embedding_node_property}` "
                    f"properties on nodes with label `{self.node_label}` "
                    "are missing or empty"
                )
            else:
                raise ValueError(
                    "Inspect the `retrieval_query` and ensure it doesn't "
                    "return None for the `_embedding_` metadata column"
                )

        docs = [
            (
                Document(
                    page_content=dict_to_yaml_str(result["text"])
                    if isinstance(result["text"], dict)
                    else result["text"],
                    metadata={
                        k: v for k, v in result["metadata"].items() if v is not None
                    },
                ),
                result["score"],
            )
            for result in results
        ]

        return docs

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        params: Dict[str, Any] = {},
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter (Optional[Dict[str, Any]]): Dictionary of argument(s) to
                    filter on metadata.
                Defaults to None.
            params (Dict[str, Any]): The search params for the index type.
                Defaults to empty dict.

        Returns:
            List of Documents most similar to the query vector.
        """
        docs_and_scores = self.similarity_search_with_score_by_vector(
            embedding=embedding, k=k, filter=filter, params=params, **kwargs
        )
        return [doc for doc, _ in docs_and_scores]

    @classmethod
    def from_texts(
        cls: Type[AgensgraphVector],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> AgensgraphVector:
        """
        Return AgensgraphVector initialized from texts and embeddings.
        AgensGraph credentials are required in the form of `url`(postgresql://username:password@host:port/dbname)
        and optional `graph_name` parameters.
        """
        embeddings = embedding.embed_documents(list(texts))

        return cls.__from(
            texts,
            embeddings,
            embedding,
            metadatas=metadatas,
            ids=ids,
            distance_strategy=distance_strategy,
            **kwargs,
        )

    @classmethod
    def from_embeddings(
        cls,
        text_embeddings: List[Tuple[str, List[float]]],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        ids: Optional[List[str]] = None,
        pre_delete_collection: bool = False,
        **kwargs: Any,
    ) -> AgensgraphVector:
        """Construct AgensgraphVector wrapper from raw documents and pre-
        generated embeddings.

        Return AgensgraphVector initialized from documents and embeddings.
        AgensGraph credentials are required in the form of `url`(postgresql://username:password@host:port/dbname)
        and optional `graph_name` parameters.

        Example:
            .. code-block:: python

                from langchain_agensgraph.vectorstores.agensgraph_vector import AgensgraphVector
                from langchain_community.embeddings import OpenAIEmbeddings
                embeddings = OpenAIEmbeddings()
                text_embeddings = embeddings.embed_documents(texts)
                text_embedding_pairs = list(zip(texts, text_embeddings))
                vectorstore = AgensgraphVector.from_embeddings(
                    text_embedding_pairs, embeddings)
        """
        texts = [t[0] for t in text_embeddings]
        embeddings = [t[1] for t in text_embeddings]

        return cls.__from(
            texts,
            embeddings,
            embedding,
            metadatas=metadatas,
            ids=ids,
            distance_strategy=distance_strategy,
            pre_delete_collection=pre_delete_collection,
            **kwargs,
        )

    @classmethod
    def from_existing_index(
        cls: Type[AgensgraphVector],
        embedding: Embeddings,
        index_name: str,
        search_type: SearchType = DEFAULT_SEARCH_TYPE,
        keyword_index_name: Optional[str] = None,
        **kwargs: Any,
    ) -> AgensgraphVector:
        """
        Get instance of an existing AgensGraph vector index. This method will
        return the instance of the store without inserting any new
        embeddings.
        AgensGraph credentials are required in the form of `url`(postgresql://username:password@host:port/dbname)
        and optional `graph_name` parameters along with
        the `index_name` definition.
        """

        if search_type == SearchType.HYBRID and not keyword_index_name:
            raise ValueError(
                "keyword_index name has to be specified when using hybrid search option"
            )

        store = cls(
            embedding=embedding,
            index_name=index_name,
            keyword_index_name=keyword_index_name,
            search_type=search_type,
            **kwargs,
        )

        embedding_dimension, index_type = store.retrieve_existing_index()

        # Raise error if relationship index type
        if index_type == "RELATIONSHIP":
            raise ValueError(
                "Relationship vector index is not supported with "
                "`from_existing_index` method. Please use the "
                "`from_existing_relationship_index` method."
            )

        if not index_type:
            raise ValueError(
                "The specified vector index name does not exist. "
                "Make sure to check if you spelled it correctly"
            )

        # Check if embedding function and vector index dimensions match
        if embedding_dimension and not store.embedding_dimension == embedding_dimension:
            raise ValueError(
                "The provided embedding function and vector index "
                "dimensions do not match.\n"
                f"Embedding function dimension: {store.embedding_dimension}\n"
                f"Vector index dimension: {embedding_dimension}"
            )

        if search_type == SearchType.HYBRID:
            fts_node_label = store.retrieve_existing_fts_index()
            # If the FTS index doesn't exist yet
            if not fts_node_label:
                raise ValueError(
                    "The specified keyword index name does not exist. "
                    "Make sure to check if you spelled it correctly"
                )
            else:  # Validate that FTS and Vector index use the same information
                if not fts_node_label == store.node_label:
                    raise ValueError(
                        "Vector and keyword index don't index the same node label"
                    )

        return store

    @classmethod
    def from_existing_relationship_index(
        cls: Type[AgensgraphVector],
        embedding: Embeddings,
        index_name: str,
        search_type: SearchType = DEFAULT_SEARCH_TYPE,
        **kwargs: Any,
    ) -> AgensgraphVector:
        """
        Get instance of an existing AgensGraph relationship vector index.
        This method will return the instance of the store without
        inserting any new embeddings.
        AgensGraph credentials are required in the form of `url`(postgresql://username:password@host:port/dbname)
        and optional `graph_name` parameters along with
        the `index_name` definition.
        """

        if search_type == SearchType.HYBRID:
            raise ValueError(
                "Hybrid search is not supported in combination "
                "with relationship vector index"
            )

        store = cls(
            embedding=embedding,
            index_name=index_name,
            **kwargs,
        )

        embedding_dimension, index_type = store.retrieve_existing_index()

        if not index_type:
            raise ValueError(
                "The specified vector index name does not exist. "
                "Make sure to check if you spelled it correctly"
            )
        # Raise error if relationship index type
        if index_type == "NODE":
            raise ValueError(
                "Node vector index is not supported with "
                "`from_existing_relationship_index` method. Please use the "
                "`from_existing_index` method."
            )

        # Check if embedding function and vector index dimensions match
        if embedding_dimension and not store.embedding_dimension == embedding_dimension:
            raise ValueError(
                "The provided embedding function and vector index "
                "dimensions do not match.\n"
                f"Embedding function dimension: {store.embedding_dimension}\n"
                f"Vector index dimension: {embedding_dimension}"
            )

        return store

    @classmethod
    def from_documents(
        cls: Type[AgensgraphVector],
        documents: List[Document],
        embedding: Embeddings,
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> AgensgraphVector:
        """
        Return AgensgraphVector initialized from documents and embeddings.
        AgensGraph credentials are required in the form of `url`(postgresql://username:password@host:port/dbname)
        and optional `graph_name` parameters.
        """

        texts = [d.page_content for d in documents]
        metadatas = [d.metadata for d in documents]

        return cls.from_texts(
            texts=texts,
            embedding=embedding,
            distance_strategy=distance_strategy,
            metadatas=metadatas,
            ids=ids,
            **kwargs,
        )

    @classmethod
    def from_existing_graph(
        cls: Type[AgensgraphVector],
        embedding: Embeddings,
        node_label: str,
        embedding_node_property: str,
        text_node_properties: List[str],
        *,
        keyword_index_name: Optional[str] = "keyword",
        index_name: str = "vector",
        search_type: SearchType = DEFAULT_SEARCH_TYPE,
        retrieval_query: str = "",
        **kwargs: Any,
    ) -> AgensgraphVector:
        """
        Initialize and return a AgensgraphVector instance from an existing graph.

        This method initializes a AgensgraphVector instance using the provided
        parameters and the existing graph. It validates the existence of
        the indices and creates new ones if they don't exist.

        Returns:
        AgensgraphVector: An instance of AgensgraphVector initialized with the provided parameters
                    and existing graph.

        Example:
        >>> agensgraph_vector = AgensgraphVector.from_existing_graph(
        ...     embedding=my_embedding,
        ...     node_label="Document",
        ...     embedding_node_property="embedding",
        ...     text_node_properties=["title", "content"]
        ... )

        Note:
        AgensGraph credentials are required in the form of `url`(postgresql://username:password@host:port/dbname)
        and optional `graph_name` parameters.
        """
        # Validate the list is not empty
        if not text_node_properties:
            raise ValueError(
                "Parameter `text_node_properties` must not be an empty list"
            )
        # Prefer retrieval query from params, otherwise construct it
        if not retrieval_query:
            retrieval_query = (
                """RETURN (
                          SELECT string_agg(E'\\n' || k || ': ' || coalesce(n->>k, ''), '')
                          FROM jsonb_array_elements_text(%(text_node_properties)s) AS k
                        ) AS text,
                        n || jsonb_build_object({embedding_property_literal}, NULL, 'id', Null,"""
                     + ",".join([f"'{prop}', Null" for prop in text_node_properties]) + """) AS metadata, score
                """
            )
        store = cls(
            embedding=embedding,
            index_name=index_name,
            keyword_index_name=keyword_index_name,
            search_type=search_type,
            retrieval_query=retrieval_query,
            node_label=node_label,
            embedding_node_property=embedding_node_property,
            text_node_properties= text_node_properties,
            **kwargs,
        )

        # Check if the vector index already exists
        embedding_dimension, index_type = store.retrieve_existing_index()

        # Raise error if relationship index type
        if index_type == "RELATIONSHIP":
            raise ValueError(
                "`from_existing_graph` method does not support "
                " existing relationship vector index. "
                "Please use `from_existing_relationship_index` method"
            )

        # If the vector index doesn't exist yet
        if not index_type:
            store.create_new_index()
        # If the index already exists, check if embedding dimensions match
        elif (
            embedding_dimension and not store.embedding_dimension == embedding_dimension
        ):
            raise ValueError(
                f"Index with name {store.index_name} already exists."
                "The provided embedding function and vector index "
                "dimensions do not match.\n"
                f"Embedding function dimension: {store.embedding_dimension}\n"
                f"Vector index dimension: {embedding_dimension}"
            )
        # FTS index for Hybrid search
        if search_type == SearchType.HYBRID:
            fts_node_label = store.retrieve_existing_fts_index()
            # If the FTS index doesn't exist yet
            if not fts_node_label:
                store.create_new_keyword_index()
            else:  # Validate that FTS and Vector index use the same information
                if not fts_node_label == store.node_label:
                    raise ValueError(
                        "Vector and keyword index don't index the same node label"
                    )

        # Populate embeddings
        while True:
            fetch_query = (
                 """MATCH (n:{label}) 
                    WHERE n.{embedding_property} IS null 
                    AND any(k IN %(text_node_properties)s WHERE n[k] IS NOT NULL) 
                    RETURN toString(id(n)) AS id, (
                                SELECT string_agg(E'\\n' || k || ': ' || coalesce(n->>k, ''), '')
                                FROM jsonb_array_elements_text(%(text_node_properties)s) AS k
                           ) AS text
                    LIMIT 1000
                """
            )
            data = store.query(sql.SQL(fetch_query).format(
                label=sql.Identifier(store.node_label),
                embedding_property=sql.Identifier(store.embedding_node_property)
            ), params={"text_node_properties": Jsonb(text_node_properties)})

            if not data:
                break
            text_embeddings = embedding.embed_documents([el["text"] for el in data])
            rows = [
                    {"id": el["id"], "embedding": embedding}
                    for el, embedding in zip(data, text_embeddings)
                ]
            params = {
                "data": Jsonb(rows)
            }

            store.query(sql.SQL(
                 """UNWIND %(data)s AS row 
                    MATCH (n:{node_label}) 
                    WHERE toString(Id(n)) = row.id 
                    SET n.{embedding_node_property} = row.embedding 
                    RETURN count(*)"""
                ).format(
                    node_label=sql.Identifier(store.node_label),
                    embedding_node_property=sql.Identifier(store.embedding_node_property)
                ),
                params=params,
            )
            # If embedding calculation should be stopped
            if len(data) < 1000:
                break
        return store

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            query: search query text.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
            filter: Filter on metadata properties, e.g.
                            {
                                "str_property": "foo",
                                "int_property": 123
                            }
        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        # Embed the query
        query_embedding = self.embedding.embed_query(query)

        # Fetch the initial documents
        got_docs = self.similarity_search_with_score_by_vector(
            embedding=query_embedding,
            query=query,
            k=fetch_k,
            return_embeddings=True,
            filter=filter,
            **kwargs,
        )

        # Get the embeddings for the fetched documents
        got_embeddings = [doc.metadata["_embedding_"] for doc, _ in got_docs]

        # Select documents using maximal marginal relevance
        selected_indices = maximal_marginal_relevance(
            np.array(query_embedding), got_embeddings, lambda_mult=lambda_mult, k=k
        )
        selected_docs = [got_docs[i][0] for i in selected_indices]

        # Remove embedding values from metadata
        for doc in selected_docs:
            del doc.metadata["_embedding_"]

        return selected_docs

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        """
        The 'correct' relevance function
        may differ depending on a few things, including:
        - the distance / similarity metric used by the VectorStore
        - the scale of your embeddings (OpenAI's are unit normed. Many others are not!)
        - embedding dimensionality
        - etc.
        """
        if self.override_relevance_score_fn is not None:
            return self.override_relevance_score_fn

        # Default strategy is to rely on distance strategy provided
        # in vectorstore constructor
        if self._distance_strategy == DistanceStrategy.COSINE:
            return lambda x: x
        elif self._distance_strategy == DistanceStrategy.EUCLIDEAN_DISTANCE:
            return lambda x: x
        else:
            raise ValueError(
                "No supported normalization function"
                f" for distance_strategy of {self._distance_strategy}."
                "Consider providing relevance_score_fn to PGVector constructor."
            )

    def _get_cursor(self) -> psycopg.Cursor:
        cursor = self.connection.cursor(row_factory=psycopg.rows.namedtuple_row)
        return cursor

    def verify_vector_support(self) -> None:
        """
        Verify if the graph store supports vector operations
        """
        with self._get_cursor() as curs:
            try:
                curs.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                self.connection.commit()
            except psycopg.Error:
                self.connection.rollback()
                raise ValueError(
                    """Vector extension not supported\nUnable to install pg_vector extension"""
                )

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
                vertex = AgensgraphVector.vertex_regex.match(v)
                if vertex:
                    label, vertex_id, properties = vertex.groups()
                    properties = json.loads(properties)
                    vertices[str(vertex_id)] = properties

        # iterate returned fields and parse appropriately
        for k in record._fields:
            v = getattr(record, k)

            if isinstance(v, str):
                vertex = AgensgraphVector.vertex_regex.match(v)
                edge = AgensgraphVector.edge_regex.match(v)

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
                curs.execute(query, params)
                self.connection.commit()
            except psycopg.Error as e:
                self.connection.rollback()
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

            return result