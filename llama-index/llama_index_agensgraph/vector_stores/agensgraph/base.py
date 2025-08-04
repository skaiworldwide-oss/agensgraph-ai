from typing import Any, Dict, List, Optional, Tuple, NamedTuple, Pattern, Union
import logging

import json, re

import psycopg
from psycopg import sql
from psycopg.types.json import Jsonb

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.schema import BaseNode, MetadataMode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
    FilterOperator,
    MetadataFilters,
    MetadataFilter,
    FilterCondition,
)
from llama_index.core.vector_stores.utils import (
    metadata_dict_to_node,
    node_to_metadata_dict,
)

_logger = logging.getLogger(__name__)

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

def check_if_not_null(props: List[str], values: List[Any]) -> None:
    """Check if variable is not null and raise error accordingly."""
    for prop, value in zip(props, values):
        if not value:
            raise ValueError(f"Parameter `{prop}` must not be None or empty string")


def sort_by_index_name(
    lst: List[Dict[str, Any]], index_name: str
) -> List[Dict[str, Any]]:
    """Sort first element to match the index_name if exists."""
    return sorted(lst, key=lambda x: x.get("name") != index_name)


def clean_params(params: List[BaseNode]) -> List[Dict[str, Any]]:
    """Convert BaseNode object to a dictionary to be imported into Agensgraph."""
    clean_params = []
    for record in params:
        text = record.get_content(metadata_mode=MetadataMode.NONE)
        embedding = record.get_embedding()
        id = record.node_id
        metadata = node_to_metadata_dict(record, remove_text=True, flat_metadata=False)
        # Remove redundant metadata information
        for k in ["document_id", "doc_id"]:
            del metadata[k]
        clean_params.append(
            {"text": text, "embedding": embedding, "id": id, "metadata": metadata}
        )
    return clean_params


def remove_lucene_chars(text: Optional[str]) -> Optional[str]:
    """Remove Lucene special characters."""
    if not text:
        return None
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


def _to_agensgraph_operator(operator: FilterOperator) -> str:
    if operator == FilterOperator.EQ:
        return "="
    elif operator == FilterOperator.GT:
        return ">"
    elif operator == FilterOperator.LT:
        return "<"
    elif operator == FilterOperator.NE:
        return "<>"
    elif operator == FilterOperator.GTE:
        return ">="
    elif operator == FilterOperator.LTE:
        return "<="
    elif operator == FilterOperator.IN:
        return "IN"
    elif operator == FilterOperator.NIN:
        return "NOT IN"
    elif operator == FilterOperator.CONTAINS:
        return "CONTAINS"
    else:
        _logger.warning(f"Unknown operator: {operator}, fallback to '='")
        return "="


def collect_params(
    input_data: List[Tuple[str, Dict[str, str]]],
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Transform the input data into the desired format.

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


def filter_to_cypher(index: int, filter: MetadataFilter) -> str:
    return (
        f'n."{filter.key}" {_to_agensgraph_operator(filter.operator)} %(param_{index})s',
        {f"param_{index}": Jsonb(filter.value)},
    )


def construct_metadata_filter(filters: MetadataFilters):
    cypher_snippets = []
    for index, filter in enumerate(filters.filters):
        cypher_snippets.append(filter_to_cypher(index, filter))

    collected_snippets = collect_params(cypher_snippets)

    if filters.condition == FilterCondition.OR:
        return (" OR ".join(collected_snippets[0]), collected_snippets[1])
    else:
        return (" AND ".join(collected_snippets[0]), collected_snippets[1])

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

_vertex_regex: Pattern = re.compile(r"(\w+)\[(\d+\.\d+)\](\{.*\})")
_edge_regex: Pattern = re.compile(r"(\w+)\[(\d+\.\d+)\]\[(\d+\.\d+),\s*(\d+\.\d+)\](\{.*\})")

class AgensgraphVectorStore(BasePydanticVectorStore):
    """
    Agensgraph Vector Store.

    Examples:
        # `pip install TODO`


        ```python
        from llama_index_agensgraph.vector_stores.agensgraph import AgensgraphVectorStore

        url = "postgresql://username:password@localhost:5432/dbname"
        embed_dim = 1536

        agensgraph_vector = AgensgraphVectorStore(url, embed_dim)
        ```

    """

    stores_text: bool = True
    flat_metadata: bool = True

    distance_strategy: str
    index_name: str
    keyword_index_name: str
    hybrid_search: bool
    node_label: str
    embedding_node_property: str
    text_node_property: str
    retrieval_query: str
    embedding_dimension: int

    _graph_name: Optional[str] = "vector_store"
    _support_metadata_filter: bool = PrivateAttr()

    def __init__(
        self,
        url: str,
        embedding_dimension: int,
        graph_name: Optional[str] = "vector_store",
        index_name: str = "vector",
        keyword_index_name: str = "keyword",
        node_label: str = "Chunk",
        embedding_node_property: str = "embedding",
        text_node_property: str = "text",
        distance_strategy: str = "cosine",
        hybrid_search: bool = False,
        retrieval_query: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            distance_strategy=distance_strategy,
            index_name=index_name,
            keyword_index_name=keyword_index_name,
            hybrid_search=hybrid_search,
            node_label=node_label,
            embedding_node_property=embedding_node_property,
            text_node_property=text_node_property,
            retrieval_query=retrieval_query,
            embedding_dimension=embedding_dimension,
        )

        if distance_strategy not in ["cosine"]:
            raise ValueError("Only cosine distance strategy is supported for now")

        self._graph_name = graph_name

        # Verify connection
        try:
            self._connection = psycopg.connect(url)
        except psycopg.OperationalError as e:
            raise ValueError(f"Failed to connect to Agensgraph database: {e}")

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

        # Create the graph and utility functions
        self.database_query(get_vector_index_info_function)
        self.database_query(get_keyword_index_info_function)
        self.database_query(sql.SQL("CREATE GRAPH IF NOT EXISTS {}").format(
            sql.Identifier(self._graph_name)
        ))
        self.database_query(sql.SQL("SET graph_path = {}").format(
            sql.Literal(self._graph_name)
        ))

        self.verify_vector_support()

        index_already_exists = self.retrieve_existing_index()
        if not index_already_exists:
            self.create_new_index()
        if hybrid_search:
            fts_node_label = self.retrieve_existing_fts_index()
            # If the FTS index doesn't exist yet
            if not fts_node_label:
                self.create_new_keyword_index()
            else:  # Validate that FTS and Vector index use the same information
                if not fts_node_label == self.node_label:
                    raise ValueError(
                        "Vector and keyword index don't index the same node label"
                    )

    def verify_label_existence(self) -> None:
        """Create label if it does not exist."""
        self.database_query(
            sql.SQL("CREATE VLABEL IF NOT EXISTS {}").format(
                sql.Identifier(self.node_label)
            )
        )

    @property
    def client(self) -> psycopg.Connection:
        return self._connection

    def create_new_index(self) -> None:
        """
        This method constructs a Cypher query and executes it
        to create a new vector index in agensgraph.
        """
        self.verify_label_existence()
        index_query = """CREATE PROPERTY INDEX IF NOT EXISTS {index_name}
            ON {node_label} USING hnsw
            (({embedding_node_property}::vector({embedding_dimension})) vector_cosine_ops)"""

        self.database_query(
            sql.SQL(index_query).format(
                index_name=sql.Identifier(self.index_name),
                node_label=sql.Identifier(self.node_label),
                embedding_node_property=sql.Identifier(self.embedding_node_property),
                embedding_dimension=self.embedding_dimension
            )
        )

    def retrieve_existing_index(self) -> bool:
        """
        Check if the vector index exists in the Agensgraph database
        and returns its embedding dimension.

        This method queries the Agensgraph database for existing indexes
        and attempts to retrieve the dimension of the vector index
        with the specified name. If the index exists, its dimension is returned.
        If the index doesn't exist, `None` is returned.

        Returns:
            int or None: The embedding dimension of the existing index if found.

        """
        index_information = self.database_query(
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
            self.index_name = index_information[0]["name"]
            self.node_label = index_information[0]["labelortype"]
            self.embedding_node_property = index_information[0]["property"]
            self.embedding_dimension = index_information[0]["dimensions"]

            return True
        except IndexError:
            return False

    def retrieve_existing_fts_index(self) -> Optional[str]:
        """
        Check if the fulltext index exists in the Agensgraph database.

        This method queries the Agensgraph database for existing fts indexes
        with the specified name.

        Returns:
            (Tuple): keyword index information

        """
        index_information = self.database_query(
            """SELECT * FROM ag_list_text_indexes(index_name => %(index_name)s,
                                                  node_label => %(node_label)s,
                                                  text_node_properties => %(text_node_properties)s)
            """,
            params={
                "index_name": self.keyword_index_name,
                "node_label": self.node_label,
                "text_node_properties": [self.text_node_property],
            },
        )
        # sort by index_name
        index_information = sort_by_index_name(index_information, self.index_name)
        try:
            self.keyword_index_name = index_information[0]["name"]
            self.text_node_property = index_information[0]["properties"][0]
            node_label = index_information[0]["labelortype"]
            return node_label
        except IndexError:
            return None

    def create_new_keyword_index(self, text_node_properties: List[str] = []) -> None:
        """
        This method constructs a Cypher query and executes it
        to create a new full text index in Agensgraph.
        """
        # make sure label exists
        self.verify_label_existence()
        node_props = text_node_properties or [self.text_node_property]

        fts_parts = [sql.SQL('(to_tsvector(\'english\', {}))').format(sql.Identifier(el)) for el in node_props]
        fts_index_query = """CREATE PROPERTY INDEX IF NOT EXISTS {index_name}
                             ON {node_label} USING gin ({expr})"""

        self.database_query(
            sql.SQL(fts_index_query).format(
                index_name=sql.Identifier(self.keyword_index_name),
                node_label=sql.Identifier(self.node_label),
                expr=sql.SQL(", ").join(fts_parts)
            )
        )

    def add(self, nodes: List[BaseNode], **add_kwargs: Any) -> List[str]:
        ids = [r.node_id for r in nodes]
        import_query = """
            UNWIND %(data)s AS row 
            MERGE (c:{label} {{id: row.id}}) 
            WITH c, row 
            SET c.{embedding_node_property} = row.embedding,
                c.{text_node_property} = row.text 
            SET c += row.metadata
        """

        self.database_query(
            sql.SQL(import_query).format(
                label=sql.Identifier(self.node_label),
                embedding_node_property=sql.Identifier(self.embedding_node_property),
                text_node_property=sql.Identifier(self.text_node_property),
            ),
            params={"data": Jsonb(clean_params(nodes))},
        )

        return ids

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
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

        if query.filters:
            # Metadata filtering and hybrid doesn't work
            if self.hybrid_search:
                raise ValueError(
                    "Metadata filtering can't be use in combination with "
                    "a hybrid search approach"
                )

            filter_snippets, filter_params = construct_metadata_filter(query.filters)
            index_query = base_index_query + 'AND ' + filter_snippets + base_cosine_query
        else:
            filter_params = {}
            if self.hybrid_search:
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

        index_query = index_query + " WITH *, n as node "
        default_retrieval = """
            RETURN node.{text_property} AS text, score, 
            node.id AS id, 
            node || jsonb_build_object({text_property_literal}, Null, 
            {embedding_property_literal}, Null, 'id', Null) AS metadata
        """

        index_query += self.retrieval_query or default_retrieval


        parameters = {
            "k": query.similarity_top_k,
            "embedding": query.query_embedding,
            "query": remove_lucene_chars(query.query_str),
            **filter_params,
        }

        results = self.database_query(sql.SQL(index_query).format(
            label=sql.Identifier(self.node_label),
            embedding_property=sql.Identifier(self.embedding_node_property),
            text_property=sql.Identifier(self.text_node_property),
            embedding_dimension=self.embedding_dimension,
            text_property_literal=sql.Literal(self.text_node_property),
            embedding_property_literal=sql.Literal(self.embedding_node_property),
        ), params=parameters)

        nodes = []
        similarities = []
        ids = []
        for record in results:
            node = metadata_dict_to_node(record["metadata"])
            node.set_content(str(record["text"]))
            nodes.append(node)
            similarities.append(record["score"])
            ids.append(record["id"])

        return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        query = """
            MATCH (n:{label})
            WHERE n.ref_doc_id = %(id)s
            DETACH DELETE n
            """
        self.database_query(
            sql.SQL(query).format(
                label=sql.Identifier(self.node_label)
            ),
            params={"id": Jsonb(ref_doc_id)},
        )

    def _get_cursor(self) -> psycopg.Cursor:
        cursor = self._connection.cursor(row_factory=psycopg.rows.namedtuple_row)
        return cursor

    def verify_vector_support(self) -> None:
        """
        Verify if the graph store supports vector operations
        """
        with self._get_cursor() as curs:
            try:
                curs.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                self._connection.commit()
            except psycopg.Error:
                self._connection.rollback()
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
                vertex = _vertex_regex.match(v)
                if vertex:
                    label, vertex_id, properties = vertex.groups()
                    properties = json.loads(properties)
                    vertices[str(vertex_id)] = properties

        # iterate returned fields and parse appropriately
        for k in record._fields:
            v = getattr(record, k)

            if isinstance(v, str):
                vertex = _vertex_regex.match(v)
                edge = _edge_regex.match(v)

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

    def database_query(self, query: str, params: dict = {}) -> List[Dict[str, Any]]:
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
                self._connection.commit()
            except psycopg.Error as e:
                self._connection.rollback()
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