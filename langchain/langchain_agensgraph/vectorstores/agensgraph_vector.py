from __future__ import annotations

import dataclasses
import enum
import logging
import re
from contextlib import asynccontextmanager, contextmanager
from hashlib import md5
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
)

import agensgraph
import numpy as np
from agensgraph import Vector
from agensgraph.introspect import DesiredIndex
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.utils import get_from_dict_or_env
from langchain_core.vectorstores import VectorStore
from psycopg import sql
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from langchain_agensgraph.graphs.agensgraph import (
    AgensGraph,
    AgensQueryException,
    _package_version,
    checked_names,
)

if TYPE_CHECKING:
    from langchain_agensgraph.engine import AgensEngine
from langchain_agensgraph.vectorstores.utils import (
    DistanceStrategy,
    distance_of,
    maximal_marginal_relevance,
)

DEFAULT_DISTANCE_STRATEGY = DistanceStrategy.COSINE

VECTOR_TYPES = ("vector", "halfvec", "sparsevec", "bit")
"""What an embedding may be stored as.

``vector`` is four bytes a dimension and the default. ``halfvec`` is two, which halves
both the table and the index for a small loss of precision. ``sparsevec`` stores only
the entries that are not zero. ``bit`` is one bit a dimension, for an embedding that has
been binary-quantized, and is the only one the Hamming and Jaccard distances apply to.
"""

BIT_ONLY = (DistanceStrategy.HAMMING, DistanceStrategy.JACCARD)
"""The two distances that measure bit strings rather than vectors."""

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
    IVFFLAT = "ivfflat"

class FullTextIndexAM(str, enum.Enum):
    """Enumerator of the full text index access methods."""

    GIN = "GIN"
    GIST = "GIST"

DEFAULT_INDEX_TYPE = IndexType.NODE
DEFAULT_VECTOR_INDEX_AM = VectorIndexAM.HNSW


@dataclasses.dataclass
class IndexConfig:
    """Typed configuration for a pgvector index.

    Replaces passing a bare ``VectorIndexAM`` enum, letting callers tune the
    access-method build parameters that materially affect recall/latency:

    * HNSW: ``m`` (max connections per layer) and ``ef_construction``
      (candidate list size at build time).
    * IVFFlat: ``lists`` (number of inverted lists / centroids).

    Example::

        store.create_new_index(
            IndexConfig(am=VectorIndexAM.HNSW, m=16, ef_construction=64)
        )
    """

    am: VectorIndexAM = DEFAULT_VECTOR_INDEX_AM
    m: Optional[int] = None
    ef_construction: Optional[int] = None
    lists: Optional[int] = None

    def with_options_clause(self) -> str:
        """Return the trailing ``WITH (...)`` clause, or '' when no options.

        An option belongs to one access method: ``m`` and ``ef_construction`` describe how
        an HNSW graph is built, ``lists`` how many an IVFFlat index divides its vectors
        into. Naming one that does not belong to the method asked for is refused rather
        than dropped -- silently ignoring it builds an index with the defaults while the
        caller believes they tuned it, and the difference only shows up as recall they
        cannot explain.
        """
        opts: List[str] = []
        if self.am == VectorIndexAM.HNSW:
            if self.lists is not None:
                raise ValueError(
                    "`lists` describes an IVFFlat index, and this one is HNSW. "
                    "HNSW takes `m` and `ef_construction`."
                )
            if self.m is not None:
                opts.append(f"m = {int(self.m)}")
            if self.ef_construction is not None:
                opts.append(f"ef_construction = {int(self.ef_construction)}")
        elif self.am == VectorIndexAM.IVFFLAT:
            wrong = [
                name
                for name, value in (("m", self.m), ("ef_construction", self.ef_construction))
                if value is not None
            ]
            if wrong:
                raise ValueError(
                    f"{' and '.join(wrong)} describe an HNSW index, and this one is "
                    "IVFFlat. IVFFlat takes `lists`."
                )
            if self.lists is not None:
                opts.append(f"lists = {int(self.lists)}")
        return f" WITH ({', '.join(opts)})" if opts else ""


@dataclasses.dataclass
class HybridSearchConfig:
    """Configuration for hybrid (vector + keyword) fusion.

    Currently supports reciprocal rank fusion (RRF), the de-facto default. The
    ``rank_constant`` k tempers the influence of lower-ranked hits; smaller k sharpens
    the contribution of top ranks. ``vector_weight`` / ``keyword_weight`` scale each
    modality's contribution.
    """

    rank_constant: int = 60
    vector_weight: float = 1.0
    keyword_weight: float = 1.0


DEFAULT_HYBRID_CONFIG = HybridSearchConfig()
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

def like_to_regex(pattern: str, *, ignore_case: bool = False) -> str:
    """A SQL ``LIKE`` pattern as the regular expression Cypher's ``=~`` takes.

    ``%`` matches any run of characters and ``_`` any single one; everything else is
    literal, so every regular-expression metacharacter in the pattern is escaped. The
    result is anchored, because ``LIKE`` matches the whole value and ``=~`` would
    otherwise match anywhere inside it.
    """
    out = ["(?i)" if ignore_case else "", "^"]
    for char in pattern:
        if char == "%":
            out.append(".*")
        elif char == "_":
            out.append(".")
        else:
            out.append(re.escape(char))
    out.append("$")
    return "".join(out)


def _handle_field_filter(
    field: str, value: Any, param_number: int = 1
) -> Tuple[str, Dict]:
    """Create a filter for a specific field.

    Args:
        field: name of field value: value to filter
            If provided as is then this will be an equality filter If provided as a
            dictionary then this will be a filter, the key will be the operator and the
            value will be the value to filter by
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

    # A scalar goes as itself. Wrapped as jsonb it is compared as jsonb, and a property
    # given a column of its own holds text -- so the comparison would no longer match the
    # index on that column and the search would read the label instead. A list or a map is
    # still jsonb, because that is what it is being compared to.
    if isinstance(filter_value, (list, dict)) and operator not in {
        "$in",
        "$nin",
        "$like",
        "$ilike",
        "$between",
    }:
        filter_value = Jsonb(filter_value)

    if operator == "$ne":
        # A property that is absent is unequal to anything, so it satisfies the test --
        # the same answer `$nin` gives for the same element. A comparison against null is
        # null rather than true, so the absence is named.
        query_snippet = (
            f'(n."{field}" IS NULL OR n."{field}" <> %(param_{param_number})s)'
        )
        return (query_snippet, {f"param_{param_number}": filter_value})
    if operator in COMPARISONS_TO_NATIVE:
        # Then we implement an equality filter native is trusted input
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
        if operator == "$in":
            # One equality per value rather than `<@` over a list. Containment is not a
            # comparison an index answers, so a bound list costs a read of the whole
            # label; equality is what the index is on, so a disjunction of them reaches
            # it.
            values = list(filter_value)
            terms = " OR ".join(
                f'n."{field}" = %(param_{param_number}_in{i})s'
                for i in range(len(values))
            )
            params = {
                f"param_{param_number}_in{i}": val for i, val in enumerate(values)
            }
            return (f"({terms})" if values else "false", params)
        if operator == "$nin":
            # One inequality per value, for the reason `$in` is one equality per value.
            # A missing property satisfies "not in", and a comparison against null is
            # null rather than true, so it is named separately.
            values = list(filter_value)
            terms = " AND ".join(
                f'n."{field}" <> %(param_{param_number}_nin{i})s'
                for i in range(len(values))
            )
            params = {
                f"param_{param_number}_nin{i}": val for i, val in enumerate(values)
            }
            if not values:
                return ("true", {})
            return (f'(n."{field}" IS NULL OR ({terms}))', params)
        # `$like`/`$ilike` take a SQL LIKE pattern, `%` and `_` included. Cypher has no
        # LIKE, so the pattern becomes the equivalent regular expression and is matched
        # with `=~`.
        if operator in {"$like", "$ilike"}:
            pattern = like_to_regex(filter_value, ignore_case=operator == "$ilike")
            query_snippet = f'n."{field}" =~ %(param_{param_number})s'
            return (query_snippet, {f"param_{param_number}": pattern})
        raise NotImplementedError(f"Unsupported operator: {operator}")
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
            # The only operators allowed at the top level are $AND and $OR First check
            # if an operator or a field
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
    raise ValueError(
        f"Expected a dictionary of filter conditions, got {type(filter).__name__}."
    )
        
class AgensgraphVector(VectorStore):
    """`AgensGraph` vector index.

    To use, you should have the ``psycopg`` python package installed.

    Args:
        url: AgensGraph connection url graph_name: The name of the graph to use in
        Agensgraph. Defaults to "vector_store". embedding: Any embedding function
        implementing
            `langchain.embeddings.base.Embeddings` interface.
        distance_strategy: The distance strategy to use. (default: COSINE) search_type:
        The type of search to be performed, either
            'vector' or 'hybrid'
        node_label: The label used for nodes in the AgensGraph database.
            (default: "Chunk")
        embedding_node_property: The property name in AgensGraph to store
        embeddings(powered by pgvector).
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

            from langchain_agensgraph.vectorstores.agensgraph_vector import
            AgensgraphVector from langchain_openai import OpenAIEmbeddings

            url="postgresql://username:password@host:port/dbname" graph_name="my_graph"
            embeddings = OpenAIEmbeddings() vectorestore =
            AgensgraphVector.from_documents(
                embedding=embeddings, documents=docs, url=url
            )


    """

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
        engine: Optional["AgensEngine"] = None,
        vector_type: str = "vector",
        filter_properties: Optional[List[str]] = None,
        search_options: Optional[Dict[str, Any]] = None,
    ) -> None:
        # Every strategy the driver has a distance for is available. The operator and
        # the operator class both come from that one place, which is what keeps them in
        # step: an index built for cosine answers `<=>` alone, and a search ordering by
        # anything else against it sorts a sequential scan instead of using it.
        self._distance = distance_of(distance_strategy)
        if vector_type not in VECTOR_TYPES:
            raise ValueError(
                f"vector_type is one of {VECTOR_TYPES}, not {vector_type!r}"
            )
        # Hamming and Jaccard measure bit strings. Refused against any other storage
        # type here rather than at the server, where it arrives as a missing operator.
        wants_bits = distance_strategy in BIT_ONLY
        if wants_bits and vector_type != "bit":
            raise ValueError(
                f"{distance_strategy} measures bit strings, so it needs "
                f'vector_type="bit", not {vector_type!r}.'
            )
        if vector_type == "bit" and not wants_bits:
            raise ValueError(
                f'vector_type="bit" is measured only by '
                f"{' or '.join(s.value for s in BIT_ONLY)}, not {distance_strategy}."
            )
        self._vector_type = vector_type

        # Resolution order: explicit graph object > engine > url/env.
        if graph:
            self.connection = graph.connection
            self.graph_name = graph.graph_name
            self._graph = graph  # share its async connection too
            self._engine = graph._engine
            self._url = None
            self._owns_connection = False
        elif engine is not None:
            # Dedicated connection for setup/introspection; the pool backs the
            # concurrent query path. Open WITHOUT graph_path and create the graph on
            # this connection first: a pooled "SET graph_path" on a not-yet- existing
            # graph errors, so the graph must exist before any pooled checkout (the url
            # path likewise creates it before binding the path).
            self.graph_name = graph_name
            self.connection = engine.open_connection()
            self.connection.execute(
                sql.SQL("CREATE GRAPH IF NOT EXISTS {}").format(
                    sql.Identifier(graph_name)
                )
            )
            self.connection.graph(graph_name)
            self._graph = None
            self._engine = engine
            self._url = engine.conninfo
            self._owns_connection = True
        else:
            url = get_from_dict_or_env({"url": url}, "url", "AGENSGRAPH_URL")

            # If graph is not provided, create a new one. Tag the connection so it is
            # identifiable in pg_stat_activity. Autocommit, so a read never leaves the
            # connection sitting in a transaction holding its snapshot and its locks
            # until the next statement.
            self.connection = agensgraph.connect(
                url,
                application_name=f"langchain-agensgraph/{_package_version()}",
                autocommit=True,
            )
            self.graph_name = graph_name
            self._graph = None
            self._engine = None
            self._url = url
            self._owns_connection = True
        self._aconn: Optional[agensgraph.AsyncConnection] = None
        # A unique btree index on the system id (__id__) is what makes MERGE (ingest),
        # delete, and get_by_ids index-backed instead of seq scans. Created lazily on
        # first write (see ``_ensure_id_index``).
        self._create_id_index = True
        self._id_index_ready = False

        # Verify if the version support vector index
        self.verify_vector_support()
        # An embedding is sent as itself rather than as its decimal spelling once the
        # types are registered: 6,152 bytes against 21,504 at 1,536 dimensions. Asked of
        # the graph when there is one, so its async connection is told the same.
        if self._graph is not None:
            self._graph.register_vectors()
        else:
            self.connection.register_vectors()

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

        checked_names(
            index_name=index_name,
            keyword_index_name=keyword_index_name,
            node_label=node_label,
            embedding_node_property=embedding_node_property,
            text_node_property=text_node_property,
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
        # Properties a metadata filter is expected to name. The planner takes either a
        # btree on the filtered property or the vector index, so an unindexed filter
        # key leaves it reading the label. Naming them here builds those indexes.
        self._filter_properties = list(filter_properties or [])
        # Passed to the driver's `vector_search_options`. `hnsw.ef_search` decides how
        # many candidates the index looks at, which is what sets recall.
        self._search_options = dict(search_options or {})
        # Now that the storage type, the distance and the search options are known, and
        # each of them may need a pgvector newer than this one.
        self._verify_vector_version()
        # How wide the embeddings are, learned when it is first needed rather than now.
        # Learning it costs a call to the embedding function -- which for a hosted model
        # is a request, a wait and a charge -- and a store built over an index that exists
        # already is told by the index instead.
        self._embedding_dimension: Optional[int] = None

        # Select the graph through the driver rather than by sending `SET graph_path`.
        # The driver holds the label table a composite value is resolved through, and a
        # statement changing the selection behind its back leaves it unable to trust that
        # table -- so it forgets which graph is selected, and every method that reads the
        # catalogs refuses until one is named again.
        self.query(sql.SQL("CREATE GRAPH IF NOT EXISTS {}").format(
            sql.Identifier(self.graph_name)
        ))
        self.connection.graph(self.graph_name)

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
        Check if the vector index exists in the AgensGraph database and returns its
        embedding dimension.

        This method queries the AgensGraph database for existing indexes and attempts to
        retrieve the dimension of the vector index with the specified name. If the index
        exists, its dimension is returned. If the index doesn't exist, `None` is
        returned.

        Returns:
            int or None: The embedding dimension of the existing index if found.
        """
        # An index answers for this store when it carries the name asked for, or when it
        # is on the label and the property asked for -- so one of a misspelled name and a
        # misspelled label is enough to find it, and the store then takes the rest of the
        # description from what it found.
        index_information = sort_by_index_name(
            [
                found
                for found in self._vector_indexes()
                if found["name"] == self.index_name
                or (
                    found["labelortype"] == self.node_label
                    and found["property"] == self.embedding_node_property
                )
            ],
            self.index_name,
        )
        try:
            self.logger.debug("Index information: %s", index_information[0])
            self.index_name = index_information[0]["name"]
            self.node_label = index_information[0]["labelortype"]
            self.embedding_node_property = index_information[0]["property"]
            self._index_type = index_information[0]["entitytype"]
            embedding_dimension = index_information[0]["dimensions"]

            return embedding_dimension, index_information[0]["entitytype"]
        except IndexError:
            return self._retrieve_promoted_vector_index()

    # An index over a cast, which is how an embedding kept in the property map is
    # indexed: `((embedding)::vector(1536)) vector_cosine_ops`. The property being cast
    # and the width it is cast to are both read out of the definition the server prints.
    _CAST_INDEX = re.compile(
        r"\(+\s*(?P<property>[A-Za-z_][A-Za-z0-9_]*|\"[^\"]+\")\s*\)+::"
        r"\s*\"?(?P<type>vector|halfvec|sparsevec|bit)\"?\s*\(\s*(?P<width>\d+)\s*\)",
        re.IGNORECASE,
    )
    _TSVECTOR_INDEX = re.compile(
        r"to_tsvector\(\s*[^,]+,\s*(?P<property>\"[^\"]+\"|[A-Za-z_][A-Za-z0-9_]*)",
        re.IGNORECASE,
    )

    @staticmethod
    def _unquote(name: str) -> str:
        return name[1:-1].replace('""', '"') if name.startswith('"') else name

    def _vector_indexes(self) -> List[Dict[str, Any]]:
        """Every index of this graph that answers a distance, as this store reads them.

        Read from the definition the server prints for each index, which the driver
        fetches. Nothing is installed to ask this question: a store is not entitled to
        leave a function of its own behind in somebody's database, and one that is
        created on every construction is created against whatever ``search_path``
        happens to resolve to.
        """
        found: List[Dict[str, Any]] = []
        for index in self.connection.indexes():
            match = self._CAST_INDEX.search(index.definition)
            if match is None:
                continue
            found.append(
                {
                    "name": index.name,
                    "labelortype": index.label,
                    "property": self._unquote(match.group("property")),
                    "entitytype": self._entity_type(index.label),
                    "dimensions": int(match.group("width")),
                }
            )
        return found

    def _text_indexes(self) -> List[Dict[str, Any]]:
        """Every index of this graph that answers a keyword search.

        One holds a text vector per property it covers, so the properties are read in the
        order the definition names them -- which is the order they were given in.
        """
        found: List[Dict[str, Any]] = []
        for index in self.connection.indexes():
            properties = [
                self._unquote(m.group("property"))
                for m in self._TSVECTOR_INDEX.finditer(index.definition)
            ]
            if not properties:
                continue
            found.append(
                {
                    "name": index.name,
                    "labelortype": index.label,
                    "properties": properties,
                    "entitytype": self._entity_type(index.label),
                }
            )
        return found

    def _entity_type(self, label: str) -> str:
        """Whether a label holds vertices or edges, as the store names the two."""
        kinds = getattr(self, "_label_kinds", None)
        if kinds is None:
            kinds = {
                row["labname"]: row["labkind"]
                for row in self.query(
                    "SELECT l.labname AS labname, l.labkind AS labkind "
                    "FROM pg_catalog.ag_label l "
                    "JOIN pg_catalog.ag_graph g ON l.graphid = g.oid "
                    "WHERE g.graphname = %(graph)s",
                    params={"graph": self.graph_name},
                )
            }
            self._label_kinds = kinds
        kind = kinds.get(label)
        if kind == "v":
            return IndexType.NODE.value
        if kind == "e":
            return IndexType.RELATIONSHIP.value
        return "UNKNOWN"

    def _retrieve_promoted_vector_index(self) -> Tuple[Optional[int], Optional[str]]:
        """The same index, where the embedding has a column of its own.

        An embedding kept in the property map is indexed over a cast, and the width comes
        out of that cast. A label whose embedding is a promoted column is indexed on the
        column itself, so there is no cast to read -- and the store would say the index is
        missing when it is there and in use.

        The width comes from the column instead, which is where it is declared.
        """
        rows = self.query(
            """
            SELECT c.relname AS name, a.attname AS property,
                   format_type(a.atttypid, a.atttypmod) AS coltype
            FROM pg_catalog.pg_index i
            JOIN pg_catalog.pg_class c ON c.oid = i.indexrelid
            JOIN pg_catalog.pg_class t ON t.oid = i.indrelid
            JOIN pg_catalog.pg_namespace n ON n.oid = t.relnamespace
            JOIN pg_catalog.pg_am am ON am.oid = c.relam
            JOIN pg_catalog.pg_attribute a
              ON a.attrelid = t.oid AND a.attnum = i.indkey[0]
            WHERE n.nspname = %(graph)s AND t.relname = %(label)s
              AND am.amname IN ('hnsw', 'ivfflat')
              AND i.indexprs IS NULL
              AND a.attname = %(property)s
            """,
            params={
                "graph": self.graph_name,
                "label": self.node_label,
                "property": self.embedding_node_property,
            },
        )
        rows = sort_by_index_name(rows, self.index_name)
        if not rows:
            return None, None
        match = re.search(r"vector\((\d+)\)", rows[0]["coltype"] or "")
        self.index_name = rows[0]["name"]
        self._index_type = IndexType.NODE
        return (int(match.group(1)) if match else None), IndexType.NODE

    def retrieve_existing_fts_index(self) -> Optional[str]:
        """
        Check if the fulltext index exists in the AgensGraph database

        This method queries the AgensGraph database for existing fts indexes with the
        specified name.

        Returns:
            (Tuple): keyword index information
        """

        # A keyword index answers for this store only when all three agree: the name, the
        # label, and the set of properties it covers. Unlike the distance index there is
        # nothing to recover a mismatch from -- the properties are what the search reads,
        # so an index over different ones would answer a different question.
        wanted = sorted(self._text_properties())
        index_information = sort_by_index_name(
            [
                found
                for found in self._text_indexes()
                if found["name"] == self.keyword_index_name
                and found["labelortype"] == self.node_label
                and sorted(found["properties"]) == wanted
            ],
            self.keyword_index_name,
        )
        try:
            self.keyword_index_name = index_information[0]["name"]
            self.text_node_property = index_information[0]["properties"][0]
            node_label = index_information[0]["labelortype"]

            self.logger.debug("Keyword index information: %s", index_information[0])
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

    def _id_index_ddl(self):
        """DDL for the unique index on the system id (``__id__``).

        This is the index that turns the per-row ``MERGE (c {{__id__: ...}})`` during
        ingest — and ``delete``/``get_by_ids`` — from a sequential scan into an index
        lookup. Relationship indexes are stored on edges, which AgensGraph already
        key-indexes, so we only create it for node stores.
        """
        return sql.SQL(
            "CREATE UNIQUE PROPERTY INDEX IF NOT EXISTS {name} "
            "ON {label} (__id__)"
        ).format(
            name=sql.Identifier(f"{self.node_label}_id_index"),
            label=sql.Identifier(self.node_label),
        )

    def _ensure_id_index(self) -> None:
        if (
            not self._create_id_index
            or self._id_index_ready
            or self._index_type == IndexType.RELATIONSHIP
        ):
            return
        self.verify_label_existence()
        self.query(self._id_index_ddl())
        self._id_index_ready = True

    async def _aensure_id_index(self) -> None:
        if (
            not self._create_id_index
            or self._id_index_ready
            or self._index_type == IndexType.RELATIONSHIP
        ):
            return
        await self.aquery(
            sql.SQL("CREATE VLABEL IF NOT EXISTS {}").format(
                sql.Identifier(self.node_label)
            )
        )
        await self.aquery(self._id_index_ddl())
        self._id_index_ready = True


    def create_new_index(
        self,
        vector_index_am: VectorIndexAM = None,
        index_config: Optional[IndexConfig] = None,
    ) -> None:
        """
        Construct and execute a Cypher query to create a new vector index.

        Args:
            vector_index_am: Legacy access-method selector. Ignored when
                ``index_config`` is given.
            index_config: Typed :class:`IndexConfig` carrying the access method
                plus build parameters (HNSW ``m``/``ef_construction``, IVFFlat
                ``lists``). Preferred over ``vector_index_am``.
        """
        if index_config is None:
            am = vector_index_am if vector_index_am is not None else self._vector_index_am
            index_config = IndexConfig(am=am)

        # make sure label exists
        self.verify_label_existence()
        index_query = """CREATE PROPERTY INDEX IF NOT EXISTS {index_name}
            ON {node_label} USING {vector_index_am}
            (({embedding_node_property}::{vector_type}({embedding_dimension})) {similarity_metric}){with_options}"""

        # Composed here rather than taken from the driver's `vector_index()`, which builds
        # the same statement for the same shape and quotes its identifiers at least as
        # strictly. Two things it does not express: `IF NOT EXISTS`, without which two
        # processes opening the same store race to create the index and the loser raises;
        # and a build-options clause checked against the access method, which is what
        # refuses `lists` on HNSW rather than dropping it silently. The one part that could
        # drift between the index and the search, the operator class, comes from the
        # driver's own pairing just below.
        #
        # Its sibling `nearest()` is not usable here at all: the search this index serves
        # carries filters, metadata and an optional keyword half, and `nearest()` describes
        # a bare k-nearest read.
        self.query(
            sql.SQL(index_query).format(
                index_name=sql.Identifier(self.index_name),
                node_label=sql.Identifier(self.node_label),
                vector_index_am=sql.SQL(index_config.am.value),
                embedding_node_property=sql.Identifier(self.embedding_node_property),
                vector_type=sql.Identifier(self._vector_type),
                embedding_dimension=self.embedding_dimension,
                # The operator class comes from the driver's own pairing of operator and
                # class, so the index and the search cannot drift apart.
                similarity_metric=sql.SQL(self._operator_class()),
                with_options=sql.SQL(index_config.with_options_clause()),
            )
        )
        self._create_filter_indexes()

    @property
    def embedding_dimension(self) -> int:
        """How many numbers an embedding holds.

        Read from the index when there is one, since the width is declared there, and
        asked of the embedding function only when there is not. Asking is a request to
        whatever is behind that function -- for a hosted model, a wait and a charge -- so
        it is asked once, and not at all by a store built over an index that already
        exists.
        """
        if self._embedding_dimension is None:
            found = self._dimension_from_index()
            self._embedding_dimension = (
                found if found is not None else len(self.embedding.embed_query("foo"))
            )
        return self._embedding_dimension

    def _dimension_from_index(self) -> Optional[int]:
        """The width this store's own vector index was built with, if it exists."""
        try:
            for found in self._vector_indexes():
                if (
                    found["labelortype"] == self.node_label
                    and found["property"] == self.embedding_node_property
                ):
                    return found["dimensions"]
        except Exception:  # pragma: no cover - the catalogs are read again elsewhere
            return None
        return None

    @embedding_dimension.setter
    def embedding_dimension(self, value: int) -> None:
        self._embedding_dimension = value

    def _check_width(self, embeddings: Sequence[Sequence[float]]) -> None:
        """Refuse embeddings that are not the width this store's index was built for.

        Checked here because here it is free: the embeddings are already in hand, so
        nothing has to be asked of the embedding function to find out how wide they are.
        The server refuses them too, at the moment one is written, but it can only say
        that a value was the wrong length -- not that the model and the index disagree,
        which is what actually went wrong.
        """
        if not embeddings:
            return
        width = len(embeddings[0])
        if self._embedding_dimension is None:
            # The width has not been asked for yet, which is the ordinary state of a store
            # that has only been constructed. Returning here instead skipped the check on
            # exactly the paths that need it -- a plain constructor writing for the first
            # time, and a store opened over somebody else's index and only ever read --
            # leaving the server to refuse the value for being the wrong length, which
            # says nothing about the model and the index disagreeing. Read from the index,
            # which costs a catalog read and nothing from the embedding function.
            declared = self._dimension_from_index()
            if declared is None:
                self._embedding_dimension = width  # no index yet; this is what it will be
                return
            self._embedding_dimension = declared
        if width != self._embedding_dimension:
            raise ValueError(
                "The provided embedding function and vector index dimensions do not "
                f"match.\nEmbedding function dimension: {width}\n"
                f"Vector index dimension: {self._embedding_dimension}"
            )

    def _text_properties(self) -> List[str]:
        """The properties a keyword search reads, and the ones its index is built on."""
        return list(self.text_node_properties or [self.text_node_property])

    def _keyword_expressions(self) -> Tuple[sql.Composed, sql.Composed]:
        """What a keyword search matches on, and how it scores what matched.

        Every property named to the store is searched, because every one of them is in
        the index -- reading only the first meant that a store told to index a title and
        a body searched neither when the single property it fell back to did not exist,
        and the keyword half of a hybrid search returned nothing at all.

        The match is one test per property so it lines up with the index, which carries a
        vector per property rather than one over their concatenation. The score is taken
        from the vectors added together, which is one number for the whole element and is
        computed only for the rows that matched.
        """
        question = sql.SQL("plainto_tsquery('english', %(query)s)")
        vectors = [
            sql.SQL("to_tsvector('english', n.{p})").format(p=sql.Identifier(name))
            for name in self._text_properties()
        ]
        match = sql.SQL(" OR ").join(
            sql.SQL("{v} @@ {q}").format(v=v, q=question) for v in vectors
        )
        rank = sql.SQL("ts_rank_cd({v}, {q})").format(
            v=sql.SQL(" || ").join(vectors), q=question
        )
        return match, rank

    def _stored_embedding(self, vector: Sequence[float]) -> Any:
        """An embedding in the form the column it is cast to reads.

        A ``bit`` column holds a string of ones and zeros, so an embedding bound for one
        is binary quantised into that string: written as a list it would be cast from its
        printed form -- ``[1.0, 0.0, ...]`` -- and read as a bit string of that whole
        length, which is not the length the column was declared with.

        Anything else keeps the numbers, which is what ``vector``, ``halfvec`` and
        ``sparsevec`` are cast from.
        """
        if self._vector_type != "bit":
            return list(vector)
        return "".join("1" if float(v) > 0 else "0" for v in vector)

    def _query_embedding(self, vector: Sequence[float]) -> Any:
        """The embedding a search is ranked against, in the form it goes over the wire.

        Sent as itself rather than as its decimal spelling, which for 1,536 dimensions is
        6,152 bytes against 21,504, and leaves the server nothing to parse. A stored
        embedding cannot be sent this way -- it goes into the property map, which holds
        JSON -- but this one is compared against a column and so is a vector all the way
        down.

        Only the plain type is sent this way; the others are cast from what they were
        already being sent as.
        """
        if self._vector_type == "vector":
            return Vector(vector)
        return self._stored_embedding(vector)

    def _distance_operator(self) -> sql.SQL:
        """The operator a search orders by, ready to sit in a statement.

        Jaccard's is ``<%>``, and a statement is scanned for placeholders before it is
        sent, so the percent is doubled -- left alone, ``%>`` is read as a placeholder
        spelling that does not exist and the search is refused before it reaches the
        server.
        """
        return sql.SQL(str(self._distance).replace("%", "%%"))  # type: ignore[arg-type]

    def _operator_class(self) -> str:
        """The operator class an index needs to answer this store's distance.

        The driver names the one for a ``vector`` column, which is the common case. The
        other storage types have their own, differing only in the prefix --
        ``halfvec_cosine_ops`` for a half-precision column, ``sparsevec_l2_ops`` for a
        sparse one -- and a bit column's classes are already named for it, so those are
        taken as they come.
        """
        named = self._distance.operator_class
        if self._vector_type in ("vector", "bit"):
            return named
        return named.replace("vector_", f"{self._vector_type}_", 1)

    def _create_filter_indexes(self) -> None:
        """Index the properties a metadata filter is expected to name.

        The planner serves a filtered search from the btree on the filtered
        property or from the vector index, and with neither it reads the label.
        """
        if not self._filter_properties:
            return
        self.connection.ensure_indexes(
            [
                DesiredIndex(
                    label=self.node_label,
                    properties=(name,),
                    name=f"{self.node_label}_{name}_idx",
                )
                for name in self._filter_properties
            ]
        )

    def create_new_keyword_index(self, fulltext_index_am: FullTextIndexAM = None) -> None:
        """
        This method constructs a Cypher query and executes it to create a new full text
        index in AgensGraph.
        """
        if fulltext_index_am is None:
            fulltext_index_am = self._fulltext_index_am
        # make sure label exists
        self.verify_label_existence()
        node_props = self._text_properties()

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
        elif embedding_dimension:
            # The index says how wide an embedding is, so the store takes it from
            # there. Whether the caller's embedding function agrees is checked
            # against the embeddings themselves on the first write -- asking the
            # function directly costs a request to whatever is behind it, on every
            # construction.
            store.embedding_dimension = embedding_dimension

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

        # The unique index that makes ingest fast is created lazily by
        # ``add_embeddings`` on the actual MERGE key (``__id__``); honor the caller's
        # opt-out here.
        store._create_id_index = create_id_index

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
        *,
        batch_size: int = 1000,
        **kwargs: Any,
    ) -> List[str]:
        """Add embeddings to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore. embeddings: List of
            list of embedding vectors. metadatas: List of metadatas associated with the
            texts. batch_size: Maximum rows per Cypher UNWIND batch. Large ingests
                (>10k rows) are split so each round-trip stays bounded in memory/lock
                duration.
            kwargs: vectorstore specific parameters
        """
        texts_list = list(texts)
        if ids is None:
            ids = [md5(text.encode("utf-8")).hexdigest() for text in texts_list]
        else:
            # ``add_documents`` (LangChain base) builds ``ids`` from ``doc.id``; missing
            # entries arrive as ``None``. Fill them in deterministically so MERGE keys
            # cannot be NULL.
            ids = [
                i if i is not None else md5(t.encode("utf-8")).hexdigest()
                for i, t in zip(ids, texts_list)
            ]

        if not metadatas:
            metadatas = [{} for _ in texts_list]

        self._check_width(embeddings)

        if not (len(texts_list) == len(embeddings) == len(metadatas) == len(ids)):
            raise ValueError(
                "add_embeddings: texts, embeddings, metadatas, ids must have "
                "the same length"
            )

        # Index __id__ before the MERGE loop so each MERGE is an index lookup rather
        # than a seq scan (otherwise ingest is O(n^2)).
        self._ensure_id_index()

        import_query = sql.SQL(
             """UNWIND %(data)s AS row
                MERGE (c:{label} {{__id__: row.id}}) WITH c, row
                SET c += row.metadata
                SET c.{text_property} = row.text
                SET c.{embedding_property} = row.embedding
                SET c.__id__ = row.id """
        ).format(
            label=sql.Identifier(self.node_label),
            embedding_property=sql.Identifier(self.embedding_node_property),
            text_property=sql.Identifier(self.text_node_property),
        )

        n = len(texts_list)
        step = max(1, int(batch_size))
        for start in range(0, n, step):
            end = min(start + step, n)
            batch_rows = [
                {"text": t, "metadata": m,
                 "embedding": self._stored_embedding(e), "id": i}
                for t, m, e, i in zip(
                    texts_list[start:end],
                    metadatas[start:end],
                    embeddings[start:end],
                    ids[start:end],
                )
            ]
            self.query(import_query, params={"data": Jsonb(batch_rows)})

        return ids

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        *,
        batch_size: int = 1000,
        embed_batch_size: Optional[int] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore. metadatas: Optional
            list of metadatas associated with the texts. batch_size: Insert-side batch
            size (see ``add_embeddings``). embed_batch_size: If set, batch the embedding
            API calls too.
                Useful when the embedding provider has per-request payload limits.
                Defaults to None (one call for all texts).
            kwargs: vectorstore specific parameters

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        texts_list = list(texts)
        if embed_batch_size and embed_batch_size > 0 and len(texts_list) > embed_batch_size:
            embeddings: List[List[float]] = []
            for start in range(0, len(texts_list), embed_batch_size):
                embeddings.extend(
                    self.embedding.embed_documents(
                        texts_list[start : start + embed_batch_size]
                    )
                )
        else:
            embeddings = self.embedding.embed_documents(texts_list)
        return self.add_embeddings(
            texts=texts_list,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
            batch_size=batch_size,
            **kwargs,
        )

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        params: Optional[Dict[str, Any]] = None,
        filter: Optional[Dict[str, Any]] = None,
        effective_search_ratio: float = 1.0,
        **kwargs: Any,
    ) -> List[Document]:
        """Run similarity search with AgensgraphVector.

        Args:
            query (str): Query text to search for. k (int): Number of results to return.
            Defaults to 4. params (Dict[str, Any]): The search params for the index
            type.
                Defaults to empty dict.
            filter (Optional[Dict[str, Any]]): Dictionary of argument(s) to
                    filter on metadata.
                Defaults to None.
            effective_search_ratio (float): Over-fetch multiplier passed to
                the HNSW/IVFFlat scan. Use ``>1.0`` for higher recall (esp. when
                combined with ``filter``); the index returns ``int(k * ratio)``
                candidates and the final ``k`` are taken after scoring. Defaults to
                ``1.0``.

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
            effective_search_ratio=effective_search_ratio,
            **kwargs,
        )

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        params: Optional[Dict[str, Any]] = None,
        filter: Optional[Dict[str, Any]] = None,
        effective_search_ratio: float = 1.0,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to. k: Number of Documents to
            return. Defaults to 4. params (Dict[str, Any]): The search params for the
            index type.
                Defaults to empty dict.
            filter (Optional[Dict[str, Any]]): Dictionary of argument(s) to
                    filter on metadata.
                Defaults to None.
            effective_search_ratio (float): Over-fetch multiplier for the
                ANN index. ``>1.0`` increases recall at marginal latency cost.

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
            effective_search_ratio=effective_search_ratio,
            **kwargs,
        )
        return docs

    def similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        effective_search_ratio: float = 1.0,
        hybrid_config: Optional[HybridSearchConfig] = None,
        retrieval_query: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return the top k documents nearest this vector, with their scores.

        ``retrieval_query`` shapes this one search without touching the store: several
        retrievers can share one store -- and so one connection pool -- while each reads
        a different context. Building a second store per shape instead would re-run the
        constructor's connection and index checks for what is only a different tail on
        the same statement. ``None`` means the store's own; the same contract applies
        (the query sees ``node``/``relationship`` and ``score``, returns ``text``,
        ``score``, ``doc_id``, ``metadata``, and doubles literal braces).
        """
        statement, parameters, trim = self._build_search(
            embedding,
            k,
            filter,
            params,
            effective_search_ratio,
            hybrid_config,
            retrieval_query=retrieval_query,
            **kwargs,
        )
        results = self.query(
            statement,
            params=parameters,
            search_options=self._options(kwargs, parameters.get("k")),
        )
        return self._search_results_to_documents(
            results, trim, retrieval_query=retrieval_query, **kwargs
        )

    def _options(
        self, kwargs: Dict[str, Any], fetch_k: Optional[int] = None
    ) -> Dict[str, Any]:
        """The search settings for one call: the store's, with this call's laid over.

        An HNSW index returns at most ``hnsw.ef_search`` candidates, forty by default, so
        asking for more than forty -- with ``k``, or with the over-fetch
        ``effective_search_ratio`` asks for -- quietly got forty. The index is told how
        many to look at, unless the caller has said what it wants.

        pgvector takes 1..1000 for it and refuses anything outside. A caller asking for
        more than that is told so rather than quietly given a thousand: silently returning
        fewer results than were asked for is the bug this setting exists to fix, and
        capping without a word would move it to a higher number rather than remove it.
        """
        options = {**self._search_options, **(kwargs.get("search_options") or {})}
        if fetch_k and fetch_k > 40 and "hnsw.ef_search" not in options:
            if fetch_k > 1000:
                raise ValueError(
                    f"This search asks the index for {fetch_k} candidates, and pgvector "
                    "accepts at most 1000 for hnsw.ef_search. Ask for a smaller k (or a "
                    "smaller effective_search_ratio), or pass "
                    "search_options={'hnsw.ef_search': 1000} to accept looking at fewer "
                    "candidates than results requested."
                )
            options["hnsw.ef_search"] = fetch_k
        return options

    def _build_search(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        effective_search_ratio: float = 1.0,
        hybrid_config: Optional[HybridSearchConfig] = None,
        retrieval_query: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[Any, Dict[str, Any], int]:
        """Build the search statement and its parameters, without running it.

        Shared by the blocking and the awaiting path, which differ only in how they
        wait for the answer.

        Perform a similarity search in the AgensGraph database using a given vector and
        return the top k similar documents with their scores.

        This method uses a Cypher query to find the top k documents that are most
        similar to a given embedding. The similarity is measured using a vector index in
        the AgensGraph database. The results are returned as a list of tuples, each
        containing a Document object and its similarity score.

        Args:
            embedding (List[float]): The embedding vector to compare against. k (int,
            optional): The number of top similar documents to retrieve. filter
            (Optional[Dict[str, Any]]): Dictionary of argument(s) to
                    filter on metadata.
                Defaults to None.
            params (Dict[str, Any]): The search params for the index type.
                Defaults to empty dict.

        Returns:
            List[Tuple[Document, float]]: A list of tuples, each containing
                                a Document object and its similarity score.
        """
        filter_params = {}

        # No dimension guard on the pattern: `array_size()` is opaque to the planner, so
        # a query carrying it drops the vector index and reads every element. The
        # dimension is enforced where the value is written -- the vector index refuses a
        # wrong-length embedding at insert time -- and an element carrying no embedding
        # is ranked last.
        if self._index_type == IndexType.RELATIONSHIP:
            base_index_query = """MATCH ()-[n:{label}]->() """
        else:
            base_index_query = """MATCH (n:{label}) """

        base_cosine_query = """
            WITH n, n.{embedding_property}::{vector_type}({embedding_dimension})
            {distance_op} %(embedding)s::{vector_type}({embedding_dimension}) AS
            inv_score ORDER BY inv_score LIMIT %(k)s WITH n, 1 - inv_score AS score
            """

        if filter:
            # Metadata filtering and hybrid doesn't work
            if self.search_type == SearchType.HYBRID:
                raise ValueError(
                    "Metadata filtering can't be use in combination with "
                    "a hybrid search approach"
                )

            filter_snippets, filter_params = construct_metadata_filter(filter)
            # `WHERE`, not `AND`: the pattern above carries no clause of its own.
            index_query = base_index_query + "WHERE " + filter_snippets + base_cosine_query
        else:
            if self.search_type == SearchType.HYBRID:
                # Both halves carry the element's own identity, and that is what they are
                # joined on. Every vertex has one, whereas `__id__` is a property only
                # ``add_embeddings`` writes -- so a store built over a graph that was
                # already there had null on both sides of the join, nothing matched, and
                # every hit came back scored by one half of the fusion instead of two.
                index_query = (
                    """
                        UNWIND [1] as a WITH (
                            SELECT jsonb_agg(jsonb_build_object('n', n, 'score', score))
                            FROM (
                                WITH semantic_search AS (
                                    SELECT eid, n,
                                           RANK () OVER (ORDER BY inv_score) AS rank
                                    FROM (""" +
                                        base_index_query +
                                     """RETURN id(n) AS eid, properties(n) as n,
                                               n.{embedding_property}::{vector_type}({embedding_dimension})
                                               {distance_op}
                                               %(embedding)s::{vector_type}({embedding_dimension})
                                               AS inv_score
                                        ORDER BY inv_score LIMIT %(k)s
                                    )t
                                ), keyword_search AS (
                                    SELECT eid, n,
                                           RANK () OVER (ORDER BY score DESC) AS rank
                                    FROM (
                                        MATCH (n:{label}) WHERE {keyword_match}
                                        RETURN id(n) AS eid, properties(n) as n,
                                               {keyword_rank} AS score
                                        ORDER BY score DESC LIMIT %(k)s
                                    )t
                                ) SELECT
                                    COALESCE(semantic_search.n, keyword_search.n) AS n,
                                    {vector_weight} * COALESCE(1.0 / ({rank_constant} +
                                    semantic_search.rank), 0.0) + {keyword_weight} *
                                    COALESCE(1.0 / ({rank_constant} +
                                    keyword_search.rank), 0.0) AS score
                                FROM semantic_search FULL OUTER JOIN keyword_search ON
                                semantic_search.eid = keyword_search.eid
                                ORDER BY score DESC
                            )
                        ) AS outputs UNWIND outputs as output WITH output.score AS
                        score,
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

        # A per-call query wins over the store's; either silences the default tail.
        wanted_query = (
            retrieval_query if retrieval_query is not None else self.retrieval_query
        )
        if not wanted_query:
            if kwargs.get("return_embeddings"):
                wanted_query = (
                    """RETURN {var}.{text_property} AS text, score,
                    {var}.__id__ AS doc_id, {var} ||
                    jsonb_build_object({text_property_literal}, Null,
                    {embedding_property_literal}, Null, '__id__', Null,
                    '_embedding_', {var}.{embedding_property}) AS metadata"""
                ).replace("{var}", var)
            else:
                wanted_query = (
                    """RETURN {var}.{text_property} AS text, score,
                    {var}.__id__ AS doc_id, {var} ||
                    jsonb_build_object({text_property_literal}, Null,
                    {embedding_property_literal}, Null, '__id__', Null) AS metadata"""
                ).replace("{var}", var)

        read_query = index_query + wanted_query

        # Over-fetch from the ANN index when caller asks for higher recall.
        # ``effective_search_ratio`` >= 1.0; final results are trimmed to k.
        ratio = max(1.0, float(effective_search_ratio))
        fetch_k = max(k, int(k * ratio))

        hcfg = hybrid_config or DEFAULT_HYBRID_CONFIG

        parameters = {
            "k": fetch_k,
            "embedding": self._query_embedding(embedding),
            # Only the HYBRID read query references %(query)s; for a plain VECTOR search
            # by vector the caller need not pass a `query` text.
            "query": remove_lucene_chars(kwargs.get("query") or ""),
            "embedding_property": self.embedding_node_property,
            "text_property": self.text_node_property,
            "text_node_properties": Jsonb(self.text_node_properties),
            **(params or {}),
            **filter_params,
        }

        keyword_match, keyword_rank = self._keyword_expressions()
        composed = sql.SQL(read_query).format(
            label=sql.Identifier(self.node_label),
            keyword_match=keyword_match,
            keyword_rank=keyword_rank,
            embedding_property=sql.Identifier(self.embedding_node_property),
            text_property=sql.Identifier(self.text_node_property),
            embedding_dimension=self.embedding_dimension,
            vector_type=sql.Identifier(self._vector_type),
            distance_op=self._distance_operator(),
            text_property_literal=sql.Literal(self.text_node_property),
            embedding_property_literal=sql.Literal(self.embedding_node_property),
            # Hybrid RRF fusion knobs (only referenced by the HYBRID query).
            rank_constant=sql.Literal(int(hcfg.rank_constant)),
            vector_weight=sql.Literal(float(hcfg.vector_weight)),
            keyword_weight=sql.Literal(float(hcfg.keyword_weight)),
        )
        return composed, parameters, k

    def _search_results_to_documents(
        self,
        results: List[Dict[str, Any]],
        k: int,
        retrieval_query: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Turn what a search returned into documents.

        ``retrieval_query`` here is only the per-call override the search ran with, so
        a complaint about a bad ``text`` column blames the query that actually built the
        row rather than whichever one the store happens to hold.
        """
        custom_query = retrieval_query or self.retrieval_query
        if any(result["text"] is None for result in results):
            if not custom_query:
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
        # A retrieval query that carries no embedding at all is not a complaint: the
        # caller that wants one reads it from the store instead. A null one is, since it
        # says the element has no embedding stored.
        if kwargs.get("return_embeddings") and any(
            "_embedding_" in result["metadata"]
            and result["metadata"]["_embedding_"] is None
            for result in results
        ):
            if not custom_query:
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
                    id=result.get("doc_id"),
                    page_content=dict_to_yaml_str(result["text"])
                    if isinstance(result["text"], dict)
                    else result["text"],
                    metadata={
                        # Drop both our system field and any None values.
                        k: v
                        for k, v in result["metadata"].items()
                        if v is not None and k != "__id__"
                    },
                ),
                result["score"],
            )
            for result in results
        ]

        # Trim over-fetch back to caller-requested k.
        return docs[:k]

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to. k: Number of Documents
            to return. Defaults to 4. filter (Optional[Dict[str, Any]]): Dictionary of
            argument(s) to
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
        Return AgensgraphVector initialized from texts and embeddings. AgensGraph
        credentials are required in the form of
        `url`(postgresql://username:password@host:port/dbname) and optional `graph_name`
        parameters.
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

        Return AgensgraphVector initialized from documents and embeddings. AgensGraph
        credentials are required in the form of
        `url`(postgresql://username:password@host:port/dbname) and optional `graph_name`
        parameters.

        Example:
            .. code-block:: python

                from langchain_agensgraph.vectorstores.agensgraph_vector import
                AgensgraphVector from langchain_openai import OpenAIEmbeddings
                embeddings = OpenAIEmbeddings() text_embeddings =
                embeddings.embed_documents(texts) text_embedding_pairs = list(zip(texts,
                text_embeddings)) vectorstore = AgensgraphVector.from_embeddings(
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
        Get instance of an existing AgensGraph vector index. This method will return the
        instance of the store without inserting any new embeddings. AgensGraph
        credentials are required in the form of
        `url`(postgresql://username:password@host:port/dbname) and optional `graph_name`
        parameters along with the `index_name` definition.
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
        if embedding_dimension:
            # The index says how wide an embedding is, so the store takes it from
            # there. Whether the caller's embedding function agrees is checked
            # against the embeddings themselves on the first write -- asking the
            # function directly costs a request to whatever is behind it, on every
            # construction.
            store.embedding_dimension = embedding_dimension

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
        Get instance of an existing AgensGraph relationship vector index. This method
        will return the instance of the store without inserting any new embeddings.
        AgensGraph credentials are required in the form of
        `url`(postgresql://username:password@host:port/dbname) and optional `graph_name`
        parameters along with the `index_name` definition.
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
        if embedding_dimension:
            # The index says how wide an embedding is, so the store takes it from
            # there. Whether the caller's embedding function agrees is checked
            # against the embeddings themselves on the first write -- asking the
            # function directly costs a request to whatever is behind it, on every
            # construction.
            store.embedding_dimension = embedding_dimension

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
        Return AgensgraphVector initialized from documents and embeddings. AgensGraph
        credentials are required in the form of
        `url`(postgresql://username:password@host:port/dbname) and optional `graph_name`
        parameters.
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
        parameters and the existing graph. It validates the existence of the indices and
        creates new ones if they don't exist.

        Returns:
        AgensgraphVector: An instance of AgensgraphVector initialized with the provided
        parameters
                    and existing graph.

        Example:
        >>> agensgraph_vector = AgensgraphVector.from_existing_graph(
        ...     embedding=my_embedding,
        ...     node_label="Document",
        ...     embedding_node_property="embedding",
        ...     text_node_properties=["title", "content"]
        ... )

        Note:
        AgensGraph credentials are required in the form of
        `url`(postgresql://username:password@host:port/dbname) and optional `graph_name`
        parameters.
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
                          SELECT string_agg(E'\\n' || k || ': ' || coalesce(n->>k, ''),
                          '') FROM jsonb_array_elements_text(%(text_node_properties)s)
                          AS k
                        ) AS text, n.__id__ AS doc_id,
                        n || jsonb_build_object({embedding_property_literal}, NULL, '__id__', Null,"""
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
        elif embedding_dimension:
            # The index says how wide an embedding is, so the store takes it from
            # there. Whether the caller's embedding function agrees is checked
            # against the embeddings themselves on the first write -- asking the
            # function directly costs a request to whatever is behind it, on every
            # construction.
            store.embedding_dimension = embedding_dimension
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
                    WHERE n.{embedding_property} IS null AND any(k IN
                    %(text_node_properties)s WHERE n[k] IS NOT NULL) RETURN
                    toString(id(n)) AS id, (
                                SELECT string_agg(E'\\n' || k || ': ' || coalesce(n->>k,
                                ''), '') FROM
                                jsonb_array_elements_text(%(text_node_properties)s) AS k
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
                    {"id": el["id"], "embedding": store._stored_embedding(embedding)}
                    for el, embedding in zip(data, text_embeddings)
                ]
            params = {
                "data": Jsonb(rows)
            }

            # `__id__` is written as well, and it is the element's own identity in text
            # because a graph that was already here had no id of ours to carry. Without
            # one a document came back with no id, and `delete` and `get_by_ids`, which
            # name an element by it, could not reach anything in the store.
            store.query(sql.SQL(
                 """UNWIND %(data)s AS row
                    MATCH (n:{node_label}) WHERE id(n) = (row.id)::graphid SET
                    n.{embedding_node_property} = row.embedding,
                    n.__id__ = row.id
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

        Maximal marginal relevance optimizes for similarity to query AND diversity among
        selected documents.

        Args:
            query: search query text. k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm. lambda_mult:
            Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding to maximum
                        diversity and 1 to minimum diversity. Defaults to 0.5.
            filter: Filter on metadata properties, e.g.
                            {
                                "str_property": "foo", "int_property": 123
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
        if not got_docs:
            return []

        got_embeddings = self._candidate_embeddings([doc for doc, _ in got_docs])

        # Select documents using maximal marginal relevance
        selected_indices = maximal_marginal_relevance(
            np.array(query_embedding), got_embeddings, lambda_mult=lambda_mult, k=k
        )
        selected_docs = [got_docs[i][0] for i in selected_indices]

        for doc in selected_docs:
            doc.metadata.pop("_embedding_", None)

        return selected_docs

    def _candidate_embeddings(self, docs: List[Document]) -> List[List[float]]:
        """The embedding of each candidate, in the order they were ranked.

        Taken from the documents when the search carried them, and read from the store
        when it did not: a caller's own retrieval query shapes the document it returns and
        has no reason to carry an embedding, so asking it for one made this refuse to run
        for every store that has one.
        """
        carried = [doc.metadata.get("_embedding_") for doc in docs]
        if all(vector is not None for vector in carried):
            return carried  # type: ignore[return-value]

        ids = [doc.id for doc in docs]
        missing = [i for i in ids if i is None]
        if missing:
            raise ValueError(
                "Maximal marginal relevance needs an embedding per candidate, and the "
                f"search returned a document with no id to read one by. Have the "
                f"retrieval query return `{self.text_node_property}`'s element id as "
                "`doc_id`, or return the embedding as `_embedding_` in the metadata."
            )
        stored = self._read_embeddings(ids)  # type: ignore[arg-type]
        return [
            carried[i] if carried[i] is not None else stored[doc_id]
            for i, doc_id in enumerate(ids)
        ]

    def _named_by_id(
        self, ids: Sequence[str], params: Dict[str, Any]
    ) -> sql.Composed:
        """The elements carrying these ids, as the opening of a statement.

        Each id is an equality the unique index over it answers. A bound list would be
        compared by containment instead -- the list asked whether it holds the property --
        and containment is not something that index can answer, so the label would be read
        whole.

        The ids are unwound rather than written out as one equality each. Written out, the
        statement grows with the list and every term has to be planned and then tried,
        which with no index to answer them is a pass over the label per term. Unwound it
        is one probe repeated, which is the right shape whether or not the index is there
        -- and it is not always there, since a store opened over a graph somebody else
        built has whatever indexes that graph has.
        """
        params["wanted_ids"] = Jsonb(list(ids))
        return sql.SQL(
            "UNWIND %(wanted_ids)s AS wanted "
            "MATCH (n:{label}) WHERE n.__id__ = wanted "
        ).format(label=sql.Identifier(self.node_label))

    def _read_embeddings_query(
        self, ids: List[str]
    ) -> Tuple[sql.Composed, Dict[str, Any]]:
        """Read the stored embedding of each named element."""
        params: Dict[str, Any] = {}
        statement = sql.SQL(
            "{by} "
            "RETURN n.__id__ AS id, n.{embedding_property} AS embedding"
        ).format(
            by=self._named_by_id(ids, params),
            embedding_property=sql.Identifier(self.embedding_node_property),
        )
        return statement, params

    def _read_embeddings(self, ids: List[str]) -> Dict[str, List[float]]:
        """The stored embedding of each named element, read as numbers rather than text.

        This is the one read whose rows are almost entirely float arrays, and the only one
        whose result goes straight into arithmetic. Asked for in the binary form the server
        already holds them in, rather than as decimal text that has to be printed, sent and
        parsed back.
        """
        statement, params = self._read_embeddings_query(ids)
        return {
            row["id"]: row["embedding"]
            for row in self.query(statement, params, binary_=True)
        }

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

        # Every strategy's search returns `1 - distance`. Only cosine, Hamming and
        # Jaccard have a bounded distance, so the score is clamped into [0, 1] rather
        # than scaled: it is a ranking. A caller who knows their embedding's scale
        # passes `relevance_score_fn`.
        return lambda score: max(0.0, min(1.0, score))

    @property
    def _in_transaction(self) -> bool:
        """Does the calling thread or task hold a transaction on this connection?

        Built from a graph, the store runs on that graph's connection, so a block the
        graph opened -- ``read_only``, ``add_graph_documents`` -- is a block the store's
        statements are inside. The graph is the one that knows, and it answers for the
        caller asking rather than for every caller, so a concurrent search still goes to
        the pool.
        """
        return self._graph is not None and self._graph._in_transaction

    @property
    def _in_async_transaction(self) -> bool:
        """The same, for the async connection the store shares with a graph."""
        return self._graph is not None and self._graph._in_async_transaction

    @contextmanager
    def _acquire(self) -> "Iterator[agensgraph.Connection]":
        """Yield the connection ``query`` should run on (pooled or dedicated)."""
        if self._engine is not None and not self._in_transaction:
            with self._engine.connection(self.graph_name) as conn:
                yield conn
        else:
            yield self.connection

    @asynccontextmanager
    async def _aacquire(self) -> "AsyncIterator[agensgraph.AsyncConnection]":
        """Async sibling of :meth:`_acquire`."""
        if self._engine is not None and not self._in_async_transaction:
            async with self._engine.aconnection(self.graph_name) as conn:
                yield conn
        else:
            yield await self._aconn_get()

    def verify_vector_support(self) -> None:
        """Make sure vectors can be read in this database.

        Asked of the catalogs, which any role may read. Creating the extension is a
        privilege an application role usually does not hold, so trying it turned a
        database where pgvector was simply not created yet into a permission error, and
        reported that error as pgvector not being on disk -- which sends whoever reads it
        to build something that may already be built.
        """
        if self.connection.has_vectors():
            return
        raise ValueError(
            "This database has no pgvector extension, which a vector store needs. "
            "Run CREATE EXTENSION vector as a role that may, and if the extension is "
            "not on the server build it from https://github.com/pgvector/pgvector "
            "against this install's pg_config and `make install` first. AgensGraph "
            "does not bundle it."
        )

    # What each feature needs, as pgvector versions its own.
    _HALF_AND_SPARSE_VECTORS = (0, 7, 0)
    _ITERATIVE_INDEX_SCANS = (0, 8, 0)

    def _verify_vector_version(self) -> None:
        """Refuse a feature this pgvector is too old to have, saying which.

        pgvector gates its own features on its version, and a server that does not have
        one reports it as a type or an operator that does not exist -- which reads as a
        mistake in the statement rather than as a version to upgrade.
        """
        version = self.connection.vector_version()
        if version is None:  # pragma: no cover - verify_vector_support ran first
            return

        def refuse(what: str, needs: Tuple[int, ...]) -> None:
            raise ValueError(
                f"{what} needs pgvector {'.'.join(str(p) for p in needs)} or later, "
                f"and this database has {'.'.join(str(p) for p in version)}."
            )

        if self._vector_type in ("halfvec", "sparsevec") and (
            version < self._HALF_AND_SPARSE_VECTORS
        ):
            refuse(f'vector_type="{self._vector_type}"', self._HALF_AND_SPARSE_VECTORS)
        if self._distance_strategy == DistanceStrategy.TAXICAB and (
            version < self._HALF_AND_SPARSE_VECTORS
        ):
            refuse("TAXICAB distance", self._HALF_AND_SPARSE_VECTORS)
        if self._search_options.get("hnsw.iterative_scan") and (
            version < self._ITERATIVE_INDEX_SCANS
        ):
            refuse("hnsw.iterative_scan", self._ITERATIVE_INDEX_SCANS)

    def query(
        self,
        query: Any,
        params: Optional[dict] = None,
        search_options: Optional[Dict[str, Any]] = None,
        binary_: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Query the graph by taking a cypher query, executing it and converting the result

        Args:
            query (str): a cypher query to be executed params (dict): parameters for the
            query

        Returns:
            List[Dict[str, Any]]: a list of dictionaries containing the result set. A
            vertex is an :class:`agensgraph.Vertex`, an edge an ``Edge`` and a path a
            ``Path`` -- the driver decodes the wire, so nothing here reads the printed
            text.
        """
        in_txn = self._in_transaction
        with self._acquire() as conn:
            try:
                if search_options:
                    # On the same connection the search is about to run on, which is the
                    # only place it means anything, and inside a transaction so that it
                    # ends with the search. Set for the session it would stay on a pooled
                    # connection and tune every later borrower's search instead.
                    with conn.transaction():
                        conn.vector_search_options(search_options)
                        result = conn.execute_query(
                            query, params, row_=dict_row, binary_=binary_
                        )
                else:
                    result = conn.execute_query(
                        query, params, row_=dict_row, binary_=binary_
                    )
                if not in_txn and not conn.autocommit:
                    conn.commit()
            except Exception as e:
                if not in_txn and not conn.autocommit:
                    conn.rollback()
                raise AgensQueryException(
                    {
                        "message": "Error executing graph query: {}".format(query),
                        "detail": str(e),
                    }
                ) from e
            return result.records

    # ----- VectorStore parity (delete / get_by_ids / etc.) -----

    def delete(
        self,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Optional[bool]:
        """Delete nodes whose ``id`` property is in ``ids``.

        Uses ``DETACH DELETE`` so any incident edges are removed too. Returns ``True``
        if the request was issued. ``None`` if ``ids`` is empty/None.
        """
        if not ids:
            return None
        params: Dict[str, Any] = {}
        delete_query = sql.SQL(
            "{by} DETACH DELETE n"
        ).format(by=self._named_by_id(list(ids), params))
        self.query(delete_query, params=params)
        return True

    def get_by_ids(self, ids: List[str], /) -> List[Document]:
        """Return ``Document`` objects for nodes whose ``id`` is in ``ids``.

        Order of the returned list is **not** guaranteed to match ``ids``. Missing ids
        are simply absent — no exception is raised.
        """
        if not ids:
            return []
        params: Dict[str, Any] = {}
        fetch_query = sql.SQL(
            "{by} "
            "RETURN n.__id__ AS id, n.{text_property} AS text, properties(n) AS meta"
        ).format(
            by=self._named_by_id(list(ids), params),
            text_property=sql.Identifier(self.text_node_property),
        )
        rows = self.query(fetch_query, params=params)
        docs: List[Document] = []
        for row in rows:
            meta = row.get("meta") or {}
            # Strip the AgensgraphVector system fields so user metadata is clean.
            for sys_field in ("__id__", self.embedding_node_property, self.text_node_property):
                meta.pop(sys_field, None)
            text = row.get("text") or ""
            doc_id = row.get("id")
            docs.append(Document(id=doc_id, page_content=text, metadata=meta))
        return docs

    # ----- async surface (hot RAG paths) -----

    async def _aconn_get(self) -> agensgraph.AsyncConnection:
        """Lazily-open an async connection bound to ``self.graph_name``.

        Shares with the parent ``AgensGraph`` if one was passed to ``__init__``.
        Otherwise opens its own ``AsyncConnection`` against the stored URL.
        """
        if self._graph is not None:
            return await self._graph._aconn_get()
        if self._aconn is None or self._aconn.closed:
            self._aconn = await agensgraph.AsyncConnection.connect(
                self._url, autocommit=True
            )
            await self._aconn.register_vectors()
            # Selects the graph and fills the label table in one call.
            await self._aconn.graph(self.graph_name)
        return self._aconn

    async def aquery(
        self,
        query: Any,
        params: Optional[dict] = None,
        search_options: Optional[Dict[str, Any]] = None,
        binary_: bool = False,
    ) -> List[Dict[str, Any]]:
        """Async sibling of :meth:`query` (pooled when an engine is configured)."""
        in_txn = self._in_async_transaction
        async with self._aacquire() as conn:
            try:
                if search_options:
                    # Inside a transaction, so the tuning ends with the search rather than
                    # staying on a pooled connection for whoever borrows it next.
                    async with conn.transaction():
                        await conn.vector_search_options(search_options)
                        result = await conn.execute_query(
                            query, params, row_=dict_row, binary_=binary_
                        )
                else:
                    result = await conn.execute_query(
                        query, params, row_=dict_row, binary_=binary_
                    )
                if not in_txn and not conn.autocommit:
                    await conn.commit()
            except Exception as e:
                if not in_txn and not conn.autocommit:
                    await conn.rollback()
                raise AgensQueryException(
                    {
                        "message": "Error executing graph query: {}".format(query),
                        "detail": str(e),
                    }
                ) from e
            return result.records

    async def aadd_embeddings(
        self,
        texts: Iterable[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        *,
        batch_size: int = 1000,
        **kwargs: Any,
    ) -> List[str]:
        """Async sibling of :meth:`add_embeddings`."""
        texts_list = list(texts)
        if ids is None:
            ids = [md5(text.encode("utf-8")).hexdigest() for text in texts_list]
        else:
            ids = [
                i if i is not None else md5(t.encode("utf-8")).hexdigest()
                for i, t in zip(ids, texts_list)
            ]
        if not metadatas:
            metadatas = [{} for _ in texts_list]
        self._check_width(embeddings)

        if not (len(texts_list) == len(embeddings) == len(metadatas) == len(ids)):
            raise ValueError(
                "aadd_embeddings: texts, embeddings, metadatas, ids must have "
                "the same length"
            )
        await self._aensure_id_index()
        import_query = sql.SQL(
            """UNWIND %(data)s AS row
               MERGE (c:{label} {{__id__: row.id}}) WITH c, row
               SET c += row.metadata
               SET c.{text_property} = row.text
               SET c.{embedding_property} = row.embedding
               SET c.__id__ = row.id """
        ).format(
            label=sql.Identifier(self.node_label),
            embedding_property=sql.Identifier(self.embedding_node_property),
            text_property=sql.Identifier(self.text_node_property),
        )
        n = len(texts_list)
        step = max(1, int(batch_size))
        for start in range(0, n, step):
            end = min(start + step, n)
            batch_rows = [
                {"text": t, "metadata": m,
                 "embedding": self._stored_embedding(e), "id": i}
                for t, m, e, i in zip(
                    texts_list[start:end],
                    metadatas[start:end],
                    embeddings[start:end],
                    ids[start:end],
                )
            ]
            await self.aquery(import_query, params={"data": Jsonb(batch_rows)})
        return ids

    async def aadd_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        *,
        batch_size: int = 1000,
        embed_batch_size: Optional[int] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Async sibling of :meth:`add_texts`.

        ``self.embedding`` is invoked synchronously since LangChain's
        :class:`Embeddings` interface does not require async support; callers that need
        fully async embedding should pre-compute embeddings and use
        :meth:`aadd_embeddings`.
        """
        texts_list = list(texts)
        if embed_batch_size and embed_batch_size > 0 and len(texts_list) > embed_batch_size:
            embeddings: List[List[float]] = []
            for start in range(0, len(texts_list), embed_batch_size):
                embeddings.extend(
                    self.embedding.embed_documents(
                        texts_list[start : start + embed_batch_size]
                    )
                )
        else:
            embeddings = self.embedding.embed_documents(texts_list)
        return await self.aadd_embeddings(
            texts=texts_list,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
            batch_size=batch_size,
            **kwargs,
        )

    async def asimilarity_search(
        self,
        query: str,
        k: int = 4,
        params: Optional[Dict[str, Any]] = None,
        filter: Optional[Dict[str, Any]] = None,
        effective_search_ratio: float = 1.0,
        **kwargs: Any,
    ) -> List[Document]:
        """Async sibling of :meth:`similarity_search`."""
        embedding = self.embedding.embed_query(text=query)
        results = await self.asimilarity_search_with_score_by_vector(
            embedding=embedding,
            k=k,
            query=query,
            params=params,
            filter=filter,
            effective_search_ratio=effective_search_ratio,
            **kwargs,
        )
        return [doc for doc, _ in results]

    async def asimilarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        params: Optional[Dict[str, Any]] = None,
        filter: Optional[Dict[str, Any]] = None,
        effective_search_ratio: float = 1.0,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Async sibling of :meth:`similarity_search_with_score`."""
        embedding = self.embedding.embed_query(query)
        return await self.asimilarity_search_with_score_by_vector(
            embedding=embedding,
            k=k,
            query=query,
            params=params,
            filter=filter,
            effective_search_ratio=effective_search_ratio,
            **kwargs,
        )

    async def asimilarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        effective_search_ratio: float = 1.0,
        hybrid_config: Optional[HybridSearchConfig] = None,
        retrieval_query: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Async sibling of :meth:`similarity_search_with_score_by_vector`.

        The statement is built by the same code the blocking path builds it with and
        then awaited, so two searches on one store overlap. ``hybrid_config`` and
        ``retrieval_query`` are named here for the same reason they are named on the
        blocking twin: before this they only reached `_build_search` by falling through
        ``**kwargs``, which worked but promised nothing -- an unrelated keyword of the
        same name added later would have bound silently.
        """
        statement, parameters, trim = self._build_search(
            embedding,
            k,
            filter,
            params,
            effective_search_ratio,
            hybrid_config,
            retrieval_query=retrieval_query,
            **kwargs,
        )
        results = await self.aquery(
            statement,
            params=parameters,
            search_options=self._options(kwargs, parameters.get("k")),
        )
        return self._search_results_to_documents(
            results, trim, retrieval_query=retrieval_query, **kwargs
        )

    async def asearch(
        self, query: str, search_type: str, **kwargs: Any
    ) -> List[Document]:
        """Async sibling of ``search``, which is what a retriever calls.

        The base class's default runs the blocking ``search`` in an executor, so a
        retriever built on this store never reached the awaiting path at all.
        """
        if search_type == "similarity":
            return await self.asimilarity_search(query, **kwargs)
        if search_type == "similarity_score_threshold":
            scored = await self.asimilarity_search_with_relevance_scores(query, **kwargs)
            return [doc for doc, _ in scored]
        if search_type == "mmr":
            return await self.amax_marginal_relevance_search(query, **kwargs)
        raise ValueError(
            f"search_type of {search_type} not allowed. Expected search_type to be "
            f"'similarity', 'similarity_score_threshold' or 'mmr'."
        )

    async def asimilarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Async sibling of :meth:`similarity_search_by_vector`.

        Overridden because the base class's default hands the blocking method to an
        executor, which is a thread doing what this can now simply await.
        """
        pairs = await self.asimilarity_search_with_score_by_vector(
            embedding=embedding, k=k, filter=filter, params=params, **kwargs
        )
        return [doc for doc, _ in pairs]

    async def aadd_documents(
        self, documents: List[Document], **kwargs: Any
    ) -> List[str]:
        """Async sibling of ``add_documents``.

        The base class's default runs the blocking path in an executor even though an
        awaiting ingest exists, so the indexing API never reached it.
        """
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        # Every document's id is passed, present or not: `add_embeddings` fills a
        # missing one in from the text.
        ids = kwargs.pop("ids", None) or [doc.id for doc in documents]
        return await self.aadd_texts(texts, metadatas=metadatas, ids=ids, **kwargs)

    async def amax_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Async sibling of :meth:`max_marginal_relevance_search`."""
        query_embedding = self.embedding.embed_query(query)
        got = await self.asimilarity_search_with_score_by_vector(
            embedding=query_embedding,
            query=query,
            k=fetch_k,
            return_embeddings=True,
            filter=filter,
            **kwargs,
        )
        if not got:
            return []
        selected = maximal_marginal_relevance(
            np.array(query_embedding),
            await self._acandidate_embeddings([doc for doc, _ in got]),
            lambda_mult=lambda_mult,
            k=k,
        )
        docs = [got[i][0] for i in selected]
        for doc in docs:
            doc.metadata.pop("_embedding_", None)
        return docs

    async def _acandidate_embeddings(
        self, docs: List[Document]
    ) -> List[List[float]]:
        """Async sibling of :meth:`_candidate_embeddings`."""
        carried = [doc.metadata.get("_embedding_") for doc in docs]
        if all(vector is not None for vector in carried):
            return carried  # type: ignore[return-value]
        ids = [doc.id for doc in docs]
        if any(i is None for i in ids):
            raise ValueError(
                "Maximal marginal relevance needs an embedding per candidate, and the "
                "search returned a document with no id to read one by. Have the "
                "retrieval query return the element id as `doc_id`, or return the "
                "embedding as `_embedding_` in the metadata."
            )
        statement, params = self._read_embeddings_query(ids)  # type: ignore[arg-type]
        # In the binary form the server holds them in; see `_read_embeddings`.
        rows = await self.aquery(statement, params, binary_=True)
        stored = {row["id"]: row["embedding"] for row in rows}
        return [
            carried[i] if carried[i] is not None else stored[doc_id]
            for i, doc_id in enumerate(ids)
        ]

    async def adelete(
        self, ids: Optional[List[str]] = None, **kwargs: Any
    ) -> Optional[bool]:
        """Async sibling of :meth:`delete`."""
        if not ids:
            return None
        params: Dict[str, Any] = {}
        delete_query = sql.SQL(
            "{by} DETACH DELETE n"
        ).format(by=self._named_by_id(list(ids), params))
        await self.aquery(delete_query, params=params)
        return True

    async def aget_by_ids(self, ids: List[str], /) -> List[Document]:
        """Async sibling of :meth:`get_by_ids`."""
        if not ids:
            return []
        params: Dict[str, Any] = {}
        fetch_query = sql.SQL(
            "{by} "
            "RETURN n.__id__ AS id, n.{text_property} AS text, properties(n) AS meta"
        ).format(
            by=self._named_by_id(list(ids), params),
            text_property=sql.Identifier(self.text_node_property),
        )
        rows = await self.aquery(fetch_query, params=params)
        docs: List[Document] = []
        for row in rows:
            meta = row.get("meta") or {}
            for sys_field in ("__id__", self.embedding_node_property, self.text_node_property):
                meta.pop(sys_field, None)
            text = row.get("text") or ""
            doc_id = row.get("id")
            docs.append(Document(id=doc_id, page_content=text, metadata=meta))
        return docs

    async def aclose(self) -> None:
        """Close the async connection (if any)."""
        if self._aconn is not None and not self._aconn.closed:
            await self._aconn.close()
            self._aconn = None

    def close(self) -> None:
        """Close the sync connection if this store owns it. Idempotent.

        When the store was built from a shared ``AgensGraph`` (``graph=``) the
        connection belongs to that graph and is left open.
        """
        if (
            getattr(self, "_owns_connection", False)
            and getattr(self, "connection", None) is not None
            and not self.connection.closed
        ):
            self.connection.close()

    def __enter__(self) -> "AgensgraphVector":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    async def __aenter__(self) -> "AgensgraphVector":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()
        self.close()
