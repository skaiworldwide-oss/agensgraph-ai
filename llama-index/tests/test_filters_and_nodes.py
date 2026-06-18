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

import os

import pytest

from llama_index.core.graph_stores.types import EntityNode
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import (
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
)

from llama_index_agensgraph.graph_stores.agensgraph import AgensPropertyGraphStore
from llama_index_agensgraph.vector_stores.agensgraph import AgensgraphVectorStore

agens_db = os.environ.get("AGENS_DB")
agens_user = os.environ.get("AGENS_USER")
agens_password = os.environ.get("AGENS_PASSWORD")
agens_host = os.environ.get("AGENS_HOST") or "localhost"
agens_port = os.environ.get("AGENS_PORT") or 5432

agens_available = bool(agens_db and agens_user and agens_password)

pytestmark = pytest.mark.skipif(
    not agens_available,
    reason="Requires AGENS_DB, AGENS_USER and AGENS_PASSWORD environment variables.",
)


def _conf():
    return {
        "dbname": agens_db,
        "user": agens_user,
        "password": agens_password,
        "host": agens_host,
        "port": agens_port,
    }


def _url() -> str:
    return (
        f"postgresql://{agens_user}:{agens_password}"
        f"@{agens_host}:{agens_port}/{agens_db}"
    )


def F(key, value, op):
    return MetadataFilter(key=key, value=value, operator=op)


def MF(filters, condition=FilterCondition.AND):
    return MetadataFilters(filters=filters, condition=condition)


# --------------------------------------------------------------------------- #
# Vector store: full operator coverage + node management
# --------------------------------------------------------------------------- #


@pytest.fixture()
def vec() -> AgensgraphVectorStore:
    store = AgensgraphVectorStore(
        url=_url(),
        embedding_dimension=4,
        graph_name="test_filters_vec",
        node_label="Chunk",
    )
    store.clear()
    store.add(
        [
            TextNode(text="alpha", embedding=[1.0, 0.0, 0.0, 0.0],
                     metadata={"topic": "a", "rank": 10, "tags": ["x", "y"], "bio": "Hello World"}),
            TextNode(text="beta", embedding=[0.0, 1.0, 0.0, 0.0],
                     metadata={"topic": "b", "rank": 20, "tags": ["y", "z"], "bio": "goodbye"}),
            TextNode(text="gamma", embedding=[0.0, 0.0, 1.0, 0.0],
                     metadata={"topic": "a", "rank": 30, "tags": [], "bio": ""}),
        ]
    )
    return store


def _texts(store, filters):
    res = store.query(
        VectorStoreQuery(
            query_embedding=[0.3, 0.3, 0.3, 0.0], similarity_top_k=10, filters=filters
        )
    )
    return sorted(n.get_content() for n in res.nodes)


@pytest.mark.parametrize(
    "filters,expected",
    [
        (MF([F("topic", "a", FilterOperator.EQ)]), ["alpha", "gamma"]),
        (MF([F("topic", "a", FilterOperator.NE)]), ["beta"]),
        (MF([F("rank", 20, FilterOperator.GT)]), ["gamma"]),
        (MF([F("rank", 20, FilterOperator.GTE)]), ["beta", "gamma"]),
        (MF([F("rank", 20, FilterOperator.LT)]), ["alpha"]),
        (MF([F("rank", 20, FilterOperator.LTE)]), ["alpha", "beta"]),
        (MF([F("topic", ["a", "b"], FilterOperator.IN)]), ["alpha", "beta", "gamma"]),
        (MF([F("topic", ["a"], FilterOperator.NIN)]), ["beta"]),
        (MF([F("bio", "World", FilterOperator.CONTAINS)]), ["alpha"]),
        (MF([F("bio", "hello", FilterOperator.TEXT_MATCH_INSENSITIVE)]), ["alpha"]),
        (MF([F("tags", ["z", "w"], FilterOperator.ANY)]), ["beta"]),
        (MF([F("tags", ["x", "y"], FilterOperator.ALL)]), ["alpha"]),
        (MF([F("tags", None, FilterOperator.IS_EMPTY)]), ["gamma"]),
    ],
)
def test_vector_store_all_operators(vec, filters, expected):
    assert _texts(vec, filters) == expected


def test_vector_store_and_or_not(vec):
    # AND
    assert _texts(
        vec, MF([F("topic", "a", FilterOperator.EQ), F("rank", 15, FilterOperator.GT)])
    ) == ["gamma"]
    # OR
    assert _texts(
        vec,
        MF([F("topic", "b", FilterOperator.EQ), F("rank", 25, FilterOperator.GT)],
           FilterCondition.OR),
    ) == ["beta", "gamma"]
    # NOT
    assert _texts(vec, MF([F("topic", "a", FilterOperator.EQ)], FilterCondition.NOT)) == ["beta"]


def test_vector_store_nested_filters(vec):
    # (topic=a OR topic=b) AND rank >= 30
    nested = MF(
        [
            MF([F("topic", "a", FilterOperator.EQ), F("topic", "b", FilterOperator.EQ)],
               FilterCondition.OR),
            F("rank", 30, FilterOperator.GTE),
        ],
        FilterCondition.AND,
    )
    assert _texts(vec, nested) == ["gamma"]


def test_vector_store_nin_regression(vec):
    """NIN previously emitted invalid `NOT IN` Cypher; it must now work."""
    assert _texts(vec, MF([F("topic", ["a"], FilterOperator.NIN)])) == ["beta"]


def test_vector_store_get_delete_clear(vec):
    all_nodes = vec.get_nodes()  # no args -> all
    assert {n.get_content() for n in all_nodes} == {"alpha", "beta", "gamma"}

    # get by filter
    topic_a = vec.get_nodes(filters=MF([F("topic", "a", FilterOperator.EQ)]))
    assert {n.get_content() for n in topic_a} == {"alpha", "gamma"}

    # get by id
    one_id = topic_a[0].node_id
    assert [n.node_id for n in vec.get_nodes(node_ids=[one_id])] == [one_id]

    # delete by filter
    vec.delete_nodes(filters=MF([F("topic", "a", FilterOperator.EQ)]))
    assert {n.get_content() for n in vec.get_nodes()} == {"beta"}

    # clear
    vec.clear()
    assert vec.get_nodes() == []


@pytest.mark.asyncio
async def test_vector_store_async_node_mgmt(vec):
    nodes = await vec.aget_nodes(filters=MF([F("topic", "a", FilterOperator.EQ)]))
    assert {n.get_content() for n in nodes} == {"alpha", "gamma"}
    await vec.adelete_nodes(filters=MF([F("topic", "a", FilterOperator.EQ)]))
    assert {n.get_content() for n in await vec.aget_nodes()} == {"beta"}
    await vec.aclear()
    assert await vec.aget_nodes() == []


# --------------------------------------------------------------------------- #
# Property graph store: filtered vector_query
# --------------------------------------------------------------------------- #


@pytest.fixture()
def pg() -> AgensPropertyGraphStore:
    store = AgensPropertyGraphStore(
        "test_filters_pg", conf=_conf(), vector_dimension=4, create=True
    )
    store.structured_query("MATCH (n) DETACH DELETE n")
    store.upsert_nodes(
        [
            EntityNode(name="alice", label="PERSON",
                       properties={"embedding": [1.0, 0.0, 0.0, 0.0], "country": "FR", "age": 30}),
            EntityNode(name="bob", label="PERSON",
                       properties={"embedding": [0.0, 1.0, 0.0, 0.0], "country": "US", "age": 40}),
            EntityNode(name="carol", label="PERSON",
                       properties={"embedding": [0.0, 0.0, 1.0, 0.0], "country": "FR", "age": 50}),
        ]
    )
    return store


def _names(store, filters):
    nodes, _ = store.vector_query(
        VectorStoreQuery(
            query_embedding=[0.3, 0.3, 0.3, 0.0], similarity_top_k=10, filters=filters
        )
    )
    return sorted(n.name for n in nodes)


def test_pg_filtered_vector_query(pg):
    assert _names(pg, None) == ["alice", "bob", "carol"]
    assert _names(pg, MF([F("country", "FR", FilterOperator.EQ)])) == ["alice", "carol"]
    assert _names(pg, MF([F("age", 35, FilterOperator.GT)])) == ["bob", "carol"]
    assert _names(
        pg, MF([F("country", "FR", FilterOperator.EQ), F("age", 40, FilterOperator.GT)])
    ) == ["carol"]
    assert _names(pg, MF([F("country", ["US"], FilterOperator.IN)])) == ["bob"]
    assert _names(pg, MF([F("country", "FR", FilterOperator.EQ)], FilterCondition.NOT)) == ["bob"]


@pytest.mark.asyncio
async def test_pg_filtered_avector_query(pg):
    nodes, _ = await pg.avector_query(
        VectorStoreQuery(
            query_embedding=[0.3, 0.3, 0.3, 0.0],
            similarity_top_k=10,
            filters=MF([F("country", "FR", FilterOperator.EQ)]),
        )
    )
    assert sorted(n.name for n in nodes) == ["alice", "carol"]


# --------------------------------------------------------------------------- #
# Property graph store: deepened enhanced schema
# --------------------------------------------------------------------------- #


def test_enhanced_schema_stats():
    store = AgensPropertyGraphStore(
        "test_enhanced_stats", conf=_conf(), enhanced_schema=True, create=True
    )
    store.structured_query("MATCH (n) DETACH DELETE n")
    store.upsert_nodes(
        [
            EntityNode(name="alice", label="PERSON",
                       properties={"age": 30, "country": "FR", "tags": ["x", "y"]}),
            EntityNode(name="bob", label="PERSON",
                       properties={"age": 40, "country": "US", "tags": ["z"]}),
            EntityNode(name="carol", label="PERSON",
                       properties={"age": 50, "country": "FR", "tags": []}),
        ]
    )
    store.refresh_schema()
    props = {p["property"]: p for p in store.structured_schema["node_props"]["PERSON"]}

    # numeric -> min/max/distinct_count
    assert props["age"]["min"] == 30.0
    assert props["age"]["max"] == 50.0
    assert props["age"]["distinct_count"] == 3

    # list -> min_size/max_size
    assert props["tags"]["min_size"] == 0
    assert props["tags"]["max_size"] == 2

    # string -> example values + distinct_count
    assert set(props["country"]["values"]) == {"FR", "US"}
    assert props["country"]["distinct_count"] == 2

    schema_str = store.get_schema_str()
    assert "min: 30.0" in schema_str and "max: 50.0" in schema_str
    assert "list size" in schema_str


def test_default_schema_has_no_stats():
    """Without enhanced_schema, no value materialization / stats are attached."""
    store = AgensPropertyGraphStore("test_plain_stats", conf=_conf(), create=True)
    store.structured_query("MATCH (n) DETACH DELETE n")
    store.upsert_nodes([EntityNode(name="x", label="PERSON", properties={"age": 1})])
    store.refresh_schema()
    for p in store.structured_schema["node_props"].get("PERSON", []):
        assert "values" not in p and "min" not in p and "min_size" not in p
