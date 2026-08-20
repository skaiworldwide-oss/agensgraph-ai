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

import agensgraph
import pytest
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
)

from llama_index_agensgraph.vector_stores.agensgraph import AgensgraphVectorStore

agens_db = os.environ.get("AGENS_DB")
agens_user = os.environ.get("AGENS_USER")
agens_password = os.environ.get("AGENS_PASSWORD")
agens_host = os.environ.get("AGENS_HOST") or "localhost"
agens_port = os.environ.get("AGENS_PORT") or 5432

pytestmark = pytest.mark.skipif(
    not (agens_db and agens_user and agens_password),
    reason="Requires AGENS_DB, AGENS_USER and AGENS_PASSWORD environment variables.",
)


def _url():
    return (
        f"postgresql://{agens_user}:{agens_password}@{agens_host}:{agens_port}/{agens_db}"
    )


def _drop(graph):
    conn = agensgraph.Connection.connect(
        host=agens_host, port=agens_port, dbname=agens_db,
        user=agens_user, password=agens_password, autocommit=True,
    )
    conn.execute(f"DROP GRAPH IF EXISTS {graph} CASCADE")
    conn.close()


# a and b are nearly the same direction; c and d are far from both and from
# each other, which is what lets MMR be told apart from plain relevance.
CORPUS = [
    TextNode(id_="a", text="the cat sat on the mat",
             embedding=[1.0, 0.0, 0.0, 0.0], metadata={"g": "x"}),
    TextNode(id_="b", text="a dog barked loudly",
             embedding=[0.99, 0.1, 0.0, 0.0], metadata={"g": "x"}),
    TextNode(id_="c", text="cat and dog together",
             embedding=[0.0, 1.0, 0.0, 0.0], metadata={"g": "y"}),
    TextNode(id_="d", text="quantum entanglement",
             embedding=[0.0, 0.0, 1.0, 0.0], metadata={"g": "y"}),
]


@pytest.fixture(scope="module")
def hybrid_store():
    _drop("test_query_modes")
    store = AgensgraphVectorStore(
        url=_url(), embedding_dimension=4, graph_name="test_query_modes",
        hybrid_search=True,
    )
    store.add(CORPUS)
    return store


def _ids(store, **kw):
    query = VectorStoreQuery(
        query_embedding=[1.0, 0.0, 0.0, 0.0], similarity_top_k=3, **kw
    )
    return store.query(query).ids


def test_a_mode_this_store_cannot_answer_is_refused(hybrid_store):
    """Every mode used to be answered as a plain vector search, because the field
    was never read -- so asking for a classifier got cosine distance and no word
    about it."""
    with pytest.raises(ValueError, match="not a mode this store answers"):
        _ids(hybrid_store, mode=VectorStoreQueryMode.SVM)


def test_a_text_mode_without_text_is_refused(hybrid_store):
    for mode in (
        VectorStoreQueryMode.HYBRID,
        VectorStoreQueryMode.TEXT_SEARCH,
        VectorStoreQueryMode.SPARSE,
    ):
        with pytest.raises(ValueError, match="query_str"):
            _ids(hybrid_store, mode=mode)


def test_hybrid_on_a_store_without_a_text_index_is_refused():
    _drop("test_modes_novector")
    store = AgensgraphVectorStore(
        url=_url(), embedding_dimension=4, graph_name="test_modes_novector",
    )
    store.add(CORPUS[:1])
    with pytest.raises(ValueError, match="hybrid_search=True"):
        _ids(store, mode=VectorStoreQueryMode.HYBRID, query_str="cat")


def test_text_search_finds_by_word_not_by_distance(hybrid_store):
    """'quantum' is nowhere near the query embedding, and is the only match."""
    assert _ids(hybrid_store, mode=VectorStoreQueryMode.TEXT_SEARCH,
                query_str="quantum") == ["d"]


def test_alpha_moves_the_answer_between_the_two_sides(hybrid_store):
    vector_only = _ids(hybrid_store, mode=VectorStoreQueryMode.HYBRID,
                       query_str="quantum", alpha=1.0)
    text_only = _ids(hybrid_store, mode=VectorStoreQueryMode.HYBRID,
                     query_str="quantum", alpha=0.0)
    # The nearest by distance does not mention quantum; the only one that does is
    # far away. Weighing one side or the other has to change which comes first.
    assert vector_only[0] != text_only[0]
    assert text_only[0] == "d"


def test_hybrid_takes_a_filter(hybrid_store):
    """Asked for together these used to raise, saying filtering does not work
    with hybrid search."""
    only_y = MetadataFilters(
        filters=[MetadataFilter(key="g", value="y", operator=FilterOperator.EQ)]
    )
    got = _ids(hybrid_store, mode=VectorStoreQueryMode.HYBRID,
               query_str="cat", filters=only_y)
    assert got and set(got) <= {"c", "d"}


def test_hybrid_top_k_bounds_the_fused_answer(hybrid_store):
    assert len(_ids(hybrid_store, mode=VectorStoreQueryMode.HYBRID,
                    query_str="cat", hybrid_top_k=2)) == 2


def test_mmr_drops_a_near_duplicate_for_something_further_off(hybrid_store):
    """b points almost where a does. Ranked by relevance alone it comes second;
    told to avoid what is already chosen, something else does."""
    by_relevance = _ids(hybrid_store, mode=VectorStoreQueryMode.MMR,
                        mmr_threshold=1.0)
    diverse = _ids(hybrid_store, mode=VectorStoreQueryMode.MMR, mmr_threshold=0.1)
    assert by_relevance[:2] == ["a", "b"]
    assert diverse[0] == "a"
    assert diverse[1] != "b"


def test_mmr_does_not_hand_back_the_embedding_it_fetched(hybrid_store):
    query = VectorStoreQuery(
        query_embedding=[1.0, 0.0, 0.0, 0.0], similarity_top_k=2,
        mode=VectorStoreQueryMode.MMR, mmr_threshold=0.5,
    )
    result = hybrid_store.query(query)
    for node in result.nodes:
        assert "mmr_embedding" not in (node.metadata or {})
        assert "embedding" not in (node.metadata or {})


@pytest.mark.parametrize(
    ("strategy", "operator", "operator_class"),
    [
        ("cosine", "<=>", "vector_cosine_ops"),
        ("l2", "<->", "vector_l2_ops"),
        ("euclidean", "<->", "vector_l2_ops"),
        ("inner_product", "<#>", "vector_ip_ops"),
    ],
)
def test_the_index_is_built_for_the_distance_the_query_asks_for(
    strategy, operator, operator_class
):
    """They have to agree. An index whose operator class does not match the
    operator in the query cannot serve the ordering, and the search reads every
    row with nothing to say so."""
    graph = f"test_modes_{strategy}"
    _drop(graph)
    store = AgensgraphVectorStore(
        url=_url(), embedding_dimension=4, graph_name=graph,
        distance_strategy=strategy,
    )
    store.add(CORPUS)
    statement, _ = store._build_query(
        VectorStoreQuery(query_embedding=[1.0, 0.0, 0.0, 0.0], similarity_top_k=2)
    )
    assert operator in statement.as_string(store._connection)
    definitions = [
        index.definition
        for index in store._connection.indexes("Chunk", graph=graph)
        if "hnsw" in index.definition
    ]
    assert definitions and operator_class in definitions[0]


def test_an_unknown_distance_is_refused():
    with pytest.raises(ValueError, match="not a distance this store measures"):
        AgensgraphVectorStore(
            url=_url(), embedding_dimension=4, graph_name="test_modes_bad",
            distance_strategy="manhattan",
        )


def test_the_index_takes_build_options():
    graph = "test_modes_options"
    _drop(graph)
    store = AgensgraphVectorStore(
        url=_url(), embedding_dimension=4, graph_name=graph,
        index_options={"m": 32, "ef_construction": 128},
        search_options={"hnsw.ef_search": 120},
    )
    store.add(CORPUS)
    options = store._connection.execute(
        "select i.reloptions from pg_class i join pg_index x on x.indexrelid = i.oid"
        " join pg_class t on t.oid = x.indrelid"
        " join pg_namespace n on n.oid = t.relnamespace"
        " where n.nspname = %s and i.relname = %s",
        (graph, store.index_name),
    ).fetchone()[0]
    store._connection.commit()
    assert sorted(options) == ["ef_construction=128", "m=32"]
    # and a search still answers with the tuning applied
    assert store.query(
        VectorStoreQuery(query_embedding=[1.0, 0.0, 0.0, 0.0], similarity_top_k=2)
    ).ids
