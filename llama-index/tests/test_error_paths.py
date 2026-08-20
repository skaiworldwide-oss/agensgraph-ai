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
import psycopg
import pytest
from llama_index.core.graph_stores.types import EntityNode
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import VectorStoreQuery

from llama_index_agensgraph.graph_stores.agensgraph import AgensPropertyGraphStore
from llama_index_agensgraph.graph_stores.agensgraph.utils import AgensQueryException
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

GRAPH = "test_error_paths"


def _conf(**over):
    return {
        "dbname": agens_db, "user": agens_user, "password": agens_password,
        "host": agens_host, "port": agens_port, **over,
    }


def _url():
    return (
        f"postgresql://{agens_user}:{agens_password}@{agens_host}:{agens_port}/{agens_db}"
    )


@pytest.fixture(scope="module")
def store():
    s = AgensPropertyGraphStore(GRAPH, conf=_conf(), create=True, vector_dimension=4)
    s.structured_query("MATCH (n) DETACH DELETE n")
    return s


def test_bad_cypher_says_what_the_server_said(store):
    """Every failure read back as "unknown": the reader asked for "details" and
    all four raise sites wrote "detail"."""
    with pytest.raises(AgensQueryException) as raised:
        store.structured_query("MATCH (n) RETRUN n")
    details = str(raised.value.get_details())
    assert "42601" in details, details
    assert "syntax error" in details, details


def test_a_failure_carries_the_cause_it_came_from(store):
    """So a caller can ask the driver whether another attempt is worth making."""
    with pytest.raises(AgensQueryException) as raised:
        store.structured_query("MATCH (n) RETRUN n")
    cause = raised.value.__cause__
    assert isinstance(cause, psycopg.Error)
    assert cause.sqlstate == "42601"
    assert raised.value.cause is cause


def test_the_row_that_failed_is_not_put_into_the_message(store):
    """These results are handed to a language model, and DETAIL carries the row.

    Written through the store the label carries its own uniqueness on id, so a
    second element with that id is refused -- and what the server says about it
    names the constraint, not the contents.
    """
    store.upsert_nodes(
        [
            EntityNode(
                label="SECRET",
                name="only-one",
                properties={"tell": "correct-horse-battery-staple"},
            )
        ]
    )
    with pytest.raises(AgensQueryException) as raised:
        store.structured_query(
            'CREATE (:"SECRET" {id: %(i)s, tell: %(t)s})',
            param_map={"i": "only-one", "t": "correct-horse-battery-staple"},
        )
    details = str(raised.value.get_details())
    assert "correct-horse-battery-staple" not in details, details
    assert "23" in details.split(":")[0], details
    store.structured_query('MATCH (n:"SECRET") DETACH DELETE n')


def test_a_statement_the_connection_cannot_carry_still_reports_itself(store):
    """A connection the server has already dropped cannot be rolled back, and
    saying so must not replace the real reason."""
    other = AgensPropertyGraphStore(
        GRAPH, conf=_conf(), create=False, create_indexes=False, refresh_schema=False
    )
    other.connection.close()
    with pytest.raises(Exception) as raised:
        other.structured_query("RETURN 1 AS one")
    assert "rollback" not in str(raised.value).lower()


def test_the_connection_is_usable_after_a_refusal(store):
    """A refused statement leaves its transaction able to run nothing else, so
    without a rollback every later statement reported the abort instead."""
    for _ in range(3):
        with pytest.raises(AgensQueryException):
            store.structured_query("MATCH (n) RETRUN n")
    assert store.structured_query("RETURN 1 AS one")[0]["one"] == 1


def test_an_embedding_of_the_wrong_width_is_refused(store):
    with pytest.raises(AgensQueryException):
        store.upsert_nodes(
            [
                EntityNode(
                    label="WRONGDIM",
                    name="too-short",
                    properties={"embedding": [1.0, 2.0]},
                )
            ]
        )


def test_a_graph_that_is_not_there_and_may_not_be_made(store):
    with pytest.raises(Exception, match="does not exist"):
        AgensPropertyGraphStore(
            "test_error_paths_absent", conf=_conf(), create=False
        )


def test_a_vector_store_node_with_no_embedding_is_refused():
    """LlamaIndex lets a node carry no embedding; this store indexes on it."""
    conn = agensgraph.Connection.connect(autocommit=True, **_conf())
    conn.execute("DROP GRAPH IF EXISTS test_error_paths_vs CASCADE")
    conn.close()
    vs = AgensgraphVectorStore(
        url=_url(), embedding_dimension=4, graph_name="test_error_paths_vs"
    )
    with pytest.raises(Exception):
        vs.add([TextNode(id_="no-embedding", text="nothing to search by")])


def test_a_search_before_anything_is_written_answers_with_nothing():
    conn = agensgraph.Connection.connect(autocommit=True, **_conf())
    conn.execute("DROP GRAPH IF EXISTS test_error_paths_empty CASCADE")
    conn.close()
    vs = AgensgraphVectorStore(
        url=_url(), embedding_dimension=4, graph_name="test_error_paths_empty"
    )
    result = vs.query(
        VectorStoreQuery(query_embedding=[1.0, 0.0, 0.0, 0.0], similarity_top_k=5)
    )
    assert result.ids == []


def test_a_string_parameter_arrives_as_a_string(store):
    """An unspecified parameter is read as JSON, so '123' matched nothing --
    which is what CypherTemplateRetriever passes, a flat dict of strings."""
    store.upsert_nodes([EntityNode(label="NUMERIC", name="123")])
    rows = store.structured_query(
        'MATCH (n:"NUMERIC") WHERE n.name = %(name)s RETURN n.name AS name',
        param_map={"name": "123"},
    )
    assert [r["name"] for r in rows] == ["123"]


def test_param_map_and_params_are_the_same_argument(store):
    """The contract calls it param_map and this class called it params, so
    CypherTemplateRetriever -- which passes it by keyword -- raised TypeError."""
    statement = "RETURN %(n)s AS given"
    by_contract = store.structured_query(statement, param_map={"n": "x"})
    by_old_name = store.structured_query(statement, params={"n": "x"})
    positional = store.structured_query(statement, {"n": "x"})
    assert by_contract == by_old_name == positional


def test_two_entities_of_the_same_name_on_different_labels_are_both_kept(store):
    """LlamaIndex gives an EntityNode its name as its id, so these two share one
    -- the label is what tells them apart.

    Each is written on the label naming what it is, so they are two elements with
    their own properties, and uniqueness on id holds within a label rather than
    across the graph. A read by that id has to answer with both: dropping one
    would lose an entity the writer asked to keep.
    """
    a = EntityNode(label="ACTOR", name="Ford", properties={"trade": "acting"})
    b = EntityNode(label="CARMAKER", name="Ford", properties={"trade": "cars"})
    assert a.id == b.id == "Ford"
    store.upsert_nodes([a, b])

    got = store.get(ids=["Ford"])
    assert len(got) == 2, [n.label for n in got]
    assert {n.label for n in got} == {"ACTOR", "CARMAKER"}
    assert {n.properties["trade"] for n in got} == {"acting", "cars"}

    # And writing one of them again finds it rather than making a third.
    store.upsert_nodes(
        [EntityNode(label="ACTOR", name="Ford", properties={"trade": "acting still"})]
    )
    again = store.get(ids=["Ford"])
    assert len(again) == 2
    assert {n.properties["trade"] for n in again} == {"acting still", "cars"}
    store.delete(ids=["Ford"])


def test_a_label_outside_the_base_keeps_its_embedding_in_the_map(store):
    """Inheriting the base is what carries the promoted column down.

    A label the graph already held, written by something else, does not inherit
    it -- and indexed as though it had the column the server refuses outright:
    "operator class vector_cosine_ops does not accept data type jsonb".
    """
    store.structured_query('CREATE VLABEL IF NOT EXISTS "OUTSIDER"')
    store.structured_query(
        'CREATE (:"OUTSIDER" {id: %(i)s, embedding: %(e)s})',
        param_map={"i": "x", "e": [1.0, 0.0, 0.0, 0.0]},
    )
    assert store._promoted() is True, "the base itself should have the column"
    assert store._promoted_label("OUTSIDER") is False

    store._create_vector_index(["OUTSIDER"])
    definitions = [
        index.definition
        for index in store.connection.indexes("OUTSIDER", graph=GRAPH)
        if "hnsw" in index.definition
    ]
    store.connection.commit()
    assert definitions and "::vector(4)" in definitions[0], definitions
    store.structured_query('MATCH (n:"OUTSIDER") DETACH DELETE n')


def test_a_chunk_is_written_in_one_pass(store):
    """Setting the text, then the properties, then the embedding was three passes
    over the same row.

    Three tuple versions per chunk, each of them indexed -- and on a real corpus
    the server refused the second update outright with "attempted to delete
    invisible tuple" (55000), which is what stopped this package's own README
    example from running. One map, one write.
    """
    from llama_index.core.graph_stores.types import ChunkNode

    ops = store._build_upsert_nodes_ops(
        [
            ChunkNode(
                id_="one-pass",
                text="body",
                embedding=[1.0, 0.0, 0.0, 0.0],
                properties={"file_name": "f", "_node_content": '{"a": 1}'},
            )
        ]
    )
    chunk_ops = [
        (query, params)
        for query, params in ops
        if "Chunk" in query.as_string(store.connection)
    ]
    assert chunk_ops, [q.as_string(store.connection) for q, _ in ops]
    text = chunk_ops[0][0].as_string(store.connection)
    assert text.count("SET") == 1, text

    # and it writes what it was given
    store.upsert_nodes(
        [
            ChunkNode(
                id_="one-pass",
                text="body",
                embedding=[1.0, 0.0, 0.0, 0.0],
                properties={"file_name": "f"},
            )
        ]
    )
    got = store.get(ids=["one-pass"])
    assert len(got) == 1
    assert got[0].text == "body"
    assert got[0].properties["file_name"] == "f"


def test_a_node_type_it_cannot_write_is_reported(store, caplog):
    """Twenty TextNodes passed to upsert_nodes went missing without a word."""
    import logging

    from llama_index.core.schema import TextNode

    with caplog.at_level(logging.WARNING):
        store.upsert_nodes([TextNode(id_="dropped", text="nowhere")])
    assert any("upsert_llama_nodes" in r.getMessage() for r in caplog.records), [
        r.getMessage() for r in caplog.records
    ]
