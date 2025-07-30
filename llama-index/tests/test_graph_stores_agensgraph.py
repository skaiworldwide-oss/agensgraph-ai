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

import pytest
import os

from llama_index.core.graph_stores.types import GraphStore
from llama_index_agensgraph.graph_stores.agensgraph import AgensGraphStore

agens_db = os.environ.get("AGENS_DB")
agens_user = os.environ.get("AGENS_USER")
agens_password = os.environ.get("AGENS_PASSWORD")
agens_host = os.environ.get("AGENS_HOST") or "localhost"
agens_port = os.environ.get("AGENS_PORT") or 5432

if not agens_db or not agens_user or not agens_password:
    agens_available = False
else:
    agens_available = True

pytestmark = pytest.mark.skipif(
    not agens_available,
    reason="""
        Requires AGENS_DB, AGENS_USER and AGENS_PASSWORD environment variables.\n
        AGENS_HOST and AGENS_PORT defaults to localhost and 5432 if not provided.
    """,
)


@pytest.fixture()
def agens_graph_store() -> AgensGraphStore:
    """
    Provides a fresh AgensGraphStore for each test.
    Adjust parameters to match your test database or local AgensGraph setup.
    """
    conf = {
        "dbname": agens_db,
        "user": agens_user,
        "password": agens_password,
        "host": agens_host,
        "port": agens_port,
    }
    agens_graph_store = AgensGraphStore("test", conf=conf, create=True)
    # Teardown: remove any remaining data to avoid polluting tests
    # For a small test DB, you can delete all nodes & relationships:
    agens_graph_store.query("MATCH (n) DETACH DELETE n")
    return agens_graph_store


def test_agens_graph_store():
    names_of_bases = [b.__name__ for b in AgensGraphStore.__bases__]
    assert GraphStore.__name__ in names_of_bases


def test_01_connection_init(agens_graph_store: AgensGraphStore):
    """
    Test initial connection and constraint creation.
    Verifies that the store is connected and schema can be fetched.
    """
    assert agens_graph_store is not None
    schema_str = agens_graph_store.get_schema(refresh=True)
    # We don't necessarily expect non-empty schema if DB is brand-new,
    # but we at least can check that it's a string.
    assert isinstance(schema_str, str)


def test_02_upsert_and_get(agens_graph_store: AgensGraphStore):
    """
    Test inserting triplets and retrieving them.
    """
    # Insert a simple triplet: Alice -> LIKES -> IceCream
    agens_graph_store.upsert_triplet("Alice", "LIKES", "IceCream")

    # Retrieve edges from 'Alice'
    results = agens_graph_store.get("Alice")
    assert len(results) == 1
    rel_type, obj = results[0]
    assert rel_type == "LIKES"
    assert obj == "IceCream"


def test_03_upsert_multiple_and_get(agens_graph_store: AgensGraphStore):
    """
    Insert multiple triplets for a single subject.
    """
    # Add two different relationships from 'Alice'
    agens_graph_store.upsert_triplet("Alice", "LIKES", "IceCream")
    agens_graph_store.upsert_triplet("Alice", "DISLIKES", "Spinach")

    results = agens_graph_store.get("Alice")
    # Expect two relationships
    assert len(results) == 2
    rels = {rel[0] for rel in results}
    objs = {rel[1] for rel in results}
    assert rels == {"LIKES", "DISLIKES"}
    assert objs == {"IceCream", "Spinach"}


def test_04_get_rel_map(agens_graph_store: AgensGraphStore):
    """
    Test get_rel_map with multi-hop relationships.
    """
    # Insert:
    #   Alice -> KNOWS -> Bob -> LIVES_IN -> CityX
    #   Bob -> TRAVELED_TO -> CityY
    store = agens_graph_store
    store.upsert_triplet("Alice", "KNOWS", "Bob")
    store.upsert_triplet("Bob", "LIVES_IN", "CityX")
    store.upsert_triplet("Bob", "TRAVELED_TO", "CityY")

    # Depth 2 from 'Alice' should see: (KNOWS->Bob) + (Bob->LIVES_IN->CityX) + (Bob->TRAVELED_TO->CityY)
    rel_map = store.get_rel_map(["Alice"], depth=2, limit=30)

    assert "Alice" in rel_map
    # Flattened relationships are a bit tricky; we only check that something is returned
    # The structure is like lists of [relType, objectId].
    flattened_rels = rel_map["Alice"]
    assert len(flattened_rels) > 0


def test_05_delete_relationship_and_nodes(agens_graph_store: AgensGraphStore):
    """
    Test deleting an existing relationship (and subject/object if no other edges).
    """
    store = agens_graph_store
    store.upsert_triplet("X", "REL", "Y")

    # Confirm upsert worked
    results_before = store.get("X")
    assert len(results_before) == 1
    assert results_before[0] == ["REL", "Y"]

    # Delete that relationship
    store.delete("X", "REL", "Y")

    # Now both X and Y should be removed if no other edges remain.
    results_after = store.get("X")
    assert len(results_after) == 0


def test_06_delete_keeps_node_if_other_edges_exist(agens_graph_store: AgensGraphStore):
    """
    Test that only the specified relationship is removed,
    and the subject/object are deleted only if they have no other edges.
    """
    store = agens_graph_store
    # Insert two edges: X->REL->Y and X->OTHER->Z
    store.upsert_triplet("X", "REL", "Y")
    store.upsert_triplet("X", "OTHER", "Z")

    # Delete the first relationship
    store.delete("X", "REL", "Y")

    # 'Y' should be gone if no other edges, but X must remain (it still has an edge to Z).
    # Confirm X->OTHER->Z still exists
    results_x = store.get("X")
    assert len(results_x) == 1
    assert results_x[0] == ["OTHER", "Z"]

    # 'Y' should have 0 edges
    results_y = store.get("Y")
    assert len(results_y) == 0


def test_07_refresh_schema(agens_graph_store: AgensGraphStore):
    """
    Test the refresh_schema call.
    """
    store = agens_graph_store
    # Insert a couple triplets
    store.upsert_triplet("A", "TEST_REL", "B")
    # Refresh schema
    store.refresh_schema()
    structured = store.structured_schema
    assert "node_props" in structured
    assert "rel_props" in structured
    assert "relationships" in structured


def test_08_get_schema(agens_graph_store: AgensGraphStore):
    """
    Test get_schema with and without refresh.
    """
    store = agens_graph_store
    # Possibly empty if no data, but we can at least confirm it doesn't error
    schema_str_1 = store.get_schema(refresh=False)
    assert isinstance(schema_str_1, str)

    # Add data
    store.upsert_triplet("Person1", "LIKES", "Thing1")
    # Now refresh
    schema_str_2 = store.get_schema(refresh=True)
    assert isinstance(schema_str_2, str)
    # The new schema might mention 'LIKES' or node labels, but that depends on your DB.
    # You can do a substring check if you expect it:
    # assert "LIKES" in schema_str_2


def test_09_custom_query(agens_graph_store: AgensGraphStore):
    """
    Test running a direct custom Cypher query via .query().
    """
    store = agens_graph_store
    store.upsert_triplet("TestS", "TEST_REL", "TestO")

    # Custom query to find all nodes that have an outgoing relationship
    custom_cypher = """
    MATCH (n)-[r]->(m)
    RETURN n.id AS subject, type(r) AS relation, m.id AS object
    """
    results = store.query(custom_cypher)
    assert len(results) >= 1
    print(results)
    # Expect at least the one we inserted
    expected = {"subject": "TestS", "relation": "TEST_REL", "object": "TestO"}
    assert any(
        row["subject"] == expected["subject"]
        and row["relation"] == expected["relation"]
        and row["object"] == expected["object"]
        for row in results
    ), "Custom query did not return the inserted relationship."