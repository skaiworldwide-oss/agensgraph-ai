import os
import pytest

from llama_index.graph_stores.agensgraph import AgensPropertyGraphStore
from llama_index.core.graph_stores.types import EntityNode, ChunkNode, Relation
from llama_index.core.vector_stores.types import VectorStoreQuery
from llama_index.core.schema import TextNode


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
def agens_store() -> AgensPropertyGraphStore:
    print("Creating AgensPropertyGraphStore fixture")
    if not agens_available:
        pytest.skip("No agensgraph credentials provided")
    
    conf = {
        "database": agens_db,
        "user": agens_user,
        "password": agens_password,
        "host": agens_host,
        "port": agens_port,
    }
    agens_store = AgensPropertyGraphStore("test", conf=conf)
    agens_store.structured_query("MATCH (n) DETACH DELETE n")
    return agens_store


def test_upsert_nodes_and_get(agens_store: AgensPropertyGraphStore):
    """
    Test inserting entity and chunk nodes, then retrieving them.
    """
    entity = EntityNode(label="PERSON", name="Alice")
    chunk = ChunkNode(text="Alice is a software engineer.")
    agens_store.upsert_nodes([entity, chunk])

    # Get by ID
    retrieved_entities = agens_store.get(ids=[entity.id])
    assert len(retrieved_entities) == 1
    assert isinstance(retrieved_entities[0], EntityNode)
    assert retrieved_entities[0].name == "Alice"

    retrieved_chunks = agens_store.get(ids=[chunk.id])
    assert len(retrieved_chunks) == 1
    assert isinstance(retrieved_chunks[0], ChunkNode)
    assert retrieved_chunks[0].text == "Alice is a software engineer."

    # Get by property
    retrieved_by_prop = agens_store.get(properties={"name": "Alice"})
    assert len(retrieved_by_prop) == 1
    assert retrieved_by_prop[0].id == entity.id

    # Attempt to get unknown property
    unknown_prop = agens_store.get(properties={"non_existent_prop": "foo"})
    assert len(unknown_prop) == 0


def test_02_upsert_nodes_and_get_multiple(agens_store: AgensPropertyGraphStore):
    """
    Test inserting multiple nodes at once and retrieving them by IDs.
    """
    entity1 = EntityNode(label="PERSON", name="Bob")
    entity2 = EntityNode(label="PERSON", name="Charlie")
    chunk1 = ChunkNode(text="This is sample text.")
    chunk2 = ChunkNode(text="Another sample text.")

    # Upsert multiple
    agens_store.upsert_nodes([entity1, entity2, chunk1, chunk2])

    # Retrieve by IDs
    ids_to_get = [entity1.id, entity2.id, chunk1.id, chunk2.id]
    results = agens_store.get(ids=ids_to_get)
    assert len(results) == 4

    # Check some known values
    person_bob = [r for r in results if isinstance(r, EntityNode) and r.name == "Bob"]
    assert len(person_bob) == 1

    chunk_texts = [r for r in results if isinstance(r, ChunkNode)]
    assert len(chunk_texts) == 2


def test_03_upsert_relations_and_get(agens_store: AgensPropertyGraphStore):
    """
    Test creating relations between nodes, then retrieving them in multiple ways.
    """
    person = EntityNode(label="PERSON", name="Alice")
    city = EntityNode(label="CITY", name="Paris")
    agens_store.upsert_nodes([person, city])

    # Create a relation
    visited_relation = Relation(
        source_id=person.id,
        target_id=city.id,
        label="VISITED",
        properties={"year": 2023},
    )
    agens_store.upsert_relations([visited_relation])

    # Validate that the relation can be found in triplets
    triplets = agens_store.get_triplets(entity_names=["Alice"])
    assert len(triplets) == 1
    source, rel, target = triplets[0]
    assert source.name == "Alice"
    assert target.name == "Paris"
    assert rel.label == "VISITED"
    assert rel.properties["year"] == 2023


def test_05_filter_nodes_by_property(agens_store: AgensPropertyGraphStore):
    """
    Test get() with property filtering.
    """
    e1 = EntityNode(label="PERSON", name="Alice", properties={"country": "France"})
    e2 = EntityNode(label="PERSON", name="Bob", properties={"country": "USA"})
    e3 = EntityNode(label="PERSON", name="Charlie", properties={"country": "France"})
    agens_store.upsert_nodes([e1, e2, e3])

    # Filter
    filtered = agens_store.get(properties={"country": "France"})
    assert len(filtered) == 2
    filtered_names = {x.name for x in filtered}
    assert filtered_names == {"Alice", "Charlie"}


def test_06_delete_entities_by_names(agens_store: AgensPropertyGraphStore):
    """
    Test deleting nodes by entity_names.
    """
    e1 = EntityNode(label="PERSON", name="Alice")
    e2 = EntityNode(label="PERSON", name="Bob")
    agens_store.upsert_nodes([e1, e2])

    # Delete 'Alice'
    agens_store.delete(entity_names=["Alice"])

    # Verify
    remaining = agens_store.get()
    assert len(remaining) == 1
    assert remaining[0].name == "Bob"


def test_07_delete_nodes_by_ids(agens_store: AgensPropertyGraphStore):
    """
    Test deleting nodes by IDs.
    """
    e1 = EntityNode(label="PERSON", name="Alice")
    e2 = EntityNode(label="PERSON", name="Bob")
    e3 = EntityNode(label="PERSON", name="Charlie")
    agens_store.upsert_nodes([e1, e2, e3])

    # Delete Bob, Charlie by IDs
    agens_store.delete(ids=[e2.id, e3.id])

    all_remaining = agens_store.get()
    assert len(all_remaining) == 1
    assert all_remaining[0].name == "Alice"


def test_08_delete_relations(agens_store: AgensPropertyGraphStore):
    """
    Test deleting relationships by relation names.
    """
    e1 = EntityNode(label="PERSON", name="Alice")
    e2 = EntityNode(label="CITY", name="Paris")
    agens_store.upsert_nodes([e1, e2])

    rel = Relation(source_id=e1.id, target_id=e2.id, label="VISITED")
    agens_store.upsert_relations([rel])

    # Ensure the relationship is there
    triplets_before = agens_store.get_triplets(entity_names=["Alice"])
    assert len(triplets_before) == 1

    # Delete the relation
    agens_store.delete(relation_names=["VISITED"])

    # No more triplets
    triplets_after = agens_store.get_triplets(entity_names=["Alice"])
    assert len(triplets_after) == 0


def test_09_delete_nodes_by_properties(agens_store: AgensPropertyGraphStore):
    """
    Test deleting nodes by a property dict.
    """
    c1 = ChunkNode(text="This is a test chunk.", properties={"lang": "en"})
    c2 = ChunkNode(text="Another chunk.", properties={"lang": "fr"})
    agens_store.upsert_nodes([c1, c2])

    # Delete all English chunks
    agens_store.delete(properties={"lang": "en"})

    # Only c2 remains
    remaining = agens_store.get()
    assert len(remaining) == 1
    assert remaining[0].properties["lang"] == "fr"


def test_10_vector_query(agens_store: AgensPropertyGraphStore):
    """
    Test vector_query with some dummy embeddings.
    Note: This requires pgvector to be installed in the AgensGraph database.
    """
    entity1 = EntityNode(
        label="PERSON", name="Alice", properties={"embedding": [0.1, 0.2, 0.3]}
    )
    entity2 = EntityNode(
        label="PERSON", name="Bob", properties={"embedding": [0.9, 0.8, 0.7]}
    )
    agens_store.upsert_nodes([entity1, entity2])

    # Query embedding somewhat closer to [0.1, 0.2, 0.3] than [0.9, 0.8, 0.7]
    query = VectorStoreQuery(query_embedding=[0.1, 0.2, 0.31], similarity_top_k=2)
    results, scores = agens_store.vector_query(query)

    # Expect "Alice" to come first
    assert len(results) == 2
    names_in_order = [r.name for r in results]
    assert names_in_order[0] == "Alice"
    assert names_in_order[1] == "Bob"
    # Score check: Usually Alice's score should be higher
    assert scores[0] >= scores[1]


def test_11_get_rel_map(agens_store: AgensPropertyGraphStore):
    """
    Test get_rel_map with a multi-depth scenario.
    """
    e1 = EntityNode(label="PERSON", name="Alice")
    e2 = EntityNode(label="PERSON", name="Bob")
    e3 = EntityNode(label="CITY", name="Paris")
    e4 = EntityNode(label="CITY", name="London")
    agens_store.upsert_nodes([e1, e2, e3, e4])

    r1 = Relation(label="KNOWS", source_id=e1.id, target_id=e2.id)
    r2 = Relation(label="VISITED", source_id=e1.id, target_id=e3.id)
    r3 = Relation(label="VISITED", source_id=e2.id, target_id=e4.id)
    agens_store.upsert_relations([r1, r2, r3])

    # Depth 2 should capture up to "Alice - Bob - London" chain
    rel_map = agens_store.get_rel_map([e1], depth=2)
    # Expect at least 2-3 relationships
    labels_found = {trip[1].label for trip in rel_map}
    assert "KNOWS" in labels_found
    assert "VISITED" in labels_found


def test_12_get_schema(agens_store: AgensPropertyGraphStore):
    """
    Test get_schema. The schema might be empty or minimal if no data has been inserted yet.
    """
    # Insert some data first
    e1 = EntityNode(label="PERSON", name="Alice")
    agens_store.upsert_nodes([e1])

    schema = agens_store.get_schema(refresh=True)
    assert "node_props" in schema
    assert "rel_props" in schema
    assert "relationships" in schema


def test_13_get_schema_str(agens_store: AgensPropertyGraphStore):
    """
    Test the textual representation of the schema.
    """
    e1 = EntityNode(label="PERSON", name="Alice")
    e2 = EntityNode(label="CITY", name="Paris")
    agens_store.upsert_nodes([e1, e2])

    # Insert a relationship
    r = Relation(label="VISITED", source_id=e1.id, target_id=e2.id)
    agens_store.upsert_relations([r])

    schema_str = agens_store.get_schema_str(refresh=True)
    assert "PERSON" in schema_str
    assert "CITY" in schema_str
    assert "VISITED" in schema_str


def test_14_structured_query(agens_store: AgensPropertyGraphStore):
    """
    Test running a custom Cypher query via structured_query.
    """
    # Insert data
    e1 = EntityNode(label="PERSON", name="Alice")
    agens_store.upsert_nodes([e1])

    # Custom query
    query = """
    MATCH (n) WHERE n.name = 'Alice'
    RETURN n.name AS node_name, n.labels AS node_labels
    """
    result = agens_store.structured_query(query)
    assert len(result) == 1
    assert result[0]["node_name"] == "Alice"
    assert "PERSON" in result[0]["node_labels"]


def test_15_refresh_schema(agens_store: AgensPropertyGraphStore):
    """
    Test explicit refresh of the schema.
    """
    # Insert data
    e1 = EntityNode(label="PERSON", name="Alice", properties={"age": 30})
    agens_store.upsert_nodes([e1])

    # Refresh schema
    agens_store.refresh_schema()
    schema = agens_store.structured_schema
    assert "node_props" in schema
    person_props = schema["node_props"].get("PERSON", [])
    prop_names = {prop["property"] for prop in person_props}
    assert "age" in prop_names, "Expected 'age' property in PERSON schema."

def test_16_pg_store_all(agens_store: AgensPropertyGraphStore) -> None:
    """Test functions for Agensgraph graph store."""

    # Test inserting nodes into AgensGraph.
    entity1 = EntityNode(label="PERSON", name="Logan", properties={"age": 28})
    entity2 = EntityNode(label="ORGANIZATION", name="LlamaIndex")
    agens_store.upsert_nodes([entity1, entity2])
    # Assert the nodes are inserted correctly
    kg_nodes = agens_store.get(ids=[entity1.id])
    assert kg_nodes[0].name == entity1.name

    # Test inserting relations into AgensGraph.
    relation = Relation(
        label="WORKS_FOR",
        source_id=entity1.id,
        target_id=entity2.id,
        properties={"since": 2023},
    )

    agens_store.upsert_relations([relation])
    # Assert the relation is inserted correctly by retrieving the relation map
    kg_nodes = agens_store.get(ids=[entity1.id])
    agens_store.get_rel_map(kg_nodes, depth=1)

    # Test inserting a source text node and 'MENTIONS' relations.
    source_node = TextNode(text='Logan (age 28), works for "LlamaIndex" since 2023.')

    relations = [
        Relation(label="MENTIONS", target_id=entity1.id, source_id=source_node.node_id),
        Relation(label="MENTIONS", target_id=entity2.id, source_id=source_node.node_id),
    ]

    agens_store.upsert_llama_nodes([source_node])
    agens_store.upsert_relations(relations)

    # Assert the source node and relations are inserted correctly
    agens_store.get_llama_nodes([source_node.node_id])

    # Test retrieving nodes by properties.
    kg_nodes = agens_store.get(properties={"age": 28})

    # Test executing a structured query in AgensGraph.
    query = """MATCH (n:"__Node__") WHERE '__Entity__' IN n.labels
               RETURN n"""
    agens_store.structured_query(query)

    # Test upserting a new node with additional properties.
    new_node = EntityNode(
        label="PERSON", name="Logan", properties={"age": 28, "location": "Canada"}
    )
    agens_store.upsert_nodes([new_node])

    # Assert the node has been updated with the new property
    kg_nodes = agens_store.get(properties={"age": 28})

    # Test deleting nodes from AgensGraph.
    agens_store.delete(ids=[source_node.node_id])
    agens_store.delete(ids=[entity1.id, entity2.id])

    # Assert the nodes have been deleted
    agens_store.get(ids=[entity1.id, entity2.id])
