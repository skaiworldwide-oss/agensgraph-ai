"""
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
"""

"""Nodes live on one label per DataPoint class, and what cognee reads back from them."""

import pytest
import pytest_asyncio
from cognee.infrastructure.engine import DataPoint
from cognee.modules.chunking.models import DocumentChunk
from cognee.modules.data.processing.document_types import TextDocument
from cognee.modules.engine.models import Entity, EntityType
from cognee.modules.graph.utils import deduplicate_nodes_and_edges, get_graph_from_model
from cognee.tasks.summarization.models import TextSummary

from cognee_agensgraph.infrastructure.databases.graph.agensgraph.adapter import (
    AgensgraphAdapter,
    bounded_name,
    label_for,
)

pytestmark = pytest.mark.asyncio


class Person(DataPoint):
    name: str
    metadata: dict = {"index_fields": ["name"]}


class Place(DataPoint):
    name: str
    metadata: dict = {"index_fields": ["name"]}


@pytest_asyncio.fixture
async def adapter(conn_url):
    a = AgensgraphAdapter(conn_url)
    await a.initialize()
    await a.delete_graph()
    try:
        yield a
    finally:
        await a.delete_graph()
        await a.finalize()


async def _public_objects(adapter):
    async with adapter._engine.connection() as conn:
        cur = await conn.execute(
            "SELECT p.proname FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace "
            "WHERE n.nspname = 'public' "
            "UNION ALL "
            "SELECT c.relname FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
            "WHERE n.nspname = 'public' AND c.relkind = 'r' ORDER BY 1"
        )
        return [r[0] for r in await cur.fetchall()]


async def test_the_adapter_installs_nothing_in_public(conn_url):
    a = AgensgraphAdapter(conn_url)
    await a.initialize()
    before = await _public_objects(a)
    b = AgensgraphAdapter(conn_url)
    await b.initialize()
    await b.add_nodes([Person(name="Ann"), Place(name="Oslo")])
    after = await _public_objects(b)
    assert after == before
    await b.delete_graph()
    await b.finalize()


async def test_each_class_gets_a_label_with_its_own_uniqueness(adapter):
    ann, oslo = Person(name="Ann"), Place(name="Oslo")
    await adapter.add_nodes([ann, oslo])
    async with adapter._engine.connection() as conn:
        labels = {l.name: l for l in await conn.labels(graph=adapter.graph_name)}
        unique = {c.label for c in await conn.constraints(graph=adapter.graph_name) if c.unique}
    # labels are lower case, so an unquoted label in Cypher, which folds to lower case, finds them
    assert labels["person"].parent == "__node__" and labels["place"].parent == "__node__"
    assert {"person", "place"} <= unique
    rows = await adapter.query("MATCH (n:__Node__) RETURN label(n) AS label ORDER BY label")
    assert [r["label"] for r in rows] == ["person", "place"]
    assert len(await adapter.query("MATCH (n:Person) RETURN n")) == 1
    assert len(await adapter.query('MATCH (n:"person") RETURN n')) == 1
    # the class is still readable as a property, which is what cognee reads
    node = await adapter.get_node(str(ann.id))
    assert node["type"] == "Person" and node["name"] == "Ann"


async def test_writing_the_same_nodes_twice_creates_nothing(adapter):
    people = [Person(name=f"p{i}") for i in range(20)]
    await adapter.add_nodes(people)
    await adapter.add_nodes(people)
    rows = await adapter.query("MATCH (n:Person) RETURN count(n) AS c")
    assert rows[0]["c"] == 20
    # and the second write did not clobber cognee's integer timestamps with strings
    node = await adapter.get_node(str(people[0].id))
    assert isinstance(node["updated_at"], int)


async def test_two_long_class_names_stay_apart():
    a = "A" * 70 + "x"
    b = "A" * 70 + "y"
    assert bounded_name(a) != bounded_name(b)
    assert len(bounded_name(a).encode()) <= 63
    assert bounded_name("Entity") == "Entity"
    assert label_for("Entity") == "entity" and label_for("is_a") == "is_a"


async def test_has_edges_returns_the_existing_edges_as_tuples(adapter):
    ann, bob, oslo = Person(name="Ann"), Person(name="Bob"), Place(name="Oslo")
    await adapter.add_nodes([ann, bob, oslo])
    await adapter.add_edges([(ann.id, bob.id, "knows", {}), (ann.id, oslo.id, "visited", {})])
    found = await adapter.has_edges(
        [(ann.id, bob.id, "knows"), (ann.id, oslo.id, "knows"), (bob.id, oslo.id, "visited")]
    )
    assert found == [(str(ann.id), str(bob.id), "knows")]
    # cognee indexes the result as edge[0], edge[1], edge[2]
    assert str(found[0][0]) + str(found[0][1]) + found[0][2] == f"{ann.id}{bob.id}knows"


async def test_connections_carry_both_directions(adapter):
    ann, bob, oslo = Person(name="Ann"), Person(name="Bob"), Place(name="Oslo")
    await adapter.add_nodes([ann, bob, oslo])
    await adapter.add_edges([(ann.id, bob.id, "knows", {}), (oslo.id, ann.id, "hosted", {})])
    connections = await adapter.get_connections(ann.id)
    shapes = {(c[0]["name"], c[1]["relationship_name"], c[2]["name"]) for c in connections}
    assert shapes == {("Ann", "knows", "Bob"), ("Oslo", "hosted", "Ann")}


async def test_edges_carry_their_endpoints_and_name_as_properties(adapter):
    ann, bob = Person(name="Ann"), Person(name="Bob")
    await adapter.add_nodes([ann, bob])
    await adapter.add_edge(ann.id, bob.id, "knows", {"since": 2020})
    nodes, edges = await adapter.get_graph_data()
    assert {n[0] for n in nodes} == {str(ann.id), str(bob.id)}
    (edge,) = edges
    assert edge[:3] == (str(ann.id), str(bob.id), "knows")
    assert edge[3]["relationship_name"] == "knows" and edge[3]["since"] == 2020


async def _cognee_shaped(adapter):
    """Two documents that share one entity; a second entity belongs to one document only."""
    place = EntityType(name="Place", description="d")
    shared = Entity(name="Paris", description="d", is_a=place)
    private = Entity(name="Lyon", description="d", is_a=place)
    docs, chunks, summaries = [], [], []
    for i, ents in enumerate(([shared, private], [shared])):
        doc = TextDocument(
            name=f"text_hash{i}", raw_data_location="/x", external_metadata=None, mime_type="text/plain"
        )
        chunk = DocumentChunk(
            text=f"chunk {i}", chunk_size=1, chunk_index=0, cut_type="x", is_part_of=doc, contains=ents
        )
        docs.append(doc)
        chunks.append(chunk)
        summaries.append(TextSummary(text=f"summary {i}", made_from=chunk))
    nodes, edges = [], []
    seen_nodes, seen_edges, visited = {}, {}, {}
    for point in chunks + summaries:
        n, e = await get_graph_from_model(
            point, added_nodes=seen_nodes, added_edges=seen_edges, visited_properties=visited
        )
        nodes.extend(n)
        edges.extend(e)
    nodes, edges = deduplicate_nodes_and_edges(nodes, edges)
    await adapter.add_nodes(nodes)
    await adapter.add_edges(edges)
    return docs, chunks, summaries, shared, private, place


async def test_document_subgraph_keeps_what_another_document_still_uses(adapter):
    docs, chunks, summaries, shared, private, place = await _cognee_shaped(adapter)
    sub = await adapter.get_document_subgraph("hash0")
    assert [d["id"] for d in sub["document"]] == [str(docs[0].id)]
    assert [c["id"] for c in sub["chunks"]] == [str(chunks[0].id)]
    assert [s["id"] for s in sub["made_from_nodes"]] == [str(summaries[0].id)]
    # Paris is also in document 1, so it is not this document's to delete; Lyon is
    assert [e["id"] for e in sub["orphan_entities"]] == [str(private.id)]
    # the type is still used by Paris, which stays
    assert sub["orphan_types"] == []
    assert await adapter.get_document_subgraph("no-such-hash") is None


async def test_degree_one_and_disconnected_nodes(adapter):
    docs, chunks, summaries, shared, private, place = await _cognee_shaped(adapter)
    alone = Entity(name="Nowhere", description="d")
    await adapter.add_nodes([alone])
    # Lyon has two edges (its chunk and its type); the type has two (its entities)
    assert await adapter.get_degree_one_nodes("Entity") == []
    await adapter.delete_nodes([str(place.id)])
    ones = {n["name"] for n in await adapter.get_degree_one_nodes("Entity")}
    assert ones == {"Lyon"}
    assert await adapter.get_disconnected_nodes() == [str(alone.id)]
    with pytest.raises(ValueError):
        await adapter.get_degree_one_nodes("DocumentChunk")


async def test_nodeset_subgraph_returns_the_named_nodes_and_their_neighbourhood(adapter):
    docs, chunks, summaries, shared, private, place = await _cognee_shaped(adapter)
    nodes, edges = await adapter.get_nodeset_subgraph(Entity, ["Lyon"])
    names = {n[1].get("name") or n[1].get("text") for n in nodes}
    assert "Lyon" in names and "chunk 0" in names and "Place" in names
    assert all(src in {n[0] for n in nodes} and tgt in {n[0] for n in nodes} for src, tgt, _, _ in edges)
    assert {e[2] for e in edges} == {"contains", "is_a"}
    # a class that has never been written is an empty answer, not an error
    assert await adapter.get_nodeset_subgraph(Place, ["Oslo"]) == ([], [])


async def test_filtered_graph_data_returns_cognee_ids(adapter):
    ann, bob, oslo = Person(name="Ann"), Person(name="Bob"), Place(name="Oslo")
    await adapter.add_nodes([ann, bob, oslo])
    await adapter.add_edges([(ann.id, bob.id, "knows", {}), (ann.id, oslo.id, "visited", {})])
    nodes, edges = await adapter.get_filtered_graph_data([{"name": ["Ann", "Bob", "O'Brien"]}])
    assert {n[0] for n in nodes} == {str(ann.id), str(bob.id)}
    assert [(e[0], e[1], e[2]) for e in edges] == [(str(ann.id), str(bob.id), "knows")]
    with pytest.raises(ValueError):
        await adapter.get_filtered_graph_data([{"name) OR true OR (n.x": ["Ann"]}])


async def test_the_schema_query_of_the_natural_language_search_sees_the_classes(adapter):
    await adapter.add_nodes([Person(name="Ann"), Place(name="Oslo")])
    rows = await adapter.query(
        "MATCH (n) UNWIND keys(n) AS prop "
        "RETURN DISTINCT labels(n) AS NodeLabels, collect(DISTINCT prop) AS Properties"
    )
    seen = {label for r in rows for label in r["nodelabels"]}
    assert {"person", "place"} <= seen


async def test_edges_reach_nodes_written_by_another_process(adapter):
    ann, bob = Person(name="Ann"), Place(name="Bob")
    await adapter.add_nodes([ann, bob])
    # Another process wrote the nodes: this one has no memory of their labels and finds
    # them through the parent label instead.
    adapter._recent.clear()
    await adapter.add_edges([(ann.id, bob.id, "knows", {})])
    assert await adapter.has_edge(ann.id, bob.id, "knows") is True
    # and an edge whose endpoints are remembered lands on the same edge label
    await adapter.add_nodes([ann, bob])
    await adapter.add_edges([(bob.id, ann.id, "knows", {})])
    rows = await adapter.query("MATCH ()-[r:knows]->() RETURN count(r) AS c")
    assert rows[0]["c"] == 2


async def test_one_node_per_id_across_classes(adapter):
    # cognee derives an id from a name, so an Entity and an EntityType called the same
    # thing share an id. cognee's own adapters keep one node for it.
    shared = Entity(name="empire", description="d")
    kind = EntityType(id=shared.id, name="empire", description="d")
    await adapter.add_nodes([shared])
    await adapter.add_nodes([kind])
    rows = await adapter.query("MATCH (n:__Node__) WHERE n.id = %(id)s RETURN label(n) AS label, n.type AS type",
                               {"id": str(shared.id)})
    assert len(rows) == 1
    assert rows[0]["label"] == "entity" and rows[0]["type"] == "EntityType"
    # and an edge to that id lands on the one node, whichever class named it
    other = Entity(name="rome", description="d")
    await adapter.add_nodes([other])
    await adapter.add_edges([(other.id, kind.id, "is_a", {})])
    assert await adapter.has_edge(other.id, shared.id, "is_a") is True
    nodes, edges = await adapter.get_graph_data()
    assert len(nodes) == 2 and len(edges) == 1
    # a second process that never saw the first write finds the node where it is
    adapter._recent.clear()
    await adapter.add_nodes([EntityType(id=shared.id, name="empire", description="again")])
    rows = await adapter.query("MATCH (n:__Node__) WHERE n.id = %(id)s RETURN count(n) AS c", {"id": str(shared.id)})
    assert rows[0]["c"] == 1
