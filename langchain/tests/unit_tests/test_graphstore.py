"""The graph store's pure parts: grouping graph documents, and the schema a prompt reads."""

from __future__ import annotations

from types import SimpleNamespace

from langchain_core.documents import Document

from langchain_agensgraph.graphs import GraphDocument, Node, Relationship
from langchain_agensgraph.graphs.agensgraph import _collect, _entities, _index_facts, _rendered


def test_a_node_is_written_once_with_everything_said_about_it() -> None:
    ada = Node(id="a", type="Person", properties={"name": "Ada"})
    bare = Node(id="a", type="Person")
    org = Node(id=7, type="Org")
    doc = GraphDocument(
        nodes=[ada, org],
        relationships=[Relationship(source=bare, target=org, type="WORKS_AT", properties={"y": 1})],
        source=Document("Ada works at seven", metadata={"origin": "x"}),
    )
    vertices, edges, sources = _collect([doc], include_source=True)
    assert vertices["Person"] == {"a": {"name": "Ada"}}
    assert vertices["Org"] == {7: {}}
    assert edges[("Person", "WORKS_AT", "Org")] == [("a", 7, {"y": 1})]
    (key,) = vertices["Document"]
    assert vertices["Document"][key] == {"origin": "x", "text": "Ada works at seven"}
    assert sorted(edges[("Document", "MENTIONS", "Person")]) == [(key, "a", {})]
    assert sources == {"Document"} and doc.source.metadata == {"origin": "x"}
    assert _entities(vertices, sources) == ("Org", "Person")


def test_the_schema_text_names_each_section() -> None:
    text = _rendered(
        {
            "node_props": {"Person": [{"property": "name", "type": "STRING"}]},
            "rel_props": {"KNOWS": []},
            "relationships": [{"start": "Person", "type": "KNOWS", "end": "Person"}],
            "counts": {"Person": 3, "KNOWS": 2},
            "indexes": [{"label": "Person", "on": "(id)", "unique": True}],
        }
    )
    assert text.splitlines()[0] == "Node properties are the following:"
    assert '\'(:"Person")-[:"KNOWS"]->(:"Person")\'' in text
    assert "Element counts (estimates) are the following:" in text
    assert '(:"Person") ON (id) UNIQUE' in text


def test_only_btree_property_indexes_are_facts_for_a_prompt() -> None:
    def index(definition: str, unique: bool = False) -> SimpleNamespace:
        return SimpleNamespace(label="Person", definition=definition, unique=unique)

    facts = _index_facts(
        [
            index("CREATE UNIQUE INDEX p ON g.person USING btree (properties.'id')", True),
            index("CREATE INDEX q ON g.person USING gin (properties jsonb_path_ops)"),
            index("CREATE INDEX r ON g.person USING btree (properties.'__id__')"),
        ]
    )
    assert facts == [{"label": "Person", "on": "(properties.'id')", "unique": True}]
