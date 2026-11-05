"""The graph store against a server: queries, graph documents, the schema, model Cypher."""

from __future__ import annotations

import threading

import psycopg
import psycopg.sql
import pytest
from agensgraph import Jsonb
from agensgraph.errors import CapabilityError, ConfigurationError
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from psycopg.conninfo import make_conninfo

from langchain_agensgraph._core.engine import AgensEngine
from langchain_agensgraph.chat_message_histories import AgensChatMessageHistory
from langchain_agensgraph.errors import (
    AgensQueryError,
    DuplicateKeys,
    UnknownLabel,
    UnsafeCypherError,
)
from langchain_agensgraph.graphs import AgensGraph, GraphDocument, GraphStore, Node, Relationship
from langchain_agensgraph.observability import log_queries
from langchain_agensgraph.roles import render_grants
from tests import datagen

pytestmark = pytest.mark.server


@pytest.fixture
def graph(engine: AgensEngine, graph_name: str) -> AgensGraph:
    return AgensGraph(engine, graph_name=graph_name, create_graph=True)


def documents(n: int = 20) -> list[GraphDocument]:
    out = []
    for doc in datagen.graph_documents(n, entities=15, per_doc=4, seed=3):
        nodes = {
            e.id: Node(id=e.id, type=e.type, properties=dict(e.properties)) for e in doc.entities
        }
        out.append(
            GraphDocument(
                nodes=list(nodes.values()),
                relationships=[
                    Relationship(
                        source=nodes[link.source],
                        target=nodes[link.target],
                        type=link.type,
                        properties=dict(link.properties),
                    )
                    for link in doc.links
                ],
                source=Document(doc.source, metadata={"origin": "test"}),
            )
        )
    return out


def counts(graph: AgensGraph) -> dict[str, int]:
    rows = graph.query("MATCH (n) RETURN label(n) AS label, count(*) AS n")
    edges = graph.query("MATCH ()-[r]->() RETURN type(r) AS label, count(*) AS n")
    return {r["label"]: r["n"] for r in rows + edges}


def test_a_graph_is_made_only_when_asked(engine: AgensEngine, graph_name: str) -> None:
    with pytest.raises(UnknownLabel, match="create_graph=True"):
        AgensGraph(engine, graph_name=graph_name).query("RETURN 1")
    graph = AgensGraph(engine, graph_name=graph_name, create_graph=True)
    assert graph.query("RETURN 1 AS one") == [{"one": 1}]
    assert isinstance(graph, GraphStore)


def test_a_query_returns_rows_and_a_refusal_carries_its_code(graph: AgensGraph) -> None:
    graph.query("CREATE (:p {name: 'a', n: 1}), (:p {name: 'b', n: 2})")
    rows = graph.query("MATCH (x:p) WHERE x.n > %s RETURN x.name AS name", (1,))
    assert rows == [{"name": "b"}]
    with pytest.raises(AgensQueryError) as refused:
        graph.query("MATCH (x:p RETURN x")
    assert refused.value.sqlstate == "42601"
    with pytest.raises(AgensQueryError) as slow:
        graph.query("RETURN pg_sleep(1) AS s", timeout=0.05)
    assert slow.value.sqlstate == "57014"


def test_sanitize_drops_a_long_list_key_and_all(engine: AgensEngine, graph_name: str) -> None:
    long = "[" + ", ".join("1" for _ in range(200)) + "]"
    statement = f"RETURN {long} AS long, [1, 2] AS short"
    sanitized = AgensGraph(engine, graph_name=graph_name, create_graph=True, sanitize=True)
    assert sanitized.query(statement) == [{"short": [1, 2]}]
    plain = AgensGraph(engine, graph_name=graph_name)
    assert len(plain.query(statement)[0]["long"]) == 200


def test_query_many_applies_together_or_not_at_all(graph: AgensGraph) -> None:
    results = graph.query_many(["CREATE (:q {n: 1})", ("MATCH (x:q) RETURN count(x) AS c", None)])
    assert results[1] == [{"c": 1}]
    with pytest.raises(AgensQueryError):
        graph.query_many(["CREATE (:q {n: 2})", "MATCH (x:q RETURN x"])
    assert graph.query("MATCH (x:q) RETURN count(x) AS c") == [{"c": 1}]


def test_graph_documents_are_written_once_however_often_they_are_added(graph: AgensGraph) -> None:
    docs = documents()
    graph.add_graph_documents(docs, batch_size=7)
    first = counts(graph)
    graph.add_graph_documents(docs)
    assert counts(graph) == first
    assert sum(first[t] for t in ("Person", "Organization", "Place", "Concept")) == 15
    entity = graph.query("MATCH (n:\"Person\" {id: 'e0'}) RETURN n.name AS name, n.rank AS r")
    assert entity == [{"name": "entity 0", "r": 0}]


def test_a_bare_node_keeps_the_properties_its_vertex_has(dsn: str, graph: AgensGraph) -> None:
    graph.ensure_labels(["Paper", "Author"], ["BY"])
    for batch in range(8):
        ids = [f"p{batch}-{i}" for i in range(1000)]
        graph.query("UNWIND %s AS i CREATE (:\"Paper\" {id: i, text: 'kept'})", (Jsonb(ids),))
        docs = [
            GraphDocument(
                nodes=[],
                relationships=[
                    Relationship(
                        source=Node(id=i, type="Paper"),
                        target=Node(id="a", type="Author", properties={"name": "a"}),
                        type="BY",
                        properties={"n": batch},
                    )
                ],
            )
            for i in ids
        ]
        graph.add_graph_documents(docs, batch_size=1000)
        with psycopg.connect(dsn, autocommit=True) as conn:
            conn.execute(f'ANALYZE "{graph.graph_name}"."Paper"')
    kept = graph.query("MATCH (p:\"Paper\") WHERE p.text = 'kept' RETURN count(*) AS n")
    assert kept == [{"n": 8000}]
    edges = graph.query('MATCH ()-[r:"BY"]->() WHERE r.n IS NOT NULL RETURN count(*) AS n')
    assert edges == [{"n": 8000}]


def test_sources_and_a_base_entity_label(graph: AgensGraph) -> None:
    docs = documents(5)
    before = [dict(d.source.metadata) for d in docs]
    graph.add_graph_documents(docs, include_source=True, baseEntityLabel=True)
    assert [d.source.metadata for d in docs] == before
    assert counts(graph)["Document"] == 5
    mentions = graph.query('MATCH (:"Document")-[r:"MENTIONS"]->() RETURN count(r) AS c')
    assert mentions == [{"c": 20}]
    entities = graph.query('MATCH (n:"__Entity__") RETURN count(n) AS c')[0]["c"]
    assert entities == sum(counts(graph)[t] for t in ("Person", "Organization", "Place", "Concept"))
    text = graph.query('MATCH (d:"Document") RETURN d.text AS t, d.origin AS o LIMIT 1')
    assert text[0]["o"] == "test" and text[0]["t"].startswith("document")


def test_a_label_others_inherit_from_is_estimated_with_them_and_never_counted(
    dsn: str, graph: AgensGraph
) -> None:
    graph.add_graph_documents(documents(5), baseEntityLabel=True)
    with psycopg.connect(dsn, autocommit=True) as conn:
        labels = conn.execute(
            "SELECT l.labname::text FROM ag_label l JOIN ag_graph g ON g.oid = l.graphid "
            "WHERE g.graphname = %s",
            (graph.graph_name,),
        ).fetchall()
        for (label,) in labels:
            conn.execute(
                psycopg.sql.SQL("ANALYZE {}.{}").format(
                    psycopg.sql.Identifier(graph.graph_name), psycopg.sql.Identifier(label)
                )
            )
    with log_queries() as statements:
        graph.refresh_schema(force=True)
    estimated = graph.get_structured_schema["counts"]
    entities = ("Person", "Organization", "Place", "Concept")
    assert estimated["__Entity__"] == sum(estimated[t] for t in entities)
    assert estimated["__Entity__"] == sum(counts(graph)[t] for t in entities)
    counted = [s.statement for s in statements if s.statement.startswith("SELECT count(*) FROM")]
    assert not [statement for statement in counted if "__Entity__" in statement]


def test_a_label_keeps_its_spelling(graph: AgensGraph) -> None:
    a = Node(id=1, type="사람")
    b = Node(id=2, type="Lower case")
    graph.add_graph_documents(
        [
            GraphDocument(
                nodes=[a, b],
                relationships=[Relationship(source=a, target=b, type="안다")],
                source=Document("x"),
            )
        ]
    )
    assert graph.query('MATCH (:"사람")-[r:"안다"]->(n:"Lower case") RETURN n.id AS id') == [
        {"id": 2}
    ]


def test_concurrent_writers_cannot_duplicate_an_edge_and_old_duplicates_are_named(
    graph: AgensGraph,
) -> None:
    graph.add_graph_documents(documents(3))
    with pytest.raises(AgensQueryError) as again:
        graph.query(
            "MATCH (a)-[r]->(b) WITH a, b, type(r) AS t LIMIT 1 "
            "MATCH (a2), (b2) WHERE id(a2) = id(a) AND id(b2) = id(b) "
            'CREATE (a2)-[:"KNOWS"]->(b2) CREATE (a2)-[:"KNOWS"]->(b2)'
        )
    assert again.value.sqlstate == "23505"
    graph.query('CREATE (a:"Multi" {id: 1})-[:"LINK"]->(b:"Multi" {id: 2}), (a)-[:"LINK"]->(b)')
    a = Node(id=1, type="Multi")
    b = Node(id=2, type="Multi")
    with pytest.raises(DuplicateKeys, match="dedupe_edges=False"):
        graph.add_graph_documents(
            [
                GraphDocument(
                    nodes=[a, b],
                    relationships=[Relationship(source=a, target=b, type="LINK")],
                    source=Document("x"),
                )
            ]
        )
    graph.add_graph_documents(
        [
            GraphDocument(
                nodes=[a, b],
                relationships=[Relationship(source=a, target=b, type="LINK")],
                source=Document("x"),
            )
        ],
        dedupe_edges=False,
    )


def test_writers_adding_the_same_new_nodes_at_once_all_succeed(graph: AgensGraph) -> None:
    graph.ensure_labels(["Person"])
    failures: list[Exception] = []
    for round_ in range(3):
        people = [Node(id=f"{round_}-{i}", type="Person") for i in range(400)]
        start = threading.Barrier(4)

        def write(people: list[Node] = people, start: threading.Barrier = start) -> None:
            start.wait()
            try:
                graph.add_graph_documents([GraphDocument(nodes=people, relationships=[])])
            except Exception as exc:  # noqa: BLE001 - collected for the assertion
                failures.append(exc)

        writers = [threading.Thread(target=write) for _ in range(4)]
        for writer in writers:
            writer.start()
        for writer in writers:
            writer.join()
    assert failures == []
    assert graph.query('MATCH (n:"Person") RETURN count(n) AS n') == [{"n": 1200}]


def test_the_schema_describes_what_is_there(graph: AgensGraph) -> None:
    graph.add_graph_documents(documents())
    graph.query('CREATE VLABEL "Empty"')
    schema = graph.get_structured_schema
    assert set(schema["node_props"]) == {"Person", "Organization", "Place", "Concept"}
    assert {p["property"] for p in schema["node_props"]["Person"]} == {"id", "name", "rank"}
    assert {p["property"] for p in schema["rel_props"]["KNOWS"]} == {"weight"}
    triples = {(r["start"], r["type"], r["end"]) for r in schema["relationships"]}
    assert triples and {t for _, t, _ in triples} <= {"KNOWS", "WORKS_AT", "LOCATED_IN", "MENTIONS"}
    assert schema["counts"]["Person"] > 0 and "Empty" not in schema["counts"]
    assert any(i["label"] == "Person" and i["unique"] for i in schema["indexes"])
    text = graph.get_schema
    assert text.startswith("Node properties are the following:")
    assert "The relationships are the following:" in text and "Empty" not in text
    graph.query('CREATE (:"Empty" {x: 1})')
    assert "Empty" not in graph.get_schema
    graph.refresh_schema(force=True)
    assert "Empty" in graph.get_schema


def test_an_enhanced_schema_carries_examples(engine: AgensEngine, graph_name: str) -> None:
    graph = AgensGraph(engine, graph_name=graph_name, create_graph=True, enhanced_schema=True)
    graph.add_graph_documents(documents(5))
    person = graph.get_structured_schema["node_props"]["Person"]
    assert any(entry.get("examples") for entry in person)


def test_model_cypher_runs_as_its_own_role(
    dsn: str, engine: AgensEngine, graph: AgensGraph
) -> None:
    graph.add_graph_documents(documents(5))
    with pytest.raises(ConfigurationError):
        graph.query_untrusted('MATCH (n:"Person") RETURN count(n) AS c')
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute(render_grants(graph.graph_name))
    untrusted = make_conninfo(dsn, user="llm_ro", password="agens")
    with AgensEngine(dsn, untrusted_conninfo=untrusted) as both:
        model = AgensGraph(both, graph_name=graph.graph_name)
        rows = model.query_untrusted('MATCH (n:"Person") RETURN n.id AS id', max_rows=2)
        assert len(rows) == 2 and set(rows[0]) == {"id"}
        with pytest.raises(UnsafeCypherError, match="a write"):
            model.query_untrusted('MATCH (n:"Person") SET n.x = 1 RETURN n')
        with pytest.raises(UnsafeCypherError, match="more than one statement"):
            model.query_untrusted("MATCH (n) RETURN n; DROP GRAPH g CASCADE")
        with pytest.raises(AgensQueryError) as refused:
            model.query_untrusted('MATCH (n:"Person") RETURN n.id AS id ORDER')
        assert refused.value.sqlstate == "42601"
        write = 'CREATE (:"Person" {id: "m"}) RETURN 1 AS one'
        if model.capabilities.has("cypher_write_enforced"):
            with pytest.raises(ConfigurationError, match="writer_conninfo"):
                model.query_untrusted(write, allow_writes=True)
        else:
            with pytest.raises(CapabilityError, match="2.16 or later"):
                model.query_untrusted(write, allow_writes=True)


def test_a_callers_transaction_takes_graph_documents_with_it(graph: AgensGraph) -> None:
    graph.add_graph_documents(documents(1))
    before = counts(graph)
    with pytest.raises(RuntimeError):
        with graph.transaction():
            graph.add_graph_documents(documents(5), include_source=True)
            assert counts(graph) != before
            raise RuntimeError
    assert counts(graph) == before


@pytest.mark.timeout(60)
def test_labels_a_callers_transaction_needs_are_made_in_it(graph: AgensGraph) -> None:
    ada = Node(id="ada", type="Person", properties={"name": "Ada"})
    acme = Node(id="acme", type="Org", properties={"name": "Acme"})
    works = Relationship(source=ada, target=acme, type="WORKS_AT")
    later = GraphDocument(nodes=[ada, acme], relationships=[works])
    with pytest.raises(RuntimeError):
        with graph.transaction():
            # The second document's labels are made after the first wrote to Person, which a
            # connection of their own would wait on for as long as this transaction is open.
            first = GraphDocument(nodes=[ada], relationships=[])
            graph.add_graph_documents([first, later], batch_size=1)
            raise RuntimeError
    # The rollback took the labels too, so they are made again rather than taken as made.
    graph.add_graph_documents([later])
    assert graph.query("MATCH (a)-[r]->(b) RETURN a.id AS a, type(r) AS r, b.id AS b") == [
        {"a": "ada", "r": "WORKS_AT", "b": "acme"}
    ]


async def test_the_awaiting_interface_does_the_same(
    dsn: str, engine: AgensEngine, graph: AgensGraph
) -> None:
    await graph.aadd_graph_documents(documents(5))
    rows = await graph.aquery('MATCH (n:"Person") RETURN count(n) AS c')
    assert rows[0]["c"] > 0
    await graph.arefresh_schema(force=True)
    assert "Person" in graph.get_structured_schema["node_props"]
    await graph.aensure_labels(["Tag"], ["TAGGED"])
    assert await graph.aquery('MATCH (t:"Tag") RETURN count(*) AS c') == [{"c": 0}]
    await graph.adrop_graph()
    assert not graph._setup.done
    with psycopg.connect(dsn) as conn:
        left = conn.execute(
            "SELECT count(*) FROM ag_graph WHERE graphname = %s", (graph.graph_name,)
        ).fetchone()
    assert left == (0,)
    await engine.aclose()


def test_dropping_the_graph_takes_its_companions(
    dsn: str, engine: AgensEngine, graph_name: str
) -> None:
    graph = AgensGraph(engine, graph_name=graph_name, create_graph=True)
    graph.add_graph_documents(documents(2))
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute(f'CREATE SCHEMA "{graph_name}_store"')
    graph.drop_graph()
    with psycopg.connect(dsn, autocommit=True) as conn:
        left = conn.execute(
            "SELECT (SELECT count(*) FROM ag_graph WHERE graphname = %s), to_regnamespace(%s)",
            (graph_name, f"{graph_name}_store"),
        ).fetchone()
    assert left == (0, None)


def test_components_set_up_again_once_what_they_made_is_dropped(
    dsn: str, engine: AgensEngine, graph_name: str
) -> None:
    dropper = AgensGraph(engine, graph_name=graph_name, create_graph=True)
    other = AgensGraph(engine, graph_name=graph_name, create_graph=True)
    other.add_graph_documents(documents(2))
    chat = AgensChatMessageHistory("s", engine, graph_name=graph_name)
    chat.add_messages([HumanMessage("before", id="m1")])
    dropper.drop_graph()
    # Each made its graph, labels or tables before the drop, and makes them again after it.
    other.add_graph_documents(documents(2))
    chat.add_messages([HumanMessage("after", id="m2")])
    assert [m.content for m in chat.messages] == ["after"]
    # Dropped by someone else: the call that finds the tables gone fails, and the next sets up.
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute(f'DROP SCHEMA "{graph_name}_store" CASCADE')
    with pytest.raises(psycopg.errors.InvalidSchemaName):
        chat.add_messages([HumanMessage("lost", id="m3")])
    chat.add_messages([HumanMessage("again", id="m4")])
    assert [m.content for m in chat.messages] == ["again"]
