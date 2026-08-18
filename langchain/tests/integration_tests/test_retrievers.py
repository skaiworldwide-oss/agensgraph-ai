"""The retriever family, run against a live server."""

import os
from hashlib import md5
from typing import Any, Dict, List, Optional

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from langchain_agensgraph.graphs.agensgraph import AgensGraph
from langchain_agensgraph.observability import log_queries
from langchain_agensgraph.retrievers import (
    AgensGraphContextRetriever,
    AgensText2CypherRetriever,
    AgensVectorRetriever,
    render_graph_context,
)
from langchain_agensgraph.vectorstores.agensgraph_vector import (
    AgensgraphVector,
    SearchType,
)
from tests.integration_tests.fake_embeddings import FakeEmbeddings

url = os.environ.get(
    "AGENSGRAPH_URL",
    "postgresql://{}:{}@{}:{}/{}".format(
        os.getenv("AGENSGRAPH_USER"),
        os.getenv("AGENSGRAPH_PASSWORD"),
        os.getenv("AGENSGRAPH_HOST", "localhost"),
        os.getenv("AGENSGRAPH_PORT", 5432),
        os.getenv("AGENSGRAPH_DB"),
    ),
)

conf = {
    "dbname": os.getenv("AGENSGRAPH_DB"),
    "user": os.getenv("AGENSGRAPH_USER"),
    "password": os.getenv("AGENSGRAPH_PASSWORD"),
    "host": os.getenv("AGENSGRAPH_HOST", "localhost"),
    "port": int(os.getenv("AGENSGRAPH_PORT", 5432)),
}

GRAPH = "retriever_it"

texts = ["foo", "bar", "baz", "It is the end of the world. Take shelter!"]


def make_store(
    metadatas: Optional[List[Dict[str, Any]]] = None, **kwargs: Any
) -> AgensgraphVector:
    return AgensgraphVector.from_texts(
        texts=texts,
        metadatas=metadatas,
        embedding=FakeEmbeddings(),
        pre_delete_collection=True,
        graph_name=GRAPH,
        url=url,
        **kwargs,
    )


def cleanup(store: AgensgraphVector) -> None:
    store.query("MATCH (n) DETACH DELETE n")
    store.close()


class OneHotEmbeddings(Embeddings):
    """Each known text is its own axis, so any text can be made the seed."""

    def __init__(self, known: List[str]) -> None:
        self.known = list(known)

    def _one(self, text: str) -> List[float]:
        # An unknown text embeds to the origin -- the store probes its
        # dimension with a text of its own choosing.
        vector = [0.0] * len(self.known)
        if text in self.known:
            vector[self.known.index(text)] = 1.0
        return vector

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._one(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._one(text)


# The little graph the context tests walk:
#
#     island        foo -relates-> bar -relates-> baz
#                    \-other-> end
CONTEXT_TEXTS = ["foo", "bar", "baz", "end", "island"]
EDGES = [("foo", "bar", "relates"), ("bar", "baz", "relates"), ("foo", "end", "other")]


def _id(text: str) -> str:
    """The identity ``from_texts`` writes when none is given."""
    return md5(text.encode()).hexdigest()


def make_context_store(**kwargs: Any) -> AgensgraphVector:
    store = AgensgraphVector.from_texts(
        texts=CONTEXT_TEXTS,
        embedding=OneHotEmbeddings(CONTEXT_TEXTS),
        pre_delete_collection=True,
        graph_name=GRAPH,
        url=url,
        **kwargs,
    )
    for elabel in {edge[2] for edge in EDGES}:
        store.query(f"CREATE ELABEL IF NOT EXISTS {elabel}")
    for start, end, elabel in EDGES:
        store.query(
            "MATCH (a:\"Chunk\"), (b:\"Chunk\") "
            "WHERE a.__id__ = %(a)s AND b.__id__ = %(b)s "
            f"CREATE (a)-[:{elabel}]->(b)",
            params={"a": _id(start), "b": _id(end)},
        )
    return store


def context_texts(doc: Document) -> set:
    return {n["properties"]["text"] for n in doc.metadata["_context_nodes_"]}


def context_types(doc: Document) -> List[str]:
    return sorted(r["type"] for r in doc.metadata["_context_rels_"])


class TestAgensVectorRetriever:
    def test_invoke_returns_scored_documents(self) -> None:
        store = make_store()
        retriever = AgensVectorRetriever(store=store, k=2)
        docs = retriever.invoke("foo")
        assert len(docs) == 2
        assert docs[0].page_content == "foo"
        assert docs[0].metadata["__retriever"] == "AgensVectorRetriever"
        assert 0.0 <= docs[0].metadata["score"] <= 1.0
        # Nearer first: the query's own text outranks its neighbour.
        assert docs[0].metadata["score"] >= docs[1].metadata["score"]
        cleanup(store)

    def test_a_call_overrides_the_constructor_k(self) -> None:
        store = make_store()
        retriever = AgensVectorRetriever(store=store, k=1)
        assert len(retriever.invoke("foo")) == 1
        assert len(retriever.invoke("foo", k=3)) == 3
        cleanup(store)

    def test_filter_narrows_and_a_call_may_replace_it(self) -> None:
        store = make_store(metadatas=[{"page": str(i)} for i in range(len(texts))])
        retriever = AgensVectorRetriever(store=store, k=4, filter={"page": "1"})
        docs = retriever.invoke("foo")
        assert [d.page_content for d in docs] == ["bar"]
        # An invoke-time filter replaces the constructor's, not narrows it further.
        docs = retriever.invoke("foo", filter={"page": "2"})
        assert [d.page_content for d in docs] == ["baz"]
        cleanup(store)

    def test_hybrid_store_serves_the_same_retriever(self) -> None:
        store = make_store(search_type=SearchType.HYBRID)
        retriever = AgensVectorRetriever(store=store, k=2)
        docs = retriever.invoke("foo")
        assert len(docs) == 2
        assert docs[0].page_content == "foo"
        cleanup(store)

    def test_retrieval_query_shapes_every_call(self) -> None:
        store = make_store()
        retriever = AgensVectorRetriever(
            store=store,
            k=1,
            retrieval_query=(
                "RETURN 'shaped' AS text, score, node.__id__ AS doc_id, "
                "{{origin: 'retriever'}} AS metadata"
            ),
        )
        docs = retriever.invoke("foo")
        assert docs[0].page_content == "shaped"
        assert docs[0].metadata["origin"] == "retriever"
        # The store itself is untouched.
        assert store.similarity_search("foo", k=1)[0].page_content == "foo"
        cleanup(store)

    def test_document_formatter_reshapes_after_stamping(self) -> None:
        store = make_store()

        def shout(doc: Document) -> Document:
            return Document(
                page_content=doc.page_content.upper(), metadata=doc.metadata
            )

        retriever = AgensVectorRetriever(store=store, k=1, document_formatter=shout)
        docs = retriever.invoke("foo")
        assert docs[0].page_content == "FOO"
        assert "score" in docs[0].metadata
        cleanup(store)

    async def test_the_async_path_agrees(self) -> None:
        store = make_store()
        retriever = AgensVectorRetriever(store=store, k=3)
        wanted = retriever.invoke("foo")
        got = await retriever.ainvoke("foo")
        assert [d.page_content for d in got] == [d.page_content for d in wanted]
        assert [d.metadata["score"] for d in got] == [
            d.metadata["score"] for d in wanted
        ]
        await store.aclose()
        cleanup(store)


class TestAgensGraphContextRetriever:
    def test_one_hop_collects_the_direct_neighbourhood(self) -> None:
        store = make_context_store()
        retriever = AgensGraphContextRetriever(store=store, k=1, expand_by_hops=1)
        docs = retriever.invoke("foo")
        assert len(docs) == 1
        assert docs[0].page_content == "foo"
        assert context_texts(docs[0]) == {"bar", "end"}
        assert context_types(docs[0]) == ["other", "relates"]
        cleanup(store)

    def test_two_hops_reach_across_the_middle_vertex(self) -> None:
        store = make_context_store()
        retriever = AgensGraphContextRetriever(store=store, k=1, expand_by_hops=2)
        docs = retriever.invoke("foo")
        assert context_texts(docs[0]) == {"bar", "baz", "end"}
        assert context_types(docs[0]) == ["other", "relates", "relates"]
        cleanup(store)

    def test_a_relationship_type_narrows_the_walk(self) -> None:
        store = make_context_store()
        retriever = AgensGraphContextRetriever(
            store=store, k=1, expand_by_hops=2, relationship_type="relates"
        )
        docs = retriever.invoke("foo")
        assert context_texts(docs[0]) == {"bar", "baz"}
        assert context_types(docs[0]) == ["relates", "relates"]
        cleanup(store)

    def test_a_seed_without_neighbours_survives_with_empty_context(self) -> None:
        store = make_context_store()
        retriever = AgensGraphContextRetriever(store=store, k=1, expand_by_hops=2)
        docs = retriever.invoke("island")
        assert len(docs) == 1
        assert docs[0].page_content == "island"
        assert docs[0].metadata["_context_nodes_"] == []
        assert docs[0].metadata["_context_rels_"] == []
        cleanup(store)

    def test_the_caps_bound_the_payload(self) -> None:
        store = make_context_store()
        retriever = AgensGraphContextRetriever(
            store=store, k=1, expand_by_hops=1, max_context_nodes=1, max_context_rels=1
        )
        docs = retriever.invoke("foo")
        # Which member survives the cut is the server's choice; only the size holds.
        assert len(docs[0].metadata["_context_nodes_"]) == 1
        assert len(docs[0].metadata["_context_rels_"]) == 1
        assert context_texts(docs[0]) <= {"bar", "end"}
        cleanup(store)

    def test_the_whole_retrieval_is_one_statement(self) -> None:
        store = make_context_store()
        retriever = AgensGraphContextRetriever(store=store, k=2, expand_by_hops=2)
        with log_queries() as statements:
            docs = retriever.invoke("foo")
        assert len(docs) == 2
        assert len(statements) == 1
        cleanup(store)

    def test_a_hybrid_seed_gets_the_same_context(self) -> None:
        store = make_context_store(search_type=SearchType.HYBRID)
        retriever = AgensGraphContextRetriever(store=store, k=1, expand_by_hops=1)
        docs = retriever.invoke("foo")
        assert docs[0].page_content == "foo"
        assert context_texts(docs[0]) == {"bar", "end"}
        cleanup(store)

    async def test_the_async_path_agrees(self) -> None:
        store = make_context_store()
        retriever = AgensGraphContextRetriever(store=store, k=2, expand_by_hops=2)
        wanted = retriever.invoke("foo")
        got = await retriever.ainvoke("foo")
        assert [d.page_content for d in got] == [d.page_content for d in wanted]
        assert context_texts(got[0]) == context_texts(wanted[0])
        await store.aclose()
        cleanup(store)

    def test_it_owns_its_retrieval_query(self) -> None:
        store = make_context_store()
        with pytest.raises(ValueError, match="builds its own retrieval query"):
            AgensGraphContextRetriever(store=store, retrieval_query="RETURN 1")
        retriever = AgensGraphContextRetriever(store=store, k=1)
        with pytest.raises(ValueError, match="builds its own retrieval query"):
            retriever.invoke("foo", retrieval_query="RETURN 1")
        cleanup(store)

    def test_a_thousand_edge_hub_stays_bounded(self) -> None:
        # The caps bound the payload; the hop bound is what bounds the walking.
        # A hub with a thousand spokes is the shape that would blow either up.
        import time

        from psycopg.types.json import Jsonb

        store = AgensgraphVector.from_texts(
            texts=["hub"],
            embedding=OneHotEmbeddings(["hub"]),
            pre_delete_collection=True,
            graph_name=GRAPH,
            url=url,
        )
        store.query("CREATE VLABEL IF NOT EXISTS sat")
        store.query("CREATE ELABEL IF NOT EXISTS spoke")
        store.query(
            "UNWIND %(ids)s AS i CREATE (:sat {i: i})",
            params={"ids": Jsonb(list(range(1000)))},
        )
        store.query(
            'MATCH (a:"Chunk"), (s:sat) WHERE a.__id__ = %(h)s '
            "CREATE (a)-[:spoke]->(s)",
            params={"h": _id("hub")},
        )
        retriever = AgensGraphContextRetriever(store=store, k=1, expand_by_hops=2)
        started = time.monotonic()
        docs = retriever.invoke("hub")
        elapsed = time.monotonic() - started
        assert len(docs[0].metadata["_context_nodes_"]) == 20
        assert len(docs[0].metadata["_context_rels_"]) == 20
        assert elapsed < 5.0, f"hub expansion took {elapsed:.2f}s"
        cleanup(store)

    def test_render_graph_context_writes_the_context_into_the_text(self) -> None:
        store = make_context_store()
        retriever = AgensGraphContextRetriever(
            store=store, k=1, expand_by_hops=1, document_formatter=render_graph_context
        )
        docs = retriever.invoke("foo")
        assert "Graph context:" in docs[0].page_content
        assert "-[relates]->" in docs[0].page_content
        cleanup(store)


READ_PEOPLE = "MATCH (n:person) RETURN n.name AS name ORDER BY name"
BROKEN = "MATCH (n:person RETURN n"


def make_qa_graph() -> AgensGraph:
    graph = AgensGraph("retriever_t2c", conf, create=True, refresh_schema=False)
    graph.query("CREATE VLABEL IF NOT EXISTS person")
    graph.query("MATCH (n:person) DETACH DELETE n")
    graph.query("CREATE (:person {name: 'Ada'}), (:person {name: 'Bob'})")
    graph.refresh_schema(force=True)
    return graph


def fake_llm(*responses: str):
    from langchain_core.language_models import FakeListChatModel

    return FakeListChatModel(responses=list(responses))


class TestAgensText2CypherRetriever:
    def test_rows_become_documents_with_their_cypher(self) -> None:
        graph = make_qa_graph()
        retriever = AgensText2CypherRetriever(
            graph=graph, llm=fake_llm(READ_PEOPLE), allow_server_programs=True
        )
        docs = retriever.invoke("who is here?")
        assert len(docs) == 2
        assert '"Ada"' in docs[0].page_content
        assert docs[0].metadata["cypher"] == READ_PEOPLE
        assert docs[0].metadata["__retriever"] == "AgensText2CypherRetriever"
        graph.close()

    def test_a_generated_write_is_refused_and_nothing_changes(self) -> None:
        graph = make_qa_graph()
        retriever = AgensText2CypherRetriever(
            graph=graph, llm=fake_llm("CREATE (:person {name: 'Eve'})")
        )
        with pytest.raises(ValueError, match="Refusing"):
            retriever.invoke("add Eve")
        rows = graph.query("MATCH (n:person) RETURN count(*) AS n")
        assert rows[0]["n"] == 2
        graph.close()

    def test_an_execution_error_feeds_a_correction_round(self) -> None:
        graph = make_qa_graph()
        llm = fake_llm(BROKEN, READ_PEOPLE)
        retriever = AgensText2CypherRetriever(
            graph=graph, llm=llm, max_retries=1, allow_server_programs=True
        )
        docs = retriever.invoke("who is here?")
        assert len(docs) == 2
        assert docs[0].metadata["cypher"] == READ_PEOPLE
        graph.close()

    def test_an_empty_result_can_ask_for_a_second_opinion(self) -> None:
        # A relationship written against its schema direction plans, runs, and
        # matches nothing -- no error, just zero rows. retry_on_empty turns
        # that silence into feedback.
        graph = make_qa_graph()
        empty = "MATCH (n:person) WHERE n.name = 'Nobody' RETURN n.name AS name"
        llm = fake_llm(empty, READ_PEOPLE)
        retriever = AgensText2CypherRetriever(
            graph=graph,
            llm=llm,
            max_retries=1,
            retry_on_empty=True,
            allow_server_programs=True,
        )
        docs = retriever.invoke("who is here?")
        assert len(docs) == 2
        assert docs[0].metadata["cypher"] == READ_PEOPLE
        graph.close()

    def test_an_empty_result_is_an_answer_by_default(self) -> None:
        graph = make_qa_graph()
        empty = "MATCH (n:person) WHERE n.name = 'Nobody' RETURN n.name AS name"
        llm = fake_llm(empty, READ_PEOPLE)
        retriever = AgensText2CypherRetriever(
            graph=graph, llm=llm, max_retries=1, allow_server_programs=True
        )
        assert retriever.invoke("who is here?") == []
        # The model was not consulted a second time over a legitimate empty.
        assert llm.i == 1
        graph.close()

    def test_no_correction_happens_unasked(self) -> None:
        graph = make_qa_graph()
        llm = fake_llm(BROKEN, READ_PEOPLE)
        retriever = AgensText2CypherRetriever(
            graph=graph, llm=llm, allow_server_programs=True
        )
        with pytest.raises(ValueError, match="not runnable"):
            retriever.invoke("who is here?")
        # Only the generation call happened; no correction round was paid for.
        assert llm.i == 1
        graph.close()

    def test_a_spent_budget_refuses_another_attempt(self) -> None:
        graph = make_qa_graph()
        llm = fake_llm(BROKEN, READ_PEOPLE)
        retriever = AgensText2CypherRetriever(
            graph=graph,
            llm=llm,
            max_retries=5,
            timeout=0.001,
            allow_server_programs=True,
        )
        with pytest.raises(Exception):
            retriever.invoke("who is here?")
        assert llm.i == 1
        graph.close()

    def test_k_bounds_the_rows(self) -> None:
        graph = make_qa_graph()
        retriever = AgensText2CypherRetriever(
            graph=graph,
            llm=fake_llm(READ_PEOPLE),
            k=1,
            allow_server_programs=True,
        )
        assert len(retriever.invoke("who?")) == 1
        # The constructor's k also fed the statement's own LIMIT, so a larger
        # invoke-time k cannot return more than was fetched.
        assert len(retriever.invoke("who?", k=5)) == 1
        graph.close()

    async def test_the_async_path_agrees(self) -> None:
        graph = make_qa_graph()
        wanted = AgensText2CypherRetriever(
            graph=graph, llm=fake_llm(READ_PEOPLE), allow_server_programs=True
        ).invoke("who?")
        got = await AgensText2CypherRetriever(
            graph=graph, llm=fake_llm(READ_PEOPLE), allow_server_programs=True
        ).ainvoke("who?")
        assert [d.page_content for d in got] == [d.page_content for d in wanted]
        await graph.aclose()
        graph.close()
