"""The retriever family, run against a live server."""

import os
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

from langchain_agensgraph.retrievers import AgensVectorRetriever
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
