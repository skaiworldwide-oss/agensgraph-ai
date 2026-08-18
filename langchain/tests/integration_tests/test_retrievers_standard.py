"""LangChain's standard retriever conformance suite, over AgensVectorRetriever."""

import os
from typing import Type

import pytest
from langchain_core.retrievers import BaseRetriever
from langchain_tests.integration_tests import RetrieversIntegrationTests

from langchain_agensgraph.retrievers import AgensVectorRetriever
from langchain_agensgraph.vectorstores.agensgraph_vector import AgensgraphVector
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

# The suite constructs retrievers with k=1 and k=3, so the store must hold at
# least three documents for "foo" to return three neighbours.
TEXTS = ["foo", "bar", "baz", "qux"]

_store: AgensgraphVector = None


@pytest.fixture(autouse=True, scope="module")
def seeded_store():
    global _store
    _store = AgensgraphVector.from_texts(
        texts=TEXTS,
        embedding=FakeEmbeddings(),
        pre_delete_collection=True,
        graph_name="retriever_standard",
        url=url,
    )
    yield
    _store.query("MATCH (n) DETACH DELETE n")
    _store.close()
    _store = None


class TestAgensVectorRetrieverStandard(RetrieversIntegrationTests):
    @property
    def retriever_constructor(self) -> Type[BaseRetriever]:
        return AgensVectorRetriever

    @property
    def retriever_constructor_params(self) -> dict:
        return {"store": _store, "k": 2}

    @property
    def retriever_query_example(self) -> str:
        return "foo"
