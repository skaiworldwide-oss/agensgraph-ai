"""LangChain's standard retriever suite, over each retriever a vector store gives."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.retrievers import BaseRetriever
from langchain_tests.integration_tests import RetrieversIntegrationTests

from langchain_agensgraph._core.engine import AgensEngine
from langchain_agensgraph.retrievers import AgensGraphContextRetriever, AgensVectorRetriever
from langchain_agensgraph.vectorstores import AgensgraphVector

pytestmark = pytest.mark.server

# The suite asks for one and for three results, so the store holds more than three.
TEXTS = ["foo", "bar", "baz", "qux"]


class _Seeded(RetrieversIntegrationTests):
    store: AgensgraphVector

    @pytest.fixture(autouse=True)
    async def seeded(self, dsn: str, graph_name: str) -> AsyncIterator[None]:
        engine = AgensEngine(dsn)
        store = AgensgraphVector(
            DeterministicFakeEmbedding(size=6),
            engine=engine,
            graph_name=graph_name,
            create_graph=True,
        )
        await store.aadd_texts(TEXTS)
        type(self).store = store
        yield
        await engine.aclose()

    @property
    def retriever_query_example(self) -> str:
        return "foo"


class TestAgensVectorRetriever(_Seeded):
    @property
    def retriever_constructor(self) -> type[BaseRetriever]:
        return AgensVectorRetriever

    @property
    def retriever_constructor_params(self) -> dict[str, Any]:
        return {"vectorstore": self.store, "k": 2}


class TestAgensGraphContextRetriever(_Seeded):
    @property
    def retriever_constructor(self) -> type[BaseRetriever]:
        return AgensGraphContextRetriever

    @property
    def retriever_constructor_params(self) -> dict[str, Any]:
        return {"vectorstore": self.store, "k": 2}
