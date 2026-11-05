"""LangChain's standard vector store suite, in each configuration a store can take."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any, ClassVar

import pytest
from langchain_core.vectorstores import VectorStore
from langchain_tests.integration_tests.vectorstores import VectorStoreIntegrationTests

from langchain_agensgraph._core.engine import AgensEngine
from langchain_agensgraph.vectorstores import AgensgraphVector

pytestmark = pytest.mark.server


@pytest.fixture
async def engine(dsn: str) -> AsyncIterator[AgensEngine]:
    """An engine per test, closed on the test's own loop, which its awaiting pools belong to."""
    made = AgensEngine(dsn)
    yield made
    await made.aclose()


class TestAgensgraphVector(VectorStoreIntegrationTests):
    """An approximate index, built after the first write."""

    options: ClassVar[dict[str, Any]] = {}

    @pytest.fixture
    def vectorstore(self, engine: AgensEngine, graph_name: str) -> Iterator[VectorStore]:
        yield AgensgraphVector(
            self.get_embeddings(),
            engine=engine,
            graph_name=graph_name,
            create_graph=True,
            **self.options,
        )


class TestExactSearch(TestAgensgraphVector):
    options: ClassVar[dict[str, Any]] = {"index": None}


class TestHybridSearch(TestAgensgraphVector):
    options: ClassVar[dict[str, Any]] = {"search_type": "hybrid"}


class TestHalfPrecision(TestAgensgraphVector):
    options: ClassVar[dict[str, Any]] = {"vector_type": "halfvec"}
