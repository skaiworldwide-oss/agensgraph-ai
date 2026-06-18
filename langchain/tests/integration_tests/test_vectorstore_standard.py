"""LangChain's standard VectorStore conformance suite.

Subclasses the canonical test suite from ``langchain-tests`` (the one
``langchain-postgres`` already adopts). Validates that ``AgensgraphVector``
honors the VectorStore protocol: ``add_documents``, ``get_by_ids``,
``delete``, async siblings, and id-based mutation semantics.
"""

from __future__ import annotations

import os
import uuid
from typing import Generator

import pytest
from langchain_core.vectorstores import VectorStore
from langchain_tests.integration_tests.vectorstores import VectorStoreIntegrationTests

from langchain_agensgraph.graphs.agensgraph import AgensGraph
from langchain_agensgraph.vectorstores.agensgraph_vector import AgensgraphVector


def _conf():
    return {
        "dbname": os.getenv("AGENSGRAPH_DB"),
        "user": os.getenv("AGENSGRAPH_USER"),
        "password": os.getenv("AGENSGRAPH_PASSWORD"),
        "host": os.getenv("AGENSGRAPH_HOST", "localhost"),
        "port": int(os.getenv("AGENSGRAPH_PORT", 5432)),
    }


class TestAgensgraphVectorStandard(VectorStoreIntegrationTests):
    """Run the langchain-tests vector store conformance suite."""

    @pytest.fixture()
    def vectorstore(self) -> Generator[VectorStore, None, None]:
        # Use a unique node label per test class run so concurrent
        # CI shards do not collide on shared graph state.
        node_label = f"std_{uuid.uuid4().hex[:8]}"
        AgensGraph("standardtest", _conf(), create=True)
        store = AgensgraphVector.from_texts(
            texts=["bootstrap"],
            embedding=self.get_embeddings(),
            graph_name="standardtest",
            url=os.environ.get("AGENSGRAPH_URL"),
            node_label=node_label,
            pre_delete_collection=True,
        )
        # `from_texts` seeds one row; the suite expects an empty store.
        store.query(
            "MATCH (n) DETACH DELETE n"
        )
        try:
            yield store
        finally:
            try:
                store.query("MATCH (n) DETACH DELETE n")
            except Exception:
                pass
