"""Retrieve chunks by vector or hybrid similarity over an :class:`AgensgraphVector`."""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from pydantic import Field

from langchain_agensgraph.retrievers._base import _AgensRetrieverBase
from langchain_agensgraph.vectorstores.agensgraph_vector import AgensgraphVector


class AgensVectorRetriever(_AgensRetrieverBase):
    """One similarity search, scored documents out.

    ``search_kwargs`` go to the store's search on every call (``filter``, ``fetch_k``, ...), and
    a call's own arguments win over them. ``k`` is taken from the call, then from
    ``search_kwargs``, then from the field. ``retrieval_query`` reads each hit again through
    Cypher, with ``node`` and ``score`` in scope, returning ``text``, ``score`` and ``metadata``;
    rows returning the hit's own ``score`` come back best first, even from a query that aggregates::

        retriever = AgensVectorRetriever(
            vectorstore=store,
            retrieval_query=(
                "OPTIONAL MATCH (node)-[:MENTIONS]->(e) "
                "WITH node, score, collect(e.name) AS entities "
                "RETURN node.text AS text, score, {entities: entities} AS metadata"
            ),
        )
    """

    vectorstore: AgensgraphVector
    search_kwargs: dict[str, Any] = Field(default_factory=dict)
    retrieval_query: str | None = None

    def _arguments(self, kwargs: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        search = {**self.search_kwargs, **kwargs}
        k = int(search.pop("k", self.k))
        if self.retrieval_query is not None:
            search.setdefault("retrieval_query", self.retrieval_query)
        return k, search

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun, **kwargs: Any
    ) -> list[Document]:
        k, search = self._arguments(kwargs)
        where = search.pop("filter", None)
        hits = self.vectorstore.similarity_search_with_score(query, k, where, **search)
        return self._finalize(hits)

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun, **kwargs: Any
    ) -> list[Document]:
        k, search = self._arguments(kwargs)
        where = search.pop("filter", None)
        hits = await self.vectorstore.asimilarity_search_with_score(query, k, where, **search)
        return self._finalize(hits)
