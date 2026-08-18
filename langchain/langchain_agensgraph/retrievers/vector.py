"""Retrieve documents by vector or hybrid similarity over an AgensgraphVector."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document

from langchain_agensgraph.retrievers._base import _AgensRetrieverBase
from langchain_agensgraph.vectorstores.agensgraph_vector import (
    AgensgraphVector,
    HybridSearchConfig,
)


class AgensVectorRetriever(_AgensRetrieverBase):
    """Seed-only retrieval: one similarity search, scored documents out.

    The store decides whether that search is a plain vector scan or the
    server-side reciprocal-rank fusion of vector and keyword halves -- both run
    as one statement, so a retrieval is one server round trip either way. Every
    knob here is a default a call may override: ``retriever.invoke(query, k=8,
    filter=...)`` reaches the same search the constructor's values do.

    ``retrieval_query`` shapes what each hit returns without touching the store,
    so several retrievers can share one store -- and its one connection pool --
    while each reads a different context. The query sees ``node`` (or
    ``relationship``) and ``score``, returns ``text``, ``score``, ``doc_id`` and
    ``metadata``, and doubles literal braces.

    Example:
        .. code-block:: python

            from langchain_agensgraph import AgensgraphVector
            from langchain_agensgraph.retrievers import AgensVectorRetriever

            store = AgensgraphVector.from_existing_index(embedding, url=url)
            retriever = AgensVectorRetriever(store=store, k=6)
            docs = retriever.invoke("what failed over the weekend?")
    """

    store: AgensgraphVector
    filter: Optional[Dict[str, Any]] = None
    params: Optional[Dict[str, Any]] = None
    effective_search_ratio: float = 1.0
    search_options: Optional[Dict[str, Any]] = None
    hybrid_config: Optional[HybridSearchConfig] = None
    retrieval_query: Optional[str] = None

    def _effective_retrieval_query(self) -> Optional[str]:
        """The retrieval query this retriever runs; subclasses own theirs."""
        return self.retrieval_query

    def _search_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """This call's search arguments: the retriever's, with overrides laid over."""
        return {
            "k": kwargs.get("k", self.k),
            "filter": kwargs.get("filter", self.filter),
            "params": kwargs.get("params", self.params),
            "effective_search_ratio": kwargs.get(
                "effective_search_ratio", self.effective_search_ratio
            ),
            "search_options": kwargs.get("search_options", self.search_options),
            "hybrid_config": kwargs.get("hybrid_config", self.hybrid_config),
            "retrieval_query": self._effective_retrieval_query(),
        }

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun, **kwargs: Any
    ) -> List[Document]:
        hits = self.store.similarity_search_with_score(
            query, **self._search_kwargs(kwargs)
        )
        return self._finalize(hits)

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> List[Document]:
        # The embedding is awaited too: a provider's embed call is a network
        # round trip, and running it blocking inside the loop would serialize
        # every concurrent retrieval behind it.
        embedding = await self.store.embedding.aembed_query(query)
        hits = await self.store.asimilarity_search_with_score_by_vector(
            embedding=embedding, query=query, **self._search_kwargs(kwargs)
        )
        return self._finalize(hits)
