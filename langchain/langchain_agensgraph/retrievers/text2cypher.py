"""Retrieve the rows the graph gives for a question written in plain language."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from agensgraph import to_json
from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.prompts import BasePromptTemplate
from langchain_core.runnables import Runnable
from pydantic import ConfigDict, Field, PrivateAttr

from ..chains.cypher_qa import AgensCypherQAChain, CypherRun
from ..graphs.agensgraph import AgensGraph
from ._base import _AgensRetrieverBase


class AgensText2CypherRetriever(_AgensRetrieverBase):
    """A model writes Cypher for the question, and each row it returns is a document.

    The statement is written and run as :class:`~langchain_agensgraph.AgensCypherQAChain` does:
    rewritten for the server, run with ``query_untrusted`` as the engine's untrusted role, and
    corrected by the model up to ``max_retries`` times when it is refused or fails.
    ``retry_on_empty`` also asks again when it returns no rows, which is what an edge written
    against its schema direction looks like; it is off by default because no rows is sometimes the
    answer.

    Each document is one row as JSON, with the statement in ``metadata["cypher"]``. ``k`` is how
    many rows are read; a ``k`` passed to ``invoke`` can return fewer, not more.

    Example:
        .. code-block:: python

            retriever = AgensText2CypherRetriever(
                graph=graph, llm=llm, k=10, allow_dangerous_requests=True
            )
            rows = retriever.invoke("Which people joined after 2024?")
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    graph: AgensGraph
    llm: Runnable[Any, Any]
    k: int = Field(default=10, ge=1)
    timeout: float = Field(default=10.0, gt=0)
    max_retries: int = Field(default=2, ge=0)
    retry_on_empty: bool = False
    allow_dangerous_requests: bool = False
    allow_writes: bool = False
    unsafe_shared_role: bool = False
    validate_cypher: bool = True
    cypher_prompt: BasePromptTemplate | None = None  # type: ignore[type-arg]
    examples: Sequence[tuple[str, str]] | None = None

    _chain: AgensCypherQAChain = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        self._chain = AgensCypherQAChain.from_llm(
            self.llm,
            graph=self.graph,
            top_k=self.k,
            timeout=self.timeout,
            max_retries=self.max_retries,
            allow_dangerous_requests=self.allow_dangerous_requests,
            allow_writes=self.allow_writes,
            unsafe_shared_role=self.unsafe_shared_role,
            validate_cypher=self.validate_cypher,
            cypher_prompt=self.cypher_prompt,
            examples=self.examples,
            return_direct=True,
        )

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun, **kwargs: Any
    ) -> list[Document]:
        run = self._chain.run_cypher(
            query,
            config={"callbacks": run_manager.get_child()},
            retry_on_empty=self.retry_on_empty,
        )
        return self._documents(run, kwargs.get("k", self.k))

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun, **kwargs: Any
    ) -> list[Document]:
        run = await self._chain.arun_cypher(
            query,
            config={"callbacks": run_manager.get_child()},
            retry_on_empty=self.retry_on_empty,
        )
        return self._documents(run, kwargs.get("k", self.k))

    def _documents(self, run: CypherRun, k: int) -> list[Document]:
        documents = []
        for row in run.rows[:k]:
            document = Document(page_content=to_json(row).decode(), metadata={"cypher": run.cypher})
            if self.document_formatter is not None:
                document = self.document_formatter(document)
            documents.append(document)
        return documents
