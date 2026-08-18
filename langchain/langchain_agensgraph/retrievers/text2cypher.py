"""Retrieve rows the graph answers to a question written in plain language."""

from __future__ import annotations

from typing import Any, List, Optional

import agensgraph
from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import Field, PrivateAttr

from langchain_agensgraph.chains.cypher_qa import (
    AgensCypherQAChain,
    quote_identifiers,
    strip_fences,
)
from langchain_agensgraph.graphs.agensgraph import AgensGraph
from langchain_agensgraph.retrievers._base import _AgensRetrieverBase

_CORRECTION = """That query failed with:
{error}

Write a corrected Cypher query answering the same question. Return only the query."""

# An empty result raises nothing, so it needs its own feedback. The dominant
# cause is a relationship written against its schema direction -- the pattern
# plans, runs, and matches nothing.
_EMPTY_FEEDBACK = (
    "That query ran without error and returned zero rows. Check every "
    "relationship's direction and every label's spelling against the schema; "
    "the schema's directions are authoritative."
)


class AgensText2CypherRetriever(_AgensRetrieverBase):
    """A model writes the Cypher; the server is trusted to contain it.

    The question is turned into Cypher against the graph's schema and run under
    the same safety pipeline the QA chain uses: a write is refused by reading
    the statement, ``EXPLAIN`` proves it plans before it runs, the execution
    happens inside a read-only transaction the server enforces, and one time
    budget covers the checking and the running together.

    ``max_retries`` (default 0) turns execution failures into feedback: the
    failed query and the server's error go back to the model for a corrected
    attempt, all attempts spending the one budget -- a spent budget fails fast
    rather than paying for another round. ``retry_on_empty`` extends that to a
    query that runs clean and matches nothing, which is what a relationship
    written against its schema direction looks like: the pattern plans, runs,
    and returns zero rows. It is opt-in because an empty result is sometimes
    the true answer, and asking again costs a model call.

    Each returned document is one result row as JSON, with the Cypher that
    produced it in ``metadata["cypher"]``. ``k`` bounds the rows; the value the
    retriever was built with also feeds the generation prompt's LIMIT, so an
    invoke-time ``k`` above it cannot return more than the statement fetched.

    Example:
        .. code-block:: python

            retriever = AgensText2CypherRetriever(graph=graph, llm=llm, k=10)
            rows = retriever.invoke("Which people joined after 2024?")
    """

    graph: AgensGraph
    llm: BaseLanguageModel
    k: int = Field(default=10, ge=1)
    timeout: Optional[float] = 30.0
    validate_cypher: bool = True
    allow_dangerous_requests: bool = False
    allow_server_programs: bool = False
    cypher_prompt: Optional[ChatPromptTemplate] = None
    max_retries: int = Field(default=0, ge=0)
    retry_on_empty: bool = False

    _chain: AgensCypherQAChain = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        self._chain = AgensCypherQAChain.from_llm(
            self.llm,
            graph=self.graph,
            top_k=self.k,
            timeout=self.timeout,
            validate_cypher=self.validate_cypher,
            allow_dangerous_requests=self.allow_dangerous_requests,
            allow_server_programs=self.allow_server_programs,
            cypher_prompt=self.cypher_prompt,
        )

    # ---- generation ----

    def _corrected(self, question: str, cypher: str, error: object) -> str:
        """Ask the model again, showing it what it wrote and what the server said."""
        chain = self._chain
        messages = chain.cypher_prompt.format_messages(
            schema=chain._schema(), question=question
        )
        messages.append(AIMessage(content=cypher))
        messages.append(
            HumanMessage(content=_CORRECTION.format(error=str(error)[:500]))
        )
        raw = StrOutputParser().invoke(self.llm.invoke(messages))
        return quote_identifiers(strip_fences(raw), chain._known_names())

    async def _acorrected(self, question: str, cypher: str, error: object) -> str:
        chain = self._chain
        messages = chain.cypher_prompt.format_messages(
            schema=chain._schema(), question=question
        )
        messages.append(AIMessage(content=cypher))
        messages.append(
            HumanMessage(content=_CORRECTION.format(error=str(error)[:500]))
        )
        raw = StrOutputParser().invoke(await self.llm.ainvoke(messages))
        return quote_identifiers(strip_fences(raw), chain._known_names())

    def _out_of_time(self) -> bool:
        """Whether the budget is too spent to pay for another attempt."""
        remaining = self._chain._remaining
        return remaining is not None and remaining <= 0.005

    def _documents(self, rows: List[dict], cypher: str, k: int) -> List[Document]:
        docs = []
        for row in rows[:k]:
            doc = Document(
                page_content=agensgraph.to_json(row).decode(),
                metadata={"cypher": cypher, "__retriever": type(self).__name__},
            )
            if self.document_formatter is not None:
                doc = self.document_formatter(doc)
            docs.append(doc)
        return docs

    # ---- retrieval ----

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun, **kwargs: Any
    ) -> List[Document]:
        chain = self._chain
        cypher = chain.generate_cypher(query)
        with chain._budget():
            for attempt in range(self.max_retries + 1):
                try:
                    chain.check(cypher)
                    rows = chain.run_cypher(cypher)
                except Exception as exc:
                    if attempt == self.max_retries or self._out_of_time():
                        raise
                    cypher = self._corrected(query, cypher, exc)
                    continue
                if rows or not self.retry_on_empty:
                    break
                if attempt == self.max_retries or self._out_of_time():
                    break
                cypher = self._corrected(query, cypher, _EMPTY_FEEDBACK)
        return self._documents(rows, cypher, kwargs.get("k", self.k))

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> List[Document]:
        chain = self._chain
        generate = chain.cypher_prompt | chain.cypher_llm | StrOutputParser()
        raw = await generate.ainvoke({"schema": chain._schema(), "question": query})
        cypher = quote_identifiers(strip_fences(raw), chain._known_names())
        with chain._budget():
            for attempt in range(self.max_retries + 1):
                try:
                    await chain.acheck(cypher)
                    rows = await chain.arun_cypher(cypher)
                except Exception as exc:
                    if attempt == self.max_retries or self._out_of_time():
                        raise
                    cypher = await self._acorrected(query, cypher, exc)
                    continue
                if rows or not self.retry_on_empty:
                    break
                if attempt == self.max_retries or self._out_of_time():
                    break
                cypher = await self._acorrected(query, cypher, _EMPTY_FEEDBACK)
        return self._documents(rows, cypher, kwargs.get("k", self.k))
