"""What every retriever in this package shares: a result count, scores, a formatter."""

from __future__ import annotations

from collections.abc import Callable, Iterable

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field


class _AgensRetrieverBase(BaseRetriever):
    """``k`` documents per retrieval, each carrying its score in ``metadata["score"]``.

    A call may pass ``k=`` and other search arguments to ``invoke``; they win over the
    retriever's own. ``document_formatter`` reshapes each finished document -- rendering graph
    context into the text, say. Every document handed out is a new one, so nothing the store or a
    cache holds is changed.
    """

    k: int = Field(default=4, ge=1)
    document_formatter: Callable[[Document], Document] | None = None

    def _finalize(self, hits: Iterable[tuple[Document, float]]) -> list[Document]:
        documents = []
        for document, score in hits:
            stamped = Document(
                id=document.id,
                page_content=document.page_content,
                metadata={**document.metadata, "score": score},
            )
            if self.document_formatter is not None:
                stamped = self.document_formatter(stamped)
            documents.append(stamped)
        return documents
