"""What every retriever in this package shares.

A retriever here is a thin, typed face over one server round trip: the subclass
builds and runs its statement, and this base stamps what came back. Both the
blocking and the awaiting template methods are implemented by every subclass
directly -- none of them falls back to running the blocking path in an executor,
so concurrent retrievals overlap on the driver's own async connection.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field


class _AgensRetrieverBase(BaseRetriever):
    """Shared shape: a result count, score stamping, and a formatter hook.

    ``k`` is the number of documents a retrieval returns; a call may override it
    with ``retriever.invoke(query, k=...)`` and the override is never clamped by
    the constructor's value. ``document_formatter`` reshapes each finished
    document -- rendering graph context into ``page_content``, say -- after the
    score and provenance are stamped.
    """

    k: int = Field(default=4, ge=1)
    document_formatter: Optional[Callable[[Document], Document]] = None

    def _finalize(self, hits: List[Tuple[Document, float]]) -> List[Document]:
        """Stamp each hit with its score and which retriever produced it."""
        docs: List[Document] = []
        for doc, score in hits:
            doc.metadata["score"] = score
            doc.metadata["__retriever"] = type(self).__name__
            if self.document_formatter is not None:
                doc = self.document_formatter(doc)
            docs.append(doc)
        return docs
