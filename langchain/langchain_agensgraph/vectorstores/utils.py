"""Vector store utilities.

``DistanceStrategy`` is vendored locally so this package does not depend on
the archived ``langchain-community``. ``maximal_marginal_relevance`` is
re-exported from ``langchain_core``, which keeps it in
``langchain_core.vectorstores.utils`` as of langchain-core 1.x.
"""

from __future__ import annotations

from enum import Enum

from langchain_core.vectorstores.utils import maximal_marginal_relevance


class DistanceStrategy(str, Enum):
    """Distance strategy supported by the AgensGraph vector store."""

    EUCLIDEAN_DISTANCE = "EUCLIDEAN_DISTANCE"
    MAX_INNER_PRODUCT = "MAX_INNER_PRODUCT"
    DOT_PRODUCT = "DOT_PRODUCT"
    JACCARD = "JACCARD"
    COSINE = "COSINE"


__all__ = ["DistanceStrategy", "maximal_marginal_relevance"]
