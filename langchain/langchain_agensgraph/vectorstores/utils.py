"""Vector store utilities.

``DistanceStrategy`` is vendored locally so this package does not depend on the archived
``langchain-community``. ``maximal_marginal_relevance`` is re-exported from
``langchain_core``, which keeps it in ``langchain_core.vectorstores.utils`` as of
langchain-core 1.x.

Every member here maps onto an ``agensgraph.vector.Distance``, which carries the
operator pgvector measures with and the operator class an index needs to answer it.
"""

from __future__ import annotations

from enum import Enum

from agensgraph.vector import Distance
from langchain_core.vectorstores.utils import maximal_marginal_relevance


class DistanceStrategy(str, Enum):
    """Distance strategy supported by the AgensGraph vector store."""

    EUCLIDEAN_DISTANCE = "EUCLIDEAN_DISTANCE"
    """Euclidean distance, ``<->``. The usual choice for an unnormalised embedding."""

    COSINE = "COSINE"
    """Cosine distance, ``<=>``. The default, and what a unit-normed embedding wants."""

    MAX_INNER_PRODUCT = "MAX_INNER_PRODUCT"
    """Negative inner product, ``<#>``, so that smaller is nearer as it is for the rest."""

    DOT_PRODUCT = "DOT_PRODUCT"
    """A second name for the same operator, kept because callers use both."""

    TAXICAB = "TAXICAB"
    """L1 distance, ``<+>``. Needs pgvector 0.7 or later."""

    HAMMING = "HAMMING"
    """How many bits differ, ``<~>``. For a ``bit`` column, not a ``vector`` one."""

    JACCARD = "JACCARD"
    """How much two bit strings fail to overlap, ``<%>``. Also for a ``bit`` column."""


DISTANCE: dict[DistanceStrategy, Distance] = {
    DistanceStrategy.EUCLIDEAN_DISTANCE: Distance.L2,
    DistanceStrategy.COSINE: Distance.COSINE,
    DistanceStrategy.MAX_INNER_PRODUCT: Distance.INNER_PRODUCT,
    DistanceStrategy.DOT_PRODUCT: Distance.INNER_PRODUCT,
    DistanceStrategy.TAXICAB: Distance.L1,
    DistanceStrategy.HAMMING: Distance.HAMMING,
    DistanceStrategy.JACCARD: Distance.JACCARD,
}
"""What each strategy measures, as the driver names it.

The driver's :class:`~agensgraph.vector.Distance` carries both halves of the pairing
that has to agree for an index to be used at all: the operator a search orders by, and
the operator class the index was built with. Naming one place for both is what keeps
them from drifting apart -- an index built for cosine answers ``<=>`` and nothing else,
and a search ordering by ``<->`` against it sorts a sequential scan instead.
"""


def distance_of(strategy: DistanceStrategy) -> Distance:
    """The driver's distance for a strategy, refusing one nothing measures."""
    try:
        return DISTANCE[strategy]
    except KeyError:  # pragma: no cover - unreachable while the map is complete
        raise ValueError(
            f"no distance operator for {strategy}. One of {sorted(DISTANCE)}"
        ) from None


__all__ = [
    "DISTANCE",
    "DistanceStrategy",
    "distance_of",
    "maximal_marginal_relevance",
]
