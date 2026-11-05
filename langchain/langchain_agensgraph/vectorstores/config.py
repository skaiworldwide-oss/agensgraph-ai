"""How a vector store measures, indexes and searches.

Each :class:`DistanceStrategy` maps onto the driver's :class:`~agensgraph.Distance`, which carries
both halves of the pairing that has to agree for an index to be used: the operator a search orders
by and the operator class the index answers. An index built for cosine answers ``<=>`` and nothing
else.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Literal

from agensgraph import HNSW, Distance, IVFFlat


class DistanceStrategy(str, Enum):
    """How nearness is measured."""

    COSINE = "COSINE"
    EUCLIDEAN_DISTANCE = "EUCLIDEAN_DISTANCE"
    MAX_INNER_PRODUCT = "MAX_INNER_PRODUCT"
    DOT_PRODUCT = "DOT_PRODUCT"
    TAXICAB = "TAXICAB"
    HAMMING = "HAMMING"
    JACCARD = "JACCARD"


DISTANCE: dict[DistanceStrategy, Distance] = {
    DistanceStrategy.COSINE: Distance.COSINE,
    DistanceStrategy.EUCLIDEAN_DISTANCE: Distance.L2,
    DistanceStrategy.MAX_INNER_PRODUCT: Distance.INNER_PRODUCT,
    DistanceStrategy.DOT_PRODUCT: Distance.INNER_PRODUCT,
    DistanceStrategy.TAXICAB: Distance.L1,
}


def distance_of(strategy: DistanceStrategy | str) -> Distance:
    """The driver's distance for a strategy; Hamming and Jaccard measure bit strings, which a
    vector store of floating point embeddings does not keep."""
    chosen = DistanceStrategy(strategy)
    if chosen not in DISTANCE:
        raise ValueError(
            f"{chosen.value} measures bit strings; a vector store keeps floating point "
            f"embeddings. Choose one of {', '.join(s.value for s in DISTANCE)}"
        )
    return DISTANCE[chosen]


class SearchType(str, Enum):
    """``vector`` ranks by embedding alone; ``hybrid`` fuses it with a keyword search."""

    VECTOR = "vector"
    HYBRID = "hybrid"


@dataclass(frozen=True, slots=True)
class IndexConfig:
    """The approximate index over the embeddings.

    ``hnsw`` (the default) answers well from the first row and keeps answering as rows are added;
    ``ivfflat`` needs rows to exist before it is built and is smaller. Pass ``index=None`` to the
    store for exact search with no index at all.
    """

    method: Literal["hnsw", "ivfflat"] = "hnsw"
    m: int = 16
    ef_construction: int = 64
    lists: int = 100

    def driver(self) -> HNSW | IVFFlat:
        if self.method == "hnsw":
            return HNSW(m=self.m, ef_construction=self.ef_construction)
        if self.method == "ivfflat":
            return IVFFlat(lists=self.lists)
        raise ValueError(f"an index is hnsw or ivfflat, not {self.method!r}")


@dataclass(frozen=True, slots=True)
class KeywordConfig:
    """The keyword half of a hybrid search: a stored ``tsvector`` per chunk.

    ``properties`` are the vertex properties whose text is searched, the chunk's own text by
    default (``None``); ``regconfig`` the text search configuration that stems and stops them.
    """

    regconfig: str = "english"
    properties: tuple[str, ...] | None = None


def relevance_function(strategy: DistanceStrategy) -> Callable[[float], float]:
    """A distance turned into a relevance between 0 and 1, higher being nearer."""
    if strategy is DistanceStrategy.COSINE:
        return lambda distance: 1.0 - distance
    if strategy is DistanceStrategy.EUCLIDEAN_DISTANCE:
        return lambda distance: 1.0 - distance / math.sqrt(2)
    if strategy in (DistanceStrategy.MAX_INNER_PRODUCT, DistanceStrategy.DOT_PRODUCT):
        # pgvector's <#> is the negated inner product: -1 for unit vectors pointing the same way, 1
        # for opposite ones. Half of 1 - d keeps that order and runs from 1 to 0 between them.
        return lambda distance: (1.0 - distance) / 2.0
    raise ValueError(
        f"{strategy.value} has no relevance between 0 and 1; pass relevance_score_fn= to say "
        "what a distance means"
    )
