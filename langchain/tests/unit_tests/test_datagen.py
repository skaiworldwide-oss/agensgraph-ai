"""The test data is what the tests assume it is: the same for a seed, and shaped as documented."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from tests import datagen


def test_a_seed_gives_the_same_chunks() -> None:
    first, again = datagen.chunks(50, 8, seed=7), datagen.chunks(50, 8, seed=7)
    assert [c.id for c in first] == [c.id for c in again]
    assert [c.text for c in first] == [c.text for c in again]
    assert all(np.array_equal(a.embedding, b.embedding) for a, b in zip(first, again, strict=True))


def test_another_seed_gives_other_chunks() -> None:
    assert datagen.chunks(5, 8, seed=1)[0].id != datagen.chunks(5, 8, seed=2)[0].id


@pytest.mark.parametrize("dims", [1, 4, 2000, 2001, 4000, 4001])
def test_embeddings_are_unit_float32_at_every_boundary(dims: int) -> None:
    vectors = datagen.embeddings(3, dims)
    assert vectors.dtype == np.float32 and vectors.shape == (3, dims)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


def test_the_filters_select_what_they_say() -> None:
    chunks = datagen.chunks(1000, 4)
    assert Counter(c.metadata["s10"] for c in chunks)[3] == 100
    assert Counter(c.metadata["s1"] for c in chunks)[42] == 10
    assert Counter(c.metadata["s01"] for c in chunks)[7] == 1
    assert sum("note" in c.metadata for c in chunks) == 500
    assert sum(datagen.RARE_WORD in c.text.split() for c in chunks) == 1


def test_ids_are_unique() -> None:
    chunks = datagen.chunks(2000, 4)
    assert len({c.id for c in chunks}) == 2000


def test_graph_documents_share_entities_and_link_only_their_own() -> None:
    docs = datagen.graph_documents(100, entities=40)
    seen = Counter(entity.id for doc in docs for entity in doc.entities)
    assert max(seen.values()) > 1, "no entity recurs, so merging is never exercised"
    for doc in docs:
        own = {entity.id for entity in doc.entities}
        assert all(link.source in own and link.target in own for link in doc.links)


def test_chat_sessions_alternate_and_carry_ids_when_asked() -> None:
    sessions = datagen.chat_sessions(2, 4)
    history = sessions["session-0"]
    assert [m.type for m in history] == ["human", "ai", "human", "ai"]
    assert all(m.id for m in history)
    assert not any(m.id for m in datagen.chat_sessions(1, 2, with_ids=False)["session-0"])


def test_store_items_spread_over_their_namespaces() -> None:
    items = datagen.store_items(30, namespaces=3, depth=2)
    assert len({namespace for namespace, _, _ in items}) == 3
    assert all(len(namespace) == 2 for namespace, _, _ in items)
