"""Seeded test data: a seed gives the same data on every machine and every run.

Everything here is plain data; the tests turn it into whatever a component takes.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import numpy as np
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

VOCABULARY = (
    "graph vertex edge label property index vector search filter query plan scan join merge path "
    "node relation schema table column chunk document embedding distance cosine neighbour cluster "
    "hybrid keyword rank fusion score metadata store memory thread checkpoint message session"
).split()

RARE_WORD = "zymurgy"
"""A word in one chunk in a thousand, for a keyword search that is selective."""


@dataclass(frozen=True)
class Chunk:
    """A chunk of text, its metadata and its embedding."""

    id: str
    text: str
    metadata: dict[str, object]
    embedding: np.ndarray


@dataclass(frozen=True)
class Entity:
    id: str
    type: str
    properties: dict[str, object]


@dataclass(frozen=True)
class Link:
    source: str
    target: str
    type: str
    properties: dict[str, object]


@dataclass(frozen=True)
class GraphDoc:
    """A source text with the entities and relationships extracted from it."""

    source: str
    entities: list[Entity]
    links: list[Link]


def _ids(rng: np.random.Generator, n: int) -> list[str]:
    return [str(uuid.UUID(bytes=rng.bytes(16), version=4)) for _ in range(n)]


def embeddings(n: int, dims: int, *, seed: int = 0) -> np.ndarray:
    """``n`` unit-length float32 vectors of ``dims`` dimensions."""
    rng = np.random.default_rng(seed)
    vectors = rng.standard_normal((n, dims), dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors


def chunks(n: int, dims: int, *, seed: int = 0) -> list[Chunk]:
    """``n`` chunks with embeddings and metadata whose filters select known shares of them.

    ``s10``, ``s1`` and ``s01`` select 10 %, 1 % and 0.1 % for any one value. ``lang`` is a
    string, ``year`` an integer, ``price`` a float and ``flag`` a boolean, for the filter
    translator's types; ``note`` is on every other chunk only, for ``$exists`` and ``$ne``.
    """
    rng = np.random.default_rng(seed)
    ids = _ids(rng, n)
    vectors = embeddings(n, dims, seed=seed + 1)
    lengths = rng.integers(8, 21, size=n)
    out = []
    for i in range(n):
        words = list(rng.choice(VOCABULARY, size=int(lengths[i])))
        if i % 1000 == 0:
            words.append(RARE_WORD)
        metadata: dict[str, object] = {
            "s10": i % 10,
            "s1": i % 100,
            "s01": i % 1000,
            "lang": ("en", "ko", "ur")[i % 3],
            "year": 2000 + i % 25,
            "price": round(float(i % 997) * 0.25, 2),
            "flag": i % 2 == 0,
        }
        if i % 2 == 0:
            metadata["note"] = f"n{i}"
        out.append(Chunk(ids[i], " ".join(words), metadata, vectors[i]))
    return out


def graph_documents(
    n: int, *, entities: int = 200, per_doc: int = 6, seed: int = 0
) -> list[GraphDoc]:
    """``n`` documents drawing ``per_doc`` entities each from a shared pool of ``entities``.

    Entities recur across documents, so merging them by id is exercised, and a relationship can
    recur between the same pair, so deduplicating edges is too.
    """
    rng = np.random.default_rng(seed)
    types = ("Person", "Organization", "Place", "Concept")
    relations = ("KNOWS", "WORKS_AT", "LOCATED_IN", "MENTIONS")
    pool = [
        Entity(f"e{k}", types[k % len(types)], {"name": f"entity {k}", "rank": k % 50})
        for k in range(entities)
    ]
    docs = []
    for i in range(n):
        picked = [pool[int(k)] for k in rng.choice(entities, size=per_doc, replace=False)]
        links = [
            Link(
                a.id,
                b.id,
                relations[(i + j) % len(relations)],
                {"weight": round(float(rng.random()), 3)},
            )
            for j, (a, b) in enumerate(zip(picked, picked[1:], strict=False))
        ]
        source = " ".join(entity.properties["name"] for entity in picked)  # type: ignore[misc]
        docs.append(GraphDoc(f"document {i}: {source}", picked, links))
    return docs


def chat_sessions(
    sessions: int, messages: int, *, with_ids: bool = True, seed: int = 0
) -> dict[str, list[BaseMessage]]:
    """``sessions`` conversations of ``messages`` alternating human and AI messages."""
    rng = np.random.default_rng(seed)
    out: dict[str, list[BaseMessage]] = {}
    for s in range(sessions):
        history: list[BaseMessage] = []
        for m in range(messages):
            text = " ".join(rng.choice(VOCABULARY, size=6))
            kind = HumanMessage if m % 2 == 0 else AIMessage
            history.append(kind(content=text, id=f"s{s}m{m}" if with_ids else None))
        out[f"session-{s}"] = history
    return out


def store_items(
    n: int, *, namespaces: int = 10, depth: int = 3, seed: int = 0
) -> list[tuple[tuple[str, ...], str, dict[str, object]]]:
    """``n`` (namespace, key, value) items spread over ``namespaces`` namespaces ``depth`` deep."""
    rng = np.random.default_rng(seed)
    spaces = [tuple(f"ns{k}-{level}" for level in range(depth)) for k in range(namespaces)]
    return [
        (
            spaces[i % namespaces],
            f"key-{i}",
            {"text": " ".join(rng.choice(VOCABULARY, size=5)), "n": i, "tags": [f"t{i % 7}"]},
        )
        for i in range(n)
    ]
