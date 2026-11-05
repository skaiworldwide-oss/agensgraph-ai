"""Recorded outputs of the released 0.3.1, and how a later version is compared with them.

``record.py`` runs scripted calls through 0.3.1 on every server it supports and writes what each
returned, one JSON line per call, to ``0.3.1.jsonl``; 0.3.1 answers alike on all of them. A test of
the rewrite replays the same calls on the same data, on every leg, and compares with
:func:`matches`. The data is ``datagen`` at the sizes and seeds in :data:`DATASETS`, embedded by
:class:`FixedEmbeddings`, so both sides see identical vectors.

Values are kept in a canonical JSON form: documents, messages and store items as plain maps,
tuples as lists, and nothing that depends on when or where a run happened (timestamps, graph ids).
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import BaseMessage, message_to_dict

RECORDING = Path(__file__).with_name("0.3.1.jsonl")

DATASETS: Mapping[str, Mapping[str, int]] = {
    "vectorstore": {"chunks": 200, "dims": 8, "seed": 7},
    "graph": {"documents": 20, "entities": 15, "per_doc": 4, "seed": 3},
    "chat": {"sessions": 3, "messages": 6, "seed": 5},
    "store": {"items": 30, "namespaces": 4, "depth": 3, "seed": 9},
}

REL_TOL = 1e-6


class FixedEmbeddings(Embeddings):
    """Known texts embed to the vectors given; any other text to a vector seeded by its hash."""

    def __init__(self, known: Mapping[str, np.ndarray], dims: int) -> None:
        self.known = {text: np.asarray(vector, dtype=np.float32) for text, vector in known.items()}
        self.dims = dims

    def vector(self, text: str) -> list[float]:
        if text in self.known:
            return [float(x) for x in self.known[text]]
        seed = int.from_bytes(hashlib.blake2b(text.encode(), digest_size=8).digest(), "big")
        vector = np.random.default_rng(seed).standard_normal(self.dims).astype(np.float32)
        return [float(x) for x in vector / np.linalg.norm(vector)]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.vector(text)


def canonical(value: Any) -> Any:
    """``value`` as plain JSON values, with nothing in it that differs between runs."""
    if isinstance(value, Document):
        return {
            "id": value.id,
            "page_content": value.page_content,
            "metadata": canonical(value.metadata),
        }
    if isinstance(value, BaseMessage):
        return canonical(message_to_dict(value))
    if isinstance(value, Mapping):
        return {str(key): canonical(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return [canonical(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return canonical(value.item())
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, (list, tuple)):
        return [canonical(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((canonical(item) for item in value), key=json.dumps)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"no canonical form for {type(value).__name__}: {value!r}")


def matches(golden: Any, actual: Any, *, rel_tol: float = REL_TOL) -> bool:
    """Whether two canonical values are equal, floats to ``rel_tol``."""
    if isinstance(golden, bool) or isinstance(actual, bool):
        return golden is actual
    if isinstance(golden, (int, float)) and isinstance(actual, (int, float)):
        return math.isclose(golden, actual, rel_tol=rel_tol, abs_tol=rel_tol)
    if isinstance(golden, dict) and isinstance(actual, dict):
        return golden.keys() == actual.keys() and all(
            matches(golden[key], actual[key], rel_tol=rel_tol) for key in golden
        )
    if isinstance(golden, list) and isinstance(actual, list):
        return len(golden) == len(actual) and all(
            matches(g, a, rel_tol=rel_tol) for g, a in zip(golden, actual, strict=True)
        )
    return type(golden) is type(actual) and golden == actual


def line(component: str, step: str, args: Mapping[str, Any], result: Any) -> str:
    """One recorded call as a JSON line."""
    entry = {
        "component": component,
        "step": step,
        "args": canonical(args),
        "result": canonical(result),
    }
    return json.dumps(entry, sort_keys=True, ensure_ascii=False)


def recorded(component: str | None = None) -> list[dict[str, Any]]:
    """The recorded calls, in the order they were made; only one component's if named."""
    entries = [json.loads(text) for text in RECORDING.read_text().splitlines() if text]
    return [entry for entry in entries if component in (None, entry["component"])]
