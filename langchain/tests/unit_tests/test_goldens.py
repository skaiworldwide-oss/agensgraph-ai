"""The recorded 0.3.1 outputs, and the helpers a later version is compared with them by."""

from __future__ import annotations

import json

import numpy as np
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from tests.goldens import golden


def test_known_texts_embed_to_their_vectors_and_others_to_a_fixed_one() -> None:
    known = {"a": np.array([1.0, 0.0], dtype=np.float32)}
    embedding = golden.FixedEmbeddings(known, dims=2)
    assert embedding.embed_documents(["a"]) == [[1.0, 0.0]]
    other = embedding.embed_query("b")
    assert other == golden.FixedEmbeddings({}, dims=2).embed_query("b")
    assert abs(sum(x * x for x in other) - 1.0) < 1e-6


def test_canonical_turns_documents_messages_and_tuples_into_json() -> None:
    value = {
        "doc": Document("text", id="1", metadata={"k": (1, 2)}),
        "message": HumanMessage("hi", id="m"),
        "array": np.array([1.5], dtype=np.float32),
        "raw": b"\x01",
    }
    plain = golden.canonical(value)
    assert plain["doc"] == {"id": "1", "page_content": "text", "metadata": {"k": [1, 2]}}
    assert plain["message"]["type"] == "human"
    assert plain["array"] == [1.5]
    assert plain["raw"] == "01"
    json.dumps(plain)


def test_matches_allows_float_noise_and_nothing_else() -> None:
    assert golden.matches({"d": [0.1, 2]}, {"d": [0.1 + 1e-9, 2]})
    assert not golden.matches({"d": [0.1]}, {"d": [0.11]})
    assert not golden.matches([1, 2], [2, 1])
    assert not golden.matches(True, 1)
    assert not golden.matches({"a": 1}, {"a": 1, "b": None})


def test_the_recording_covers_every_component() -> None:
    entries = golden.recorded()
    assert {entry["component"] for entry in entries} == set(golden.DATASETS) | {"checkpointer"}
    assert all(set(entry) == {"component", "step", "args", "result"} for entry in entries)
    assert golden.recorded("chat") == [e for e in entries if e["component"] == "chat"]
