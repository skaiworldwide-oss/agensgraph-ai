"""The store's pure parts: namespace matching, filters, batching, index configuration."""

from __future__ import annotations

import re

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langgraph.store.base import PutOp

from langchain_agensgraph.store.agensgraph import (
    _filter,
    _Index,
    _last_puts,
    _match,
    _within,
)


@pytest.mark.parametrize(
    "path, kind, namespace, expected",
    [
        (("a", "*", "c"), "prefix", "a.b.c.d", True),
        (("a", "*", "c"), "prefix", "a.b.x", False),
        (("a",), "prefix", "ab.c", False),
        (("c",), "suffix", "a.b.c", True),
        (("b", "*"), "suffix", "a.b.c", True),
        (("x.y",), "prefix", "xzy", False),
    ],
)
def test_a_namespace_matches_on_whole_labels(
    path: tuple[str, ...], kind: str, namespace: str, expected: bool
) -> None:
    assert (re.search(_match(path, kind), namespace) is not None) is expected


def test_a_prefix_condition_escapes_like_wildcards() -> None:
    params: list[object] = []
    assert _within(("user_1", "100%"), params).startswith("(s.prefix = %s OR")
    assert params == ["user_1.100%", "user\\_1.100\\%.%"]
    assert _within((), []) == "TRUE"


def test_a_filter_follows_the_in_memory_store() -> None:
    params: list[object] = []
    text = _filter({"n": {"$gt": 3}, "tags": ["t1"], "meta": {"lang": "en"}, "gone": None}, params)
    assert "jsonb_typeof(s.value->'n') = 'number'" in text
    assert "jsonb_array_length(s.value->'tags') = 1" in text
    assert "jsonb_typeof(s.value->'meta') = 'object'" in text
    assert "(s.value->'gone' IS NULL OR s.value->'gone' = 'null'::jsonb)" in text
    assert params[1] == 3.0


@pytest.mark.parametrize("where", [{"n": {"$gt": "3"}}, {"n": {"$regex": "x"}}, {1: "x"}])
def test_a_filter_it_cannot_express_is_refused(where: dict) -> None:
    with pytest.raises(ValueError):
        _filter(where, [])


def test_the_last_put_to_a_key_is_the_one_kept() -> None:
    ops = [PutOp(("a",), "k", {"v": 1}), PutOp(("a",), "k", None), PutOp(("b",), "k", {"v": 3})]
    kept = _last_puts(ops)
    assert kept[(("a",), "k")].value is None and kept[(("b",), "k")].value == {"v": 3}


def test_an_index_configuration_is_checked() -> None:
    embed = DeterministicFakeEmbedding(size=4)
    index = _Index.of({"dims": 4, "embed": embed, "fields": ["text", "tags[*]"]})
    assert index.kind == "hnsw" and index.operator == "<=>" and index.vectors_per_item == 3
    assert index.operator_class == "vector_cosine_ops"
    ivf = _Index.of(
        {
            "dims": 4,
            "embed": embed,
            "ann_index_config": {"kind": "ivfflat", "nlist": 7, "vector_type": "halfvec"},
            "distance_type": "l2",
        }
    )
    assert ivf.parameters == {"lists": 7} and ivf.operator_class == "halfvec_l2_ops"
    for bad in (
        {"dims": 0, "embed": embed},
        {"dims": 4, "embed": embed, "ann_index_config": {"kind": "gist"}},
        {"dims": 4, "embed": embed, "distance_type": "hamming"},
    ):
        with pytest.raises(ValueError):
            _Index.of(bad)
