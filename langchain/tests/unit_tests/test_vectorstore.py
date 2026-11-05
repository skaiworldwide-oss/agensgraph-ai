"""The vector store's filters, configuration and statements, without a server."""

from __future__ import annotations

import math

import pytest
from agensgraph import Distance, InTable, Jsonb, VectorStorage

from langchain_agensgraph.vectorstores import DistanceStrategy, IndexConfig, KeywordConfig
from langchain_agensgraph.vectorstores._sql import Layout, keyword_text
from langchain_agensgraph.vectorstores.config import distance_of, relevance_function
from langchain_agensgraph.vectorstores.filters import translate
from tests.filter_oracle import matches


def test_no_filter_is_true() -> None:
    assert translate(None) == ("TRUE", [])
    assert translate({}) == ("TRUE", [])


def test_equality_and_membership_are_containment() -> None:
    text, params = translate({"lang": "en"})
    assert text == "l.properties @> %s::jsonb"
    assert isinstance(params[0], Jsonb) and params[0].obj == {"lang": "en"}
    text, params = translate({"lang": {"$nin": ["en", "ko"]}})
    assert text == "NOT (l.properties @> ANY(%s::jsonb[]))"
    assert [p.obj for p in params[0]] == [{"lang": "en"}, {"lang": "ko"}]


def test_a_comparison_is_bracketed_by_the_values_type() -> None:
    text, params = translate({"year": {"$gte": 2020}})
    assert "jsonb_typeof((l.properties->'year')) = 'number'" in text
    assert "(l.properties->'year') >= %s::jsonb" in text
    text, _ = translate({"name": {"$lt": "m"}})
    assert "= 'string'" in text


def test_combinators_nest() -> None:
    text, params = translate({"$or": [{"a": 1}, {"$and": [{"b": 2}, {"$not": {"c": 3}}]}]})
    assert text.count("@>") == 3 and "NOT (" in text and " OR " in text
    assert len(params) == 3


def test_a_key_is_quoted_and_a_percent_doubled() -> None:
    text, _ = translate({"it's 50%": {"$exists": True}})
    assert "'it''s 50%%'" in text


def test_empty_membership_needs_no_parameter() -> None:
    assert translate({"a": {"$in": []}}) == ("FALSE", [])
    assert translate({"a": {"$nin": []}}) == ("TRUE", [])


@pytest.mark.parametrize(
    "where",
    [
        {"$bogus": 1},
        {"a": {"$bogus": 1}},
        {"a": None},
        {"a": {"$gt": True}},
        {"a": {"$gt": [1]}},
        {"a": {"$between": [1, "z"]}},
        {"a": {"$between": [1]}},
        {"a": {"$like": 3}},
        {"a": {"$exists": "yes"}},
        {"$and": []},
        {"a": {"$in": "abc"}},
        {"a": float("nan")},
        {"": 1},
        {"a\x00b": 1},
    ],
)
def test_a_filter_that_cannot_mean_one_thing_is_refused(where: dict) -> None:
    with pytest.raises(ValueError):
        translate(where)


@pytest.mark.parametrize(
    "where, metadata, expected",
    [
        ({"a": 1}, {"a": 1.0}, True),
        ({"a": 1}, {"a": True}, False),
        ({"a": True}, {"a": 1}, False),
        ({"a": {"$ne": 1}}, {}, True),
        ({"a": {"$gt": 5}}, {"a": "9"}, False),
        ({"a": {"$gt": "5"}}, {"a": "9"}, True),
        ({"a": {"$like": "n1%"}}, {"a": "n12"}, True),
        ({"a": {"$like": "n1_"}}, {"a": "n123"}, False),
        ({"a": {"$ilike": "N%"}}, {"a": "nb"}, True),
        ({"a": {"$exists": False}}, {"b": 1}, True),
        ({"a": {"$in": [1, 2]}}, {"a": [1, 2]}, False),
        ({"$not": {"a": 1}}, {"a": 2}, True),
    ],
)
def test_the_reference_evaluator(where: dict, metadata: dict, expected: bool) -> None:
    assert matches(where, metadata) is expected


def test_each_strategy_names_a_driver_distance() -> None:
    assert distance_of("COSINE") is Distance.COSINE
    assert distance_of(DistanceStrategy.TAXICAB) is Distance.L1
    with pytest.raises(ValueError, match="bit strings"):
        distance_of(DistanceStrategy.HAMMING)


def test_relevance_runs_from_one_at_no_distance() -> None:
    assert relevance_function(DistanceStrategy.COSINE)(0.0) == 1.0
    assert relevance_function(DistanceStrategy.EUCLIDEAN_DISTANCE)(math.sqrt(2)) == 0.0
    inner = relevance_function(DistanceStrategy.MAX_INNER_PRODUCT)
    assert (inner(-1.0), inner(0.0), inner(1.0)) == (1.0, 0.5, 0.0)
    # Nearer is never less relevant: an inner product of 0.5 outranks one of -0.1.
    assert inner(-0.5) > inner(0.1)
    with pytest.raises(ValueError, match="relevance_score_fn"):
        relevance_function(DistanceStrategy.TAXICAB)


def test_an_index_configuration_becomes_the_drivers() -> None:
    assert IndexConfig().driver().m == 16
    assert IndexConfig(method="ivfflat", lists=7).driver().lists == 7


def layout(**kwargs: object) -> Layout:
    storage = VectorStorage(
        "g",
        "Doc",
        dimensions=4,
        storage=InTable(schema="g_store", name="Doc__embedding__vec"),
    )
    return Layout(
        graph="g",
        label="Doc",
        text_property="text",
        storage=storage,
        keyword=KeywordConfig(),
        filter_keys=("year",),
        **kwargs,  # type: ignore[arg-type]
    )


def test_the_layout_names_everything_in_the_companion_schema() -> None:
    made = layout()
    assert made.vector_table == '"g_store"."Doc__embedding__vec"'
    assert made.text_table == '"g_store"."Doc__embedding__fts"'
    assert made.required()[:2] == [("g", "Doc"), ("g", "Doc__id__")]
    assert ("g", "Doc__meta__year") in made.required()
    assert ("g_store", "Doc__embedding__vec") in made.required()
    assert "ORDER BY distance LIMIT" in made.nearest("TRUE", embeddings=False, exact=False)
    assert "ORDER BY distance, l.id LIMIT" in made.nearest("TRUE", embeddings=False, exact=True)


def test_the_keyword_text_reads_the_named_properties_in_order() -> None:
    assert keyword_text({"text": "a", "title": "b", "n": 3}, ["title", "text", "n", "x"]) == (
        "b\na\n3"
    )
