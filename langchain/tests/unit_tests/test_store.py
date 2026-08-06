"""Unit tests for AgensStore's pure logic — no database."""

import pytest
from langgraph.store.base import (
    GetOp,
    InvalidNamespaceError,
    ListNamespacesOp,
    MatchCondition,
    PutOp,
    SearchOp,
)

from langchain_agensgraph.store.agensgraph import (
    AgensStore,
    _descendant_bounds,
    flatten_namespace,
    unflatten_namespace,
)


class TestNamespaceEncoding:
    def test_round_trip(self):
        for ns in [("a",), ("users", "alice"), ("a", "b", "c", "d")]:
            assert unflatten_namespace(flatten_namespace(ns)) == ns

    def test_flatten_joins_with_period(self):
        flat = flatten_namespace(("users", "alice", "memories"))
        assert flat == "users.alice.memories"

    def test_empty_namespace_rejected(self):
        with pytest.raises(InvalidNamespaceError):
            flatten_namespace(())

    def test_empty_label_rejected(self):
        with pytest.raises(InvalidNamespaceError):
            flatten_namespace(("users", ""))

    def test_period_in_label_rejected(self):
        # The separator must not appear in a label or the encoding would be ambiguous.
        # LangGraph forbids it too, which is what makes "." safe to use.
        with pytest.raises(InvalidNamespaceError):
            flatten_namespace(("users", "al.ice"))

    def test_langgraph_root_rejected(self):
        with pytest.raises(InvalidNamespaceError):
            flatten_namespace(("langgraph", "x"))


class TestDescendantBounds:
    def test_bounds_use_the_separator_successor(self):
        lo, hi = _descendant_bounds("users.u5")
        assert (lo, hi) == ("users.u5.", "users.u5/")

    def test_bounds_admit_descendants_and_exclude_siblings(self):
        lo, hi = _descendant_bounds("users.u5")
        assert lo <= "users.u5.memories" < hi
        assert lo <= "users.u5.a.b.c" < hi
        # a sibling sharing the prefix as a string is not a descendant
        assert not (lo <= "users.u5x" < hi)
        assert not (lo <= "users.u50" < hi)

    def test_over_selection_is_bounded_by_the_residual_filter(self):
        # The range admits labels whose next char sorts below "."; the store's residual
        # filter (prefix = p OR prefix >= p.) is what removes them.
        lo, hi = _descendant_bounds("users.u5")
        assert "users.u5!x" < hi  # inside the raw range ...
        assert "users.u5!x" < lo  # ... but excluded by the residual filter


class TestOpGrouping:
    def test_splits_by_kind_and_separates_deletes(self):
        ops = [
            GetOp(("a",), "k"),
            PutOp(("a",), "k", {"v": 1}),
            PutOp(("a",), "gone", None),  # value=None is a delete
            SearchOp(("a",)),
            ListNamespacesOp(),
        ]
        gets, puts, deletes, searches, lists = AgensStore._group(ops)
        assert [i for i, _ in gets] == [0]
        assert [i for i, _ in puts] == [1]
        assert [i for i, _ in deletes] == [2]
        assert [i for i, _ in searches] == [3]
        assert [i for i, _ in lists] == [4]

    def test_indices_allow_results_to_be_restored_to_op_order(self):
        ops = [SearchOp(("a",)), GetOp(("a",), "k"), SearchOp(("b",))]
        gets, _, _, searches, _ = AgensStore._group(ops)
        assert [i for i, _ in searches] == [0, 2]
        assert [i for i, _ in gets] == [1]

    def test_unknown_op_is_rejected(self):
        class Weird:
            pass

        with pytest.raises(NotImplementedError):
            AgensStore._group([Weird()])


class TestWildcardMatching:
    def test_prefix_wildcard(self):
        assert AgensStore._wildcard_ok("users.alice.memories", ("users", "*"), "prefix")
        assert not AgensStore._wildcard_ok("orgs.alice", ("users", "*"), "prefix")

    def test_suffix_wildcard(self):
        ok = AgensStore._wildcard_ok
        assert ok("users.alice.memories", ("*", "memories"), "suffix")
        assert not ok("users.alice.notes", ("*", "memories"), "suffix")

    def test_length_mismatch_does_not_match(self):
        assert not AgensStore._wildcard_ok("users", ("users", "*", "*"), "prefix")


class TestDepthTruncation:
    def test_truncates_and_deduplicates(self):
        prefixes = ["a.b.c", "a.b.d", "a.e"]
        assert AgensStore._truncate_depth(prefixes, 2) == ["a.b", "a.e"]

    def test_none_is_a_passthrough(self):
        prefixes = ["a.b.c", "a.e"]
        assert AgensStore._truncate_depth(prefixes, None) == prefixes


class TestPredicateBuilders:
    """The predicates are what decide whether a read can use an index."""

    def test_key_predicate_is_or_of_equalities_not_an_in_list(self):
        # A bound IN list degrades to a jsonb containment filter and loses the index.
        params: dict = {}
        out = AgensStore._key_predicate(
            AgensStore, [("a", "k1"), ("b", "k2")], params
        ).as_string()
        assert " OR " in out
        assert "IN" not in out
        assert params["p0"].obj == "a" and params["k0"].obj == "k1"
        assert params["p1"].obj == "b" and params["k1"].obj == "k2"

    def test_namespace_predicate_is_a_seekable_range(self):
        params: dict = {}
        out = AgensStore._namespace_predicate_named("users.u5", params, "").as_string()
        # a contiguous range the planner can turn into an Index Cond ...
        assert ">=" in out and "<" in out
        # ... plus the residual that removes the over-selected siblings
        assert " OR " in out
        assert params["ns_p"].obj == "users.u5"
        assert params["ns_lo"].obj == "users.u5."
        assert params["ns_hi"].obj == "users.u5/"

    def test_namespace_predicate_tags_keep_params_distinct(self):
        params: dict = {}
        AgensStore._namespace_predicate_named("a", params, "_mc0")
        AgensStore._namespace_predicate_named("b", params, "_mc1")
        assert params["ns_p_mc0"].obj == "a"
        assert params["ns_p_mc1"].obj == "b"


class TestMatchConditions:
    def test_wildcard_conditions_are_not_pushed_to_sql(self):
        op = ListNamespacesOp(
            match_conditions=(MatchCondition(match_type="prefix", path=("users", "*")),)
        )
        assert AgensStore._has_wildcard(op)
        assert AgensStore._match_predicate_sql(op, {}) is None

    def test_plain_conditions_are_pushed_to_sql(self):
        op = ListNamespacesOp(
            match_conditions=(MatchCondition(match_type="prefix", path=("users",)),)
        )
        assert not AgensStore._has_wildcard(op)
        assert AgensStore._match_predicate_sql(op, {}) is not None

    def test_suffix_condition_uses_ends_with(self):
        op = ListNamespacesOp(
            match_conditions=(MatchCondition(match_type="suffix", path=("memories",)),)
        )
        out = AgensStore._match_predicate_sql(op, {}).as_string()
        assert "ENDS WITH" in out
