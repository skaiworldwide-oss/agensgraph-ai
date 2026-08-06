"""Integration tests for AgensStore (LangGraph BaseStore)."""

import os

import pytest
from langgraph.store.base import GetOp, PutOp, SearchOp

from langchain_agensgraph import AgensGraph, AgensStore

from .fake_embeddings import ConsistentFakeEmbeddings


def _conf():
    return {
        "dbname": os.getenv("AGENSGRAPH_DB"),
        "user": os.getenv("AGENSGRAPH_USER"),
        "password": os.getenv("AGENSGRAPH_PASSWORD"),
        "host": os.getenv("AGENSGRAPH_HOST", "localhost"),
        "port": int(os.getenv("AGENSGRAPH_PORT", 5432)),
    }


@pytest.fixture
def store():
    g = AgensGraph("store_it", _conf(), create=True)
    g.query("MATCH (n) DETACH DELETE n")
    s = AgensStore(graph=g)
    yield s
    g.close()


@pytest.fixture
def vector_store():
    g = AgensGraph("store_vec_it", _conf(), create=True)
    g.query("MATCH (n) DETACH DELETE n")
    s = AgensStore(
        graph=g,
        index={"dims": 10, "embed": ConsistentFakeEmbeddings(), "fields": ["text"]},
    )
    yield s
    g.close()


NS = ("users", "alice", "memories")


class TestRoundTrip:
    def test_put_then_get(self, store: AgensStore):
        store.put(NS, "m1", {"text": "likes tea"})
        item = store.get(NS, "m1")
        assert item is not None
        assert item.value == {"text": "likes tea"}
        assert item.namespace == NS
        assert item.key == "m1"

    def test_get_missing_returns_none(self, store: AgensStore):
        assert store.get(NS, "absent") is None

    def test_put_is_an_upsert_that_preserves_created_at(self, store: AgensStore):
        store.put(NS, "m1", {"v": 1})
        first = store.get(NS, "m1")
        store.put(NS, "m1", {"v": 2})
        second = store.get(NS, "m1")
        assert second.value == {"v": 2}
        assert second.created_at == first.created_at
        # exactly one vertex, not two
        assert len(store.search(NS)) == 1

    def test_delete(self, store: AgensStore):
        store.put(NS, "m1", {"v": 1})
        store.delete(NS, "m1")
        assert store.get(NS, "m1") is None

    def test_values_survive_nesting(self, store: AgensStore):
        value = {"a": {"b": [1, 2, {"c": "d"}]}, "n": 3, "flag": True}
        store.put(NS, "nested", value)
        assert store.get(NS, "nested").value == value


class TestNamespaces:
    def _seed(self, store: AgensStore):
        store.put(("users", "alice"), "profile", {"t": "root"})
        store.put(("users", "alice", "memories"), "m1", {"t": "a1", "topic": "x"})
        store.put(("users", "alice", "memories"), "m2", {"t": "a2", "topic": "y"})
        store.put(("users", "alice", "notes"), "n1", {"t": "a3"})
        store.put(("users", "bob", "memories"), "m1", {"t": "b1"})

    def test_search_returns_the_namespace_and_its_descendants(self, store):
        self._seed(store)
        hits = store.search(("users", "alice"))
        assert {(h.namespace, h.key) for h in hits} == {
            (("users", "alice"), "profile"),
            (("users", "alice", "memories"), "m1"),
            (("users", "alice", "memories"), "m2"),
            (("users", "alice", "notes"), "n1"),
        }

    def test_search_does_not_leak_a_sibling_namespace(self, store):
        self._seed(store)
        hits = store.search(("users", "alice"))
        assert all("bob" not in h.namespace for h in hits)

    def test_search_on_a_leaf_namespace(self, store):
        self._seed(store)
        hits = store.search(("users", "alice", "memories"))
        assert {h.key for h in hits} == {"m1", "m2"}

    def test_string_prefix_is_not_a_namespace_prefix(self, store):
        """"users.alice" must not match a namespace merely starting with those bytes."""
        store.put(("users", "alice"), "k", {"v": 1})
        store.put(("users", "alicia"), "k", {"v": 2})
        hits = store.search(("users", "alice"))
        assert {h.namespace for h in hits} == {("users", "alice")}

    def test_filter(self, store):
        self._seed(store)
        hits = store.search(("users", "alice", "memories"), filter={"topic": "y"})
        assert [h.key for h in hits] == ["m2"]

    def test_limit_and_offset(self, store):
        for i in range(5):
            store.put(NS, f"k{i}", {"i": i})
        first = store.search(NS, limit=2)
        second = store.search(NS, limit=2, offset=2)
        assert len(first) == 2 and len(second) == 2
        assert {h.key for h in first}.isdisjoint({h.key for h in second})


class TestListNamespaces:
    def _seed(self, store: AgensStore):
        store.put(("users", "alice", "memories"), "k", {"v": 1})
        store.put(("users", "bob", "memories"), "k", {"v": 1})
        store.put(("orgs", "acme", "notes"), "k", {"v": 1})

    def test_all(self, store):
        self._seed(store)
        assert set(store.list_namespaces()) == {
            ("users", "alice", "memories"),
            ("users", "bob", "memories"),
            ("orgs", "acme", "notes"),
        }

    def test_prefix(self, store):
        self._seed(store)
        assert set(store.list_namespaces(prefix=("users",))) == {
            ("users", "alice", "memories"),
            ("users", "bob", "memories"),
        }

    def test_suffix(self, store):
        self._seed(store)
        assert set(store.list_namespaces(suffix=("memories",))) == {
            ("users", "alice", "memories"),
            ("users", "bob", "memories"),
        }

    def test_wildcard_prefix(self, store):
        self._seed(store)
        got = set(store.list_namespaces(prefix=("users", "*", "memories")))
        assert got == {("users", "alice", "memories"), ("users", "bob", "memories")}

    def test_max_depth_truncates_and_deduplicates(self, store):
        self._seed(store)
        assert set(store.list_namespaces(max_depth=2)) == {
            ("users", "alice"),
            ("users", "bob"),
            ("orgs", "acme"),
        }


class TestBatch:
    def test_results_come_back_in_op_order(self, store: AgensStore):
        store.put(("a",), "k1", {"v": 1})
        store.put(("b",), "k2", {"v": 2})
        results = store.batch(
            [
                GetOp(("b",), "k2"),
                GetOp(("a",), "k1"),
                GetOp(("a",), "missing"),
            ]
        )
        assert [r.value["v"] if r else None for r in results] == [2, 1, None]

    def test_mixed_kinds_in_one_batch(self, store: AgensStore):
        store.put(("a",), "old", {"v": 0})
        results = store.batch(
            [
                PutOp(("a",), "new", {"v": 1}),
                PutOp(("a",), "old", None),  # delete
                SearchOp(("a",)),
            ]
        )
        assert results[0] is None and results[1] is None
        # the search runs after the writes in the same batch
        assert {i.key for i in results[2]} == {"new"}

    def test_batch_put_is_one_statement_for_many_items(self, store: AgensStore):
        ops = [PutOp(("bulk",), f"k{i}", {"i": i}) for i in range(50)]
        store.batch(ops)
        assert len(store.search(("bulk",), limit=100)) == 50


class TestSemanticSearch:
    def test_query_ranks_the_matching_item_first(self, vector_store: AgensStore):
        vector_store.put(("docs",), "d1", {"text": "alpha"})
        vector_store.put(("docs",), "d2", {"text": "beta"})
        vector_store.put(("docs",), "d3", {"text": "gamma"})
        hits = vector_store.search(("docs",), query="gamma", limit=3)
        assert hits
        assert hits[0].key == "d3"
        assert hits[0].score is not None

    def test_semantic_search_respects_the_namespace(self, vector_store: AgensStore):
        vector_store.put(("docs", "a"), "d1", {"text": "alpha"})
        vector_store.put(("docs", "b"), "d2", {"text": "beta"})
        hits = vector_store.search(("docs", "a"), query="beta", limit=5)
        assert all(h.namespace == ("docs", "a") for h in hits)

    def test_embedding_row_is_removed_with_its_item(self, vector_store: AgensStore):
        """The side table's foreign key must cascade on a Cypher DETACH DELETE."""
        vector_store.put(("docs",), "d1", {"text": "alpha"})
        graph = vector_store._graph
        table = f'"{graph.graph_name}_store".item_vec'
        assert int(graph.query(f"SELECT count(*) AS c FROM {table}")[0]["c"]) == 1
        vector_store.delete(("docs",), "d1")
        assert int(graph.query(f"SELECT count(*) AS c FROM {table}")[0]["c"]) == 0


class TestAsyncParity:
    @pytest.mark.asyncio
    async def test_async_round_trip(self, store: AgensStore):
        await store.aput(NS, "m1", {"v": 1})
        item = await store.aget(NS, "m1")
        assert item.value == {"v": 1}
        assert item.namespace == NS

    @pytest.mark.asyncio
    async def test_async_search_and_list(self, store: AgensStore):
        await store.aput(("users", "alice"), "k", {"v": 1})
        await store.aput(("users", "bob"), "k", {"v": 2})
        hits = await store.asearch(("users",))
        assert len(hits) == 2
        nss = await store.alist_namespaces()
        assert set(nss) == {("users", "alice"), ("users", "bob")}

    @pytest.mark.asyncio
    async def test_async_delete(self, store: AgensStore):
        await store.aput(NS, "m1", {"v": 1})
        await store.adelete(NS, "m1")
        assert await store.aget(NS, "m1") is None


class TestQueryPlans:
    """The perf contract is only real if a test enforces it.

    A read that silently falls back to a sequential scan still returns the right rows,
    so correctness tests cannot catch it. These assert the plan shape instead.
    """

    @staticmethod
    def _plan(store: AgensStore, stmt, params) -> str:
        rows = store._graph.query("EXPLAIN (COSTS OFF) " + stmt.as_string(), params)
        return "\n".join(str(next(iter(r.values()))) for r in rows)

    @pytest.fixture
    def loaded(self, store: AgensStore):
        # Enough rows that a sequential scan is a plausible plan; with a handful of
        # rows the planner would pick one no matter how the query is written.
        from psycopg.types.json import Jsonb

        rows = [
            {
                "prefix": f"users.u{i % 200}.memories",
                "key": f"k{i}",
                "value": {"n": i},
                "created_at": "t",
                "updated_at": "t",
            }
            for i in range(5000)
        ]
        store._graph.query(store._put_cypher(), {"rows": Jsonb(rows)})
        store._graph.query(
            f'ANALYZE "{store._graph.graph_name}"."{store._label}"'
        )
        return store

    def test_get_uses_the_composite_index(self, loaded: AgensStore):
        params: dict = {}
        pred = loaded._key_predicate([("users.u7.memories", "k7")], params)
        plan = self._plan(loaded, loaded._get_cypher(pred), params)
        assert "Index Scan" in plan
        assert "Seq Scan" not in plan

    def test_mget_uses_the_index_for_every_key(self, loaded: AgensStore):
        params: dict = {}
        pairs = [(f"users.u{i}.memories", f"k{i}") for i in (7, 8, 9)]
        pred = loaded._key_predicate(pairs, params)
        plan = self._plan(loaded, loaded._get_cypher(pred), params)
        assert "Seq Scan" not in plan
        # one index scan per key, unioned
        assert plan.count("Bitmap Index Scan") == len(pairs)

    def test_namespace_search_seeks_rather_than_filters(self, loaded: AgensStore):
        params: dict = {}
        pred = loaded._namespace_predicate("users.u7", params)
        plan = self._plan(loaded, loaded._search_cypher(pred, 10, 0), params)
        assert "Seq Scan" not in plan
        # the range must reach the index, not sit in a post-scan Filter
        assert "Index Cond" in plan or "Recheck Cond" in plan
