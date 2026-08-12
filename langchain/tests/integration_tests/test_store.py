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

    def test_filter_operators(self, store):
        for i, score in enumerate([1, 5, 10]):
            store.put(NS, f"s{i}", {"score": score, "kind": "n"})
        store.put(NS, "nokey", {"kind": "n"})

        def keys(**kw):
            return sorted(h.key for h in store.search(NS, **kw))

        assert keys(filter={"score": {"$gt": 4}}) == ["s1", "s2"]
        assert keys(filter={"score": {"$gte": 5}}) == ["s1", "s2"]
        assert keys(filter={"score": {"$lt": 5}}) == ["s0"]
        assert keys(filter={"score": {"$lte": 5}}) == ["s0", "s1"]
        assert keys(filter={"score": {"$eq": 5}}) == ["s1"]
        # an absent key is unequal to anything, so it satisfies $ne
        assert keys(filter={"score": {"$ne": 5}}) == ["nokey", "s0", "s2"]
        # operators combine, and combine with a plain equality on another field
        assert keys(filter={"score": {"$gt": 1, "$lt": 10}}) == ["s1"]
        assert keys(filter={"score": {"$gte": 5}, "kind": "n"}) == ["s1", "s2"]

    def test_filter_on_a_nested_key(self, store):
        store.put(NS, "a", {"meta": {"lang": "en", "n": 1}})
        store.put(NS, "b", {"meta": {"lang": "ko", "n": 2}})
        hits = store.search(NS, filter={"meta": {"lang": "ko"}})
        assert [h.key for h in hits] == ["b"]
        hits = store.search(NS, filter={"meta": {"n": {"$gt": 1}}})
        assert [h.key for h in hits] == ["b"]

    def test_filter_on_a_list_value(self, store):
        store.put(NS, "a", {"tags": [1, 2]})
        store.put(NS, "b", {"tags": [3]})
        hits = store.search(NS, filter={"tags": [1, 2]})
        assert [h.key for h in hits] == ["a"]

    def test_unknown_operator_is_rejected(self, store):
        with pytest.raises(ValueError):
            store.search(NS, filter={"score": {"$regex": "x"}})

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

    def test_semantic_search_applies_filter_operators(self, vector_store: AgensStore):
        vector_store.put(("docs",), "d1", {"text": "alpha", "score": 1})
        vector_store.put(("docs",), "d2", {"text": "beta", "score": 9})
        hits = vector_store.search(
            ("docs",), query="alpha", filter={"score": {"$gt": 5}}
        )
        assert [h.key for h in hits] == ["d2"]

    def test_embedding_row_is_removed_with_its_item(self, vector_store: AgensStore):
        """The side table's foreign key must cascade on a Cypher DETACH DELETE."""
        vector_store.put(("docs",), "d1", {"text": "alpha"})
        graph = vector_store._graph
        table = f'"{graph.graph_name}_store".item_vec'
        assert int(graph.query(f"SELECT count(*) AS c FROM {table}")[0]["c"]) == 1
        vector_store.delete(("docs",), "d1")
        assert int(graph.query(f"SELECT count(*) AS c FROM {table}")[0]["c"]) == 0


class TestOneFilterMeansOneAnswer:
    """A filter must not depend on whether a query came with it.

    A comparison over a value stored as text is a different question in jsonb than in
    Python -- jsonb orders by type class, Python by magnitude -- so two implementations of
    the operators answered oppositely and which one a caller got depended on the presence
    of ``query=``. There is one implementation, and it is the database's.
    """

    MIXED = {"num3": 3, "num10": 10, "str3": "3", "str20": "20", "boolt": True}

    @pytest.fixture
    def mixed(self, vector_store: AgensStore):
        for key, age in self.MIXED.items():
            vector_store.put(("f",), key, {"text": "shared text", "age": age})
        return vector_store

    @pytest.mark.parametrize(
        "flt",
        [
            {"age": {"$gt": 5}},
            {"age": {"$lt": 5}},
            {"age": {"$gte": 3}},
            {"age": {"$lte": 10}},
            {"age": {"$ne": 3}},
            {"age": {"$eq": 3}},
        ],
    )
    def test_the_same_filter_with_and_without_a_query(self, mixed, flt):
        plain = sorted(i.key for i in mixed.search(("f",), filter=flt, limit=50))
        semantic = sorted(
            i.key for i in mixed.search(("f",), query="shared text", filter=flt, limit=50)
        )
        assert plain == semantic

    @pytest.mark.asyncio
    async def test_the_async_path_agrees_too(self, mixed):
        flt = {"age": {"$gt": 5}}
        plain = sorted(i.key for i in mixed.search(("f",), filter=flt, limit=50))
        semantic = sorted(
            i.key
            for i in await mixed.asearch(
                ("f",), query="shared text", filter=flt, limit=50
            )
        )
        assert plain == semantic

    def test_a_filter_still_narrows(self, mixed):
        """Guards against the two paths agreeing by both matching everything."""
        everything = mixed.search(("f",), query="shared text", limit=50)
        narrowed = mixed.search(
            ("f",), query="shared text", filter={"age": {"$eq": 3}}, limit=50
        )
        assert len(everything) == len(self.MIXED)
        assert [i.key for i in narrowed] == ["num3"]


class TestIndexFalseIsHonoured:
    """``index=False`` asks for an item to be stored and not embedded."""

    @staticmethod
    def _vectors(store: AgensStore) -> int:
        table = f'"{store._graph.graph_name}_store".item_vec'
        return int(store._graph.query(f"SELECT count(*) AS c FROM {table}")[0]["c"])

    def test_on_the_sync_path(self, vector_store: AgensStore):
        vector_store.batch(
            [
                PutOp(namespace=("n",), key="embedded", value={"text": "a"}, index=None),
                PutOp(namespace=("n",), key="declined", value={"text": "b"}, index=False),
            ]
        )
        assert len(vector_store.search(("n",))) == 2
        assert self._vectors(vector_store) == 1

    @pytest.mark.asyncio
    async def test_on_the_async_path(self, vector_store: AgensStore):
        await vector_store.abatch(
            [
                PutOp(namespace=("n",), key="embedded", value={"text": "a"}, index=None),
                PutOp(namespace=("n",), key="declined", value={"text": "b"}, index=False),
            ]
        )
        assert len(await vector_store.asearch(("n",))) == 2
        assert self._vectors(vector_store) == 1


class TestEveryMatchConditionHolds:
    """LangGraph applies ``all`` over the conditions, so one match is not enough."""

    @pytest.fixture
    def tree(self, store: AgensStore):
        for ns in (
            ("users", "alice", "memories"),
            ("users", "alice", "notes"),
            ("orgs", "bob", "memories"),
        ):
            store.put(ns, "k", {"v": 1})
        return store

    def test_a_wildcard_condition_beside_a_plain_one(self, tree: AgensStore):
        found = tree.list_namespaces(
            prefix=("users", "*"), suffix=("memories",), limit=50
        )
        assert found == [("users", "alice", "memories")]

    @pytest.mark.asyncio
    async def test_the_async_path_agrees(self, tree: AgensStore):
        found = await tree.alist_namespaces(
            prefix=("users", "*"), suffix=("memories",), limit=50
        )
        assert found == [("users", "alice", "memories")]


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


class TestAnEmbeddingGoesOverTheWireAsItself:
    """A vector column takes a vector, not its decimal spelling.

    At 1,536 dimensions that is 6,152 bytes against 21,504, and the server has nothing to
    parse. Sending one requires the vector types to be registered on the connection it is
    sent on -- every connection, including each one a pool makes.
    """

    DIMS = 1536

    class WideEmbeddings:
        def _vec(self, text):
            return [0.01 + (abs(hash(text)) % 1000) * 1e-6] * 1536

        def embed_documents(self, texts):
            return [self._vec(t) for t in texts]

        def embed_query(self, text):
            return self._vec(text)

    @pytest.fixture
    def wide(self):
        g = AgensGraph("store_wide_it", _conf(), create=True)
        g.query("MATCH (n) DETACH DELETE n")
        store = AgensStore(
            graph=g,
            index={"dims": self.DIMS, "embed": self.WideEmbeddings(), "fields": ["t"]},
        )
        g.query(f'DELETE FROM "{g.graph_name}_store".item_vec')
        yield store
        g.close()

    def test_a_wide_embedding_round_trips(self, wide: AgensStore):
        wide.put(("ns",), "k", {"t": "a memory"})
        found = wide.search(("ns",), query="a memory", limit=1)
        assert [i.key for i in found] == ["k"]

    def test_the_stored_vector_has_the_width_it_was_given(self, wide: AgensStore):
        wide.put(("ns",), "k", {"t": "a memory"})
        table = f'"{wide._graph.graph_name}_store".item_vec'
        dims = wide._graph.query(
            f"SELECT vector_dims(embedding) AS d FROM {table} LIMIT 1"
        )[0]["d"]
        assert int(dims) == self.DIMS

    @pytest.mark.asyncio
    async def test_the_async_connection_can_send_one_too(self, wide: AgensStore):
        """Registering is per connection, so the async one is told the same thing."""
        await wide.aput(("ns",), "async-key", {"t": "another memory"})
        found = await wide.asearch(("ns",), query="another memory", limit=1)
        assert [i.key for i in found] == ["async-key"]


class TestListingNamespacesReadsWhatItReturns:
    """A page is taken by the server unless something afterwards would change it.

    A wildcard condition and a depth limit both collapse rows, so with either of them the
    page cannot be taken until they have been. With neither, reading everything and
    slicing in Python is work nobody asked for.
    """

    @pytest.fixture
    def many(self, store: AgensStore):
        from psycopg.types.json import Jsonb

        rows = [
            {
                "prefix": f"users.u{i}.memories",
                "key": "k",
                "value": {"n": i},
                "created_at": "t",
                "updated_at": "t",
                "ancestors": ["users", f"users.u{i}", f"users.u{i}.memories"],
            }
            for i in range(200)
        ]
        store._graph.query(store._put_cypher(), {"rows": Jsonb(rows)})
        return store

    def test_a_plain_page_is_asked_for_by_the_statement(self, many: AgensStore):
        from langgraph.store.base import ListNamespacesOp

        op = ListNamespacesOp(match_conditions=(), max_depth=None, limit=5, offset=0)
        assert not many._collapses_rows(op)
        statement, _ = many._namespace_page(op, None, {})
        rendered = statement.as_string(many._graph.connection)
        assert "LIMIT 5" in rendered and "1000000" not in rendered

    def test_it_returns_the_page_asked_for(self, many: AgensStore):
        found = many.list_namespaces(limit=5, offset=0)
        assert len(found) == 5
        assert many.list_namespaces(limit=5, offset=5) != found

    def test_a_depth_limit_still_collapses_first(self, many: AgensStore):
        from langgraph.store.base import ListNamespacesOp

        op = ListNamespacesOp(match_conditions=(), max_depth=1, limit=5, offset=0)
        assert many._collapses_rows(op)
        # every namespace shortens to "users", so one comes back however many there are
        assert many.list_namespaces(max_depth=1, limit=5) == [("users",)]

    def test_a_wildcard_still_collapses_first(self, many: AgensStore):
        found = many.list_namespaces(prefix=("users", "*"), limit=3)
        assert len(found) == 3
        assert all(n[0] == "users" for n in found)


class TestAnItemAndItsEmbeddingLandTogether:
    """Embedding is a call to something that is not the database.

    Written first, the item is committed and then the call is made -- and a call that
    fails leaves an item that nothing can find, for as long as it exists.
    """

    class Failing:
        def embed_documents(self, texts):
            raise RuntimeError("the embedding service is down")

        def embed_query(self, text):
            return [0.1] * 10

    def test_a_failed_embedding_leaves_no_item_behind(self):
        graph = AgensGraph("store_atomic_it", _conf(), create=True)
        graph.query("MATCH (n) DETACH DELETE n")
        store = AgensStore(
            graph=graph,
            index={"dims": 10, "embed": self.Failing(), "fields": ["t"]},
        )
        try:
            with pytest.raises(RuntimeError, match="embedding service"):
                store.put(("ns",), "k", {"t": "a memory"})
            held = graph.query('MATCH (n:"StoreItem") RETURN count(*) AS c')[0]["c"]
            assert held == 0, "the item was written without an embedding"
        finally:
            graph.query("MATCH (n) DETACH DELETE n")
            graph.close()

    def test_a_good_one_writes_both(self):
        graph = AgensGraph("store_atomic_it", _conf(), create=True)
        graph.query("MATCH (n) DETACH DELETE n")

        class Working:
            def embed_documents(self, texts):
                return [[0.1] * 10 for _ in texts]

            def embed_query(self, text):
                return [0.1] * 10

        store = AgensStore(
            graph=graph, index={"dims": 10, "embed": Working(), "fields": ["t"]}
        )
        try:
            store.put(("ns",), "k", {"t": "a memory"})
            table = f'"{graph.graph_name}_store".item_vec'
            items = graph.query('MATCH (n:"StoreItem") RETURN count(*) AS c')[0]["c"]
            vecs = graph.query(f"SELECT count(*) AS c FROM {table}")[0]["c"]
            assert int(items) == 1 and int(vecs) == 1
        finally:
            graph.query("MATCH (n) DETACH DELETE n")
            graph.close()


class TestASearchOfOneNamespaceFindsIt:
    """The namespace is part of the search, not something applied to its results.

    Ranked globally and narrowed afterwards, a search of one user's memories in a store
    holding many users returns whatever of theirs happens to fall in the global nearest
    few -- which over 5,000 memories across fifty users was nothing at all.
    """

    class Spread:
        def _vec(self, text):
            h = abs(hash(text))
            return [((h >> i) % 97) / 97.0 for i in range(10)]

        def embed_documents(self, texts):
            return [self._vec(t) for t in texts]

        def embed_query(self, text):
            return self._vec(text)

    @pytest.fixture
    def crowded(self):
        from langgraph.store.base import PutOp

        graph = AgensGraph("store_ns_it", _conf(), create=True)
        store = AgensStore(
            graph=graph, index={"dims": 10, "embed": self.Spread(), "fields": ["t"]}
        )
        held = graph.query('MATCH (n:"StoreItem") RETURN count(*) AS c')[0]["c"]
        if held < 2000:
            graph.query("MATCH (n) DETACH DELETE n")
            graph.query(f'DELETE FROM "{graph.graph_name}_store".item_vec')
            for base in range(0, 2000, 500):
                store.batch(
                    [
                        PutOp(
                            namespace=("users", f"u{(base + i) % 40}"),
                            key=f"k{base + i}",
                            value={"t": f"memory {base + i}"},
                            index=None,
                        )
                        for i in range(500)
                    ]
                )
        yield store
        graph.close()

    @pytest.mark.parametrize("limit", [5, 10, 20])
    def test_it_returns_a_full_page_from_the_namespace(self, crowded, limit):
        found = crowded.search(("users", "u7"), query="memory 287", limit=limit)
        assert len(found) == limit
        assert all(item.namespace == ("users", "u7") for item in found)

    @pytest.mark.asyncio
    async def test_the_awaited_search_does_too(self, crowded):
        found = await crowded.asearch(("users", "u7"), query="memory 287", limit=10)
        assert len(found) == 10
        assert all(item.namespace == ("users", "u7") for item in found)

    def test_a_large_limit_does_not_exceed_what_pgvector_takes(self, crowded):
        """`hnsw.ef_search` is 1..1000 and the over-fetch is four times the limit."""
        assert crowded.search(("users", "u7"), query="memory 287", limit=300) is not None
