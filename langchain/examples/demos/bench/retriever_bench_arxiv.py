"""The retriever numbers again, on 138k real vertices instead of a synthetic ring.

Runs against the ``arxiv`` graph demo 01 builds (50k papers, 88k authors, 275k
edges). Query vectors are sampled from the stored embeddings, so nothing here
calls an embedding API -- every millisecond is the database.

This graph has what the synthetic one deliberately lacks: real hubs. 147
Category vertices carry 75k edges and 17 Year vertices carry 50k, so an
UNFILTERED two-hop expansion walks through them to a large slice of the corpus.
That arm runs under a statement timeout and is reported as the cautionary
number; the ``relationship_type`` filter (co-author expansion over AUTHORED_BY
alone) is the fast shape the retriever documents for graphs like this.

    cd langchain
    .venv/bin/python examples/demos/bench/retriever_bench_arxiv.py
"""

from __future__ import annotations

import asyncio
import os
import pathlib
import statistics
import sys
import time
from typing import Any, Callable, Dict, List

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import psycopg
from langchain_core.embeddings import Embeddings

from _common import config, console
from langchain_agensgraph.engine import AgensEngine
from langchain_agensgraph.retrievers import (
    AgensGraphContextRetriever,
    AgensVectorRetriever,
)
from langchain_agensgraph.retrievers.graph_context import build_expansion_query
from langchain_agensgraph.vectorstores.agensgraph_vector import AgensgraphVector

GRAPH = "arxiv"
REPS = 15
WARMUP = 3
K = 8
HUB_REPS = 3
HUB_TIMEOUT = "15s"


def _url() -> str:
    return os.getenv("AGENSGRAPH_URL") or config.url()


class LookupEmbeddings(Embeddings):
    """Query texts are keys into vectors already read from the store."""

    def __init__(self, mapping: Dict[str, List[float]]) -> None:
        self.mapping = mapping

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.mapping[t] for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self.mapping[text]


def sample_queries(store: AgensgraphVector) -> Dict[str, List[float]]:
    rows = store.query(
        'MATCH (p:"Paper") RETURN p.embedding AS e ORDER BY p.id LIMIT %(n)s',
        params={"n": REPS + WARMUP},
    )
    return {f"q{i}": [float(x) for x in row["e"]] for i, row in enumerate(rows)}


def median_ms(fn: Callable[[str], Any], queries: List[str]) -> float:
    """Time an already-warm workload; warming happens once, for every arm.

    The graph is bigger than the buffer cache, so an arm that runs first pays
    the reads an arm that runs second inherits for free -- measuring per-arm
    with per-arm warmup made the ORDER of the arms the biggest term in the
    result. Every arm is warmed over every query before anything is timed.
    """
    times = []
    for q in queries[WARMUP:]:
        t0 = time.perf_counter()
        fn(q)
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


def naive_two_step(
    store: AgensgraphVector, hops: int, rel_type: str | None
) -> Callable[[str], Any]:
    """The client-side shape: the seed search, then one query per seed."""
    rel = f':"{rel_type}"' if rel_type else ""
    per_seed = (
        'MATCH (seed:"Paper") WHERE seed.id = %(sid)s '
        + f"OPTIONAL MATCH (seed)-[rels{rel}*1..{hops}]-(peer) "
        + "WITH collect(DISTINCT CASE WHEN peer IS NULL THEN NULL ELSE "
        + "jsonb_build_object('id', id(peer), 'label', label(peer), "
        + "'properties', properties(peer) || jsonb_build_object('embedding', Null)) "
        + "END)[0..%(mcn)s] AS ctx_nodes, "
        + "collect(DISTINCT CASE WHEN rels IS NULL THEN NULL ELSE "
        + "jsonb_build_object('type', label(rels[-1]), 'start', start_id(rels[-1]), "
        + "'end', end_id(rels[-1]), 'properties', properties(rels[-1])) "
        + "END)[0..%(mcr)s] AS ctx_rels "
        + "RETURN ctx_nodes, ctx_rels"
    )

    def run(query: str) -> None:
        hits = store.similarity_search_with_score(query, k=K)
        for doc, _score in hits:
            store.query(
                per_seed, params={"sid": doc.metadata["id"], "mcn": 20, "mcr": 20}
            )

    return run


def hub_arm(store: AgensgraphVector, vectors: Dict[str, List[float]]) -> str:
    """Unfiltered two hops through the Category/Year hubs, under a timeout."""
    statement, params, _ = store._build_search(
        vectors["q0"], K, retrieval_query=build_expansion_query(hops=2, hybrid=False)
    )
    rendered = statement.as_string()
    times = []
    with psycopg.connect(_url()) as conn:
        cur = conn.cursor()
        cur.execute(f"SET graph_path = {GRAPH}")
        cur.execute(f"SET statement_timeout = '{HUB_TIMEOUT}'")
        for i in range(HUB_REPS):
            vec = vectors[f"q{i}"]
            run_params = dict(
                params,
                embedding="[" + ",".join(map(str, vec)) + "]",
                max_context_nodes=20,
                max_context_rels=20,
            )
            t0 = time.perf_counter()
            try:
                cur.execute(rendered, run_params)
                cur.fetchall()
                times.append((time.perf_counter() - t0) * 1000)
            except psycopg.errors.QueryCanceled:
                conn.rollback()
                return f"timed out at {HUB_TIMEOUT} (this is the point)"
    return f"{statistics.median(times):.0f} ms median of {HUB_REPS}"


def explain_filtered(store: AgensgraphVector) -> None:
    statement, params, _ = store._build_search(
        [0.0] * 1536,
        K,
        retrieval_query=build_expansion_query(
            hops=2, hybrid=False, relationship_type="AUTHORED_BY"
        ),
    )
    params = dict(
        params,
        embedding="[" + ",".join(["0.0"] * 1536) + "]",
        max_context_nodes=20,
        max_context_rels=20,
    )
    with psycopg.connect(_url()) as conn:
        cur = conn.cursor()
        cur.execute(f"SET graph_path = {GRAPH}")
        cur.execute("SET enable_seqscan = off")
        cur.execute("EXPLAIN (COSTS OFF) " + statement.as_string(), params)
        plan = "\n".join(r[0] for r in cur.fetchall())
    checks = {
        "vector index drives the seeds": "paper_vec" in plan,
        "expansion is the VLE machinery": "VLE" in plan,
        "no Seq Scan anywhere": "Seq Scan" not in plan,
    }
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if not all(checks.values()):
        print("    plan:\n      " + plan.replace("\n", "\n      "))


def async_throughput(retriever: AgensGraphContextRetriever, queries: List[str]):
    jobs = [queries[i % len(queries)] for i in range(48)]

    async def run() -> Dict[str, float]:
        await retriever.ainvoke(jobs[0])
        t0 = time.perf_counter()
        for q in jobs:
            await retriever.ainvoke(q)
        serial_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        for at in range(0, len(jobs), 16):
            await asyncio.gather(*(retriever.ainvoke(q) for q in jobs[at : at + 16]))
        concurrent_s = time.perf_counter() - t0
        return {
            "serial_per_s": len(jobs) / serial_s,
            "concurrent_per_s": len(jobs) / concurrent_s,
        }

    return asyncio.run(run())


def main() -> None:
    print("loadavg before:", open("/proc/loadavg").read().strip())
    engine = AgensEngine(_url(), max_size=16)
    probe = AgensgraphVector.from_existing_index(
        embedding=LookupEmbeddings({}),
        engine=engine,
        graph_name=GRAPH,
        node_label="Paper",
        text_node_property="abstract",
        embedding_node_property="embedding",
        index_name="paper_vec",
    )
    vectors = sample_queries(probe)
    probe.close()
    store = AgensgraphVector.from_existing_index(
        embedding=LookupEmbeddings(vectors),
        engine=engine,
        graph_name=GRAPH,
        node_label="Paper",
        text_node_property="abstract",
        embedding_node_property="embedding",
        index_name="paper_vec",
    )
    queries = list(vectors.keys())
    counts = store.query(
        "MATCH (n) WITH count(n) AS v MATCH ()-[e]->() RETURN v, count(e) AS e"
    )[0]
    console.kv("graph", f"{GRAPH}: {counts['v']:,} vertices, {counts['e']:,} edges")

    plain = AgensVectorRetriever(store=store, k=K)
    ctx1 = AgensGraphContextRetriever(store=store, k=K, expand_by_hops=1)
    ctx2 = AgensGraphContextRetriever(
        store=store, k=K, expand_by_hops=2, relationship_type="AUTHORED_BY"
    )
    naive1 = naive_two_step(store, 1, None)
    naive2 = naive_two_step(store, 2, "AUTHORED_BY")

    console.section("warming every arm over every query")
    with console.timer("warmup"):
        for fn in (plain.invoke, ctx1.invoke, naive1, ctx2.invoke, naive2):
            for q in queries:
                fn(q)

    console.section("1. seed-only retriever (k=8)")
    console.kv("AgensVectorRetriever", f"{median_ms(plain.invoke, queries):.2f} ms")

    console.section("2. one-hop context, one statement vs one query per seed")
    ms_one = median_ms(ctx1.invoke, queries)
    ms_naive = median_ms(naive1, queries)
    console.kv(
        "hops=1",
        f"{ms_one:.2f} ms vs {ms_naive:.2f} ms naive ({ms_naive / ms_one:.2f}x)",
    )

    console.section("3. two-hop co-author context (AUTHORED_BY only)")
    ms_one = median_ms(ctx2.invoke, queries)
    ms_naive = median_ms(naive2, queries)
    console.kv(
        "hops=2 filtered",
        f"{ms_one:.2f} ms vs {ms_naive:.2f} ms naive ({ms_naive / ms_one:.2f}x)",
    )

    console.section("4. plan shape (filtered two-hop statement)")
    explain_filtered(store)

    console.section("5. async concurrency (16 at once, filtered two-hop)")
    rates = async_throughput(ctx2, queries)
    console.kv("serial", f"{rates['serial_per_s']:.0f} retrievals/s")
    console.kv(
        "16-way",
        f"{rates['concurrent_per_s']:.0f} retrievals/s "
        f"({rates['concurrent_per_s'] / rates['serial_per_s']:.2f}x)",
    )

    # Last on purpose: the unfiltered walk reads a large slice of the corpus
    # and evicts the buffer cache every arm above depends on.
    console.section("6. two hops UNFILTERED through the Category/Year hubs")
    console.kv("hops=2 unfiltered", hub_arm(store, vectors))

    print("loadavg after:", open("/proc/loadavg").read().strip())
    store.close()
    engine.close()


if __name__ == "__main__":
    main()
