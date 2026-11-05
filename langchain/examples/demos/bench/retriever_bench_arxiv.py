"""The retriever numbers again, on 138k real vertices instead of a synthetic ring.

Runs against the ``arxiv`` graph demo 01 builds (50k papers, 88k authors, 275k edges). Query
vectors are read from the stored embeddings, so nothing here calls an embedding API -- every
millisecond is the database.

This graph has what the synthetic one deliberately lacks: real hubs. 147 Category vertices carry
75k edges and 17 Year vertices carry 50k, so an UNFILTERED two-hop walk passes through them to a
large slice of the corpus. That arm runs under a statement timeout and is reported as the
cautionary number; the ``relationship_type`` filter (co-author walk over AUTHORED_BY alone) is the
fast shape the retriever documents for graphs like this.

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
from langchain_agensgraph import AgensEngine, AgensGraph, AgensgraphVector, log_queries
from langchain_agensgraph.retrievers import AgensGraphContextRetriever, AgensVectorRetriever

GRAPH = "arxiv"
REPS = 15
WARMUP = 3
K = 8
HUB_REPS = 3
HUB_TIMEOUT = 15.0


class LookupEmbeddings(Embeddings):
    """Query texts are keys into vectors already read from the store."""

    def __init__(self) -> None:
        self.mapping: Dict[str, List[float]] = {}

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.mapping[t] for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self.mapping[text]


def open_store(engine: AgensEngine, lookup: LookupEmbeddings) -> AgensgraphVector:
    return AgensgraphVector.from_existing_index(
        lookup, engine=engine, graph_name=GRAPH, label="Paper"
    )


def sample_queries(store: AgensgraphVector, lookup: LookupEmbeddings) -> List[str]:
    """Stored embeddings as query vectors: those nearest one fixed direction."""
    lookup.mapping["probe"] = [1.0] + [0.0] * (int(store.dims) - 1)
    hits = store.search_with_embeddings("probe", k=REPS + WARMUP)
    for i, (_, _, embedding) in enumerate(hits):
        lookup.mapping[f"q{i}"] = embedding
    return [f"q{i}" for i in range(len(hits))]


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
    store: AgensgraphVector, graph: AgensGraph, hops: int, rel_type: str | None
) -> Callable[[str], Any]:
    """The client-side shape: the seed search, then one query per seed."""
    rel = f':"{rel_type}"' if rel_type else ""
    per_seed = (
        'MATCH (seed:"Paper") WHERE seed.id = %(sid)s '
        f"MATCH (seed)-[rels{rel}*1..{hops}]-(peer) "
        "RETURN DISTINCT id(peer) AS id, label(peer) AS label, properties(peer) AS properties "
        "LIMIT %(cap)s"
    )

    def run(query: str) -> None:
        for doc in store.similarity_search(query, k=K):
            graph.query(per_seed, {"sid": doc.metadata["id"], "cap": 20})

    return run


def hub_arm(lookup: LookupEmbeddings, queries: List[str]) -> str:
    """Unfiltered two hops through the Category/Year hubs, under a statement timeout."""
    times = []
    with AgensEngine(config.url(), statement_timeout=HUB_TIMEOUT) as engine:
        hub = AgensGraphContextRetriever(vectorstore=open_store(engine, lookup), k=K, hops=2)
        for query in queries[:HUB_REPS]:
            t0 = time.perf_counter()
            try:
                hub.invoke(query)
            except psycopg.errors.QueryCanceled:
                return f"timed out at {HUB_TIMEOUT:.0f}s (this is the point)"
            times.append((time.perf_counter() - t0) * 1000)
    return f"{statistics.median(times):.0f} ms median of {HUB_REPS}"


def explain_filtered(retriever: AgensGraphContextRetriever, graph: AgensGraph, query: str) -> None:
    """The plan of the filtered walk the retriever sends, with the seqscan escape hatch closed."""
    with log_queries() as sent:
        retriever.invoke(query)
    walk = next(r.statement for r in sent if r.statement.startswith("WITH RECURSIVE walk"))
    seeds = [str(r["id"]) for r in graph.query('MATCH (p:"Paper") RETURN id(p) AS id LIMIT 8')]
    with psycopg.connect(config.url()) as conn:
        conn.execute(f'SET graph_path = "{GRAPH}"')
        conn.execute("SET enable_seqscan = off")
        rows = conn.execute("EXPLAIN (COSTS OFF) " + walk, (seeds, 20, 2)).fetchall()
    plan = "\n".join(r[0] for r in rows)
    checks = {
        "the walk reads only AUTHORED_BY": '"AUTHORED_BY"' in walk and "ag_edge" not in walk,
        "edges are read by index": "Index" in plan,
        "no Seq Scan anywhere": "Seq Scan" not in plan,
    }
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if not all(checks.values()):
        print("    plan:\n      " + plan.replace("\n", "\n      "))


def async_throughput(lookup: LookupEmbeddings, queries: List[str]) -> Dict[str, float]:
    jobs = [queries[i % len(queries)] for i in range(48)]

    async def run() -> Dict[str, float]:
        async with AgensEngine(config.url(), max_size=16) as engine:
            retriever = AgensGraphContextRetriever(
                vectorstore=open_store(engine, lookup),
                k=K,
                hops=2,
                relationship_type="AUTHORED_BY",
            )
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
    print("loadavg before:", os.getloadavg())
    lookup = LookupEmbeddings()
    with AgensEngine(config.url(), max_size=16) as engine:
        graph = AgensGraph(engine, graph_name=GRAPH)
        store = open_store(engine, lookup)
        queries = sample_queries(store, lookup)
        counts = graph.query(
            "MATCH (n) WITH count(*) AS v MATCH ()-[e]->() RETURN v, count(*) AS e"
        )[0]
        console.kv("graph", f"{GRAPH}: {counts['v']:,} vertices, {counts['e']:,} edges")

        plain = AgensVectorRetriever(vectorstore=store, k=K)
        ctx1 = AgensGraphContextRetriever(vectorstore=store, k=K, hops=1)
        ctx2 = AgensGraphContextRetriever(
            vectorstore=store, k=K, hops=2, relationship_type="AUTHORED_BY"
        )
        naive1 = naive_two_step(store, graph, 1, None)
        naive2 = naive_two_step(store, graph, 2, "AUTHORED_BY")

        console.section("warming every arm over every query")
        with console.timer("warmup"):
            for fn in (plain.invoke, ctx1.invoke, naive1, ctx2.invoke, naive2):
                for q in queries:
                    fn(q)

        console.section("1. seed-only retriever (k=8)")
        console.kv("AgensVectorRetriever", f"{median_ms(plain.invoke, queries):.2f} ms")

        console.section("2. one-hop context, one walk vs one query per seed")
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

        console.section("4. plan shape (filtered two-hop walk)")
        explain_filtered(ctx2, graph, queries[0])

    console.section("5. async concurrency (16 at once, filtered two-hop)")
    rates = async_throughput(lookup, queries)
    console.kv("serial", f"{rates['serial_per_s']:.0f} retrievals/s")
    console.kv(
        "16-way",
        f"{rates['concurrent_per_s']:.0f} retrievals/s "
        f"({rates['concurrent_per_s'] / rates['serial_per_s']:.2f}x)",
    )

    # Last on purpose: the unfiltered walk reads a large slice of the corpus
    # and evicts the buffer cache every arm above depends on.
    console.section("6. two hops UNFILTERED through the Category/Year hubs")
    console.kv("hops=2 unfiltered", hub_arm(lookup, queries))

    print("loadavg after:", os.getloadavg())


if __name__ == "__main__":
    main()
