"""Measured answers to the retriever module's performance claims.

Four questions, each with a number:

  1. Does AgensVectorRetriever cost anything over ``as_retriever()``?
  2. What does the graph context retriever's single walk from every seed buy over fetching each
     seed's neighbourhood with its own query?
  3. Do the search and the walk use indexes, or did the planner fall to a Seq Scan?
  4. How does the awaiting path scale when sixteen retrievals run at once?

Medians of ``REPS`` warm repetitions over a deterministic synthetic graph (``N_CHUNKS`` vertices,
~4 edges each, seeded embeddings), so two runs argue about the same workload. Point
``AGENSGRAPH_URL`` at the server to measure; the graph ``retriever_bench`` is made and dropped.

    cd langchain
    .venv/bin/python examples/demos/bench/retriever_bench.py
    N_CHUNKS=200 .venv/bin/python examples/demos/bench/retriever_bench.py   # quick
"""

from __future__ import annotations

import asyncio
import os
import pathlib
import random
import statistics
import sys
import time
from typing import Any, Callable, Dict, List

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import psycopg
from agensgraph import Jsonb
from langchain_core.embeddings import Embeddings

from _common import config, console
from _common.datautil import env_int
from langchain_agensgraph import AgensEngine, AgensGraph, AgensgraphVector, log_queries
from langchain_agensgraph.retrievers import AgensGraphContextRetriever, AgensVectorRetriever

GRAPH = "retriever_bench"
N_CHUNKS = env_int("N_CHUNKS", 2000)
DIM = 64
REPS = 15
WARMUP = 3
K = 8


def _vectors() -> List[List[float]]:
    rng = random.Random(42)
    return [[rng.uniform(-1, 1) for _ in range(DIM)] for _ in range(N_CHUNKS)]


TEXTS = [f"chunk {i}" for i in range(N_CHUNKS)]
VECTORS = _vectors()
QUERIES = [f"chunk {i}" for i in random.Random(7).sample(range(N_CHUNKS), REPS)]


class SeededEmbeddings(Embeddings):
    """Every ``chunk i`` text embeds to the same seeded vector on every run."""

    def _one(self, text: str) -> List[float]:
        return VECTORS[int(text.split()[1])]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._one(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._one(text)


def open_store(engine: AgensEngine) -> AgensgraphVector:
    return AgensgraphVector(SeededEmbeddings(), engine=engine, graph_name=GRAPH)


def build_dataset(engine: AgensEngine, graph: AgensGraph) -> AgensgraphVector:
    graph.drop_graph()
    store = AgensgraphVector(
        SeededEmbeddings(), engine=engine, graph_name=GRAPH, create_graph=True
    )
    ids = [f"c{i}" for i in range(N_CHUNKS)]
    store.add_embeddings(TEXTS, VECTORS, ids=ids)
    graph.query("CREATE ELABEL IF NOT EXISTS relates")
    pairs = [
        [ids[i], ids[(i + step) % N_CHUNKS]] for i in range(N_CHUNKS) for step in (1, 7, 42, 400)
    ]
    for at in range(0, len(pairs), 2000):
        graph.query(
            "UNWIND %(pairs)s AS p "
            'MATCH (a:"Chunk"), (b:"Chunk") '
            "WHERE a.__id__ = p[0] AND b.__id__ = p[1] "
            "CREATE (a)-[:relates]->(b)",
            {"pairs": Jsonb(pairs[at : at + 2000])},
        )
    graph.query("ANALYZE")
    return store


def median_ms(fn: Callable[[str], Any]) -> float:
    for q in QUERIES[:WARMUP]:
        fn(q)
    times = []
    for q in QUERIES:
        t0 = time.perf_counter()
        fn(q)
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


def naive_two_step(store: AgensgraphVector, graph: AgensGraph, hops: int) -> Callable[[str], Any]:
    """The client-side shape: the seed search, then one query per seed."""
    per_seed = (
        'MATCH (seed:"Chunk") WHERE seed.__id__ = %(sid)s '
        f"MATCH (seed)-[rels*1..{hops}]-(peer) "
        "RETURN DISTINCT id(peer) AS id, label(peer) AS label, properties(peer) AS properties "
        "LIMIT %(cap)s"
    )

    def run(query: str) -> None:
        for doc in store.similarity_search(query, k=K):
            graph.query(per_seed, {"sid": doc.id, "cap": 20})

    return run


def statements(retriever: AgensGraphContextRetriever, query: str) -> Dict[str, str]:
    """The search and the walk a retrieval sends, as the driver reports them."""
    with log_queries() as sent:
        retriever.invoke(query)
    texts = [record.statement for record in sent]
    return {
        "search": next(t for t in texts if "ORDER BY distance" in t),
        "walk": next(t for t in texts if t.startswith("WITH RECURSIVE walk")),
    }


def explain(statement: str, params: tuple) -> str:
    with psycopg.connect(config.url()) as conn:
        conn.execute(f'SET graph_path = "{GRAPH}"')
        conn.execute("SET enable_seqscan = off")
        rows = conn.execute("EXPLAIN (COSTS OFF) " + statement, params).fetchall()
    return "\n".join(r[0] for r in rows)


def plan_shape(retriever: AgensGraphContextRetriever, graph: AgensGraph) -> bool:
    """Plan-shape proof, with the seqscan escape hatch closed."""
    sent = statements(retriever, QUERIES[0])
    vector = "[" + ",".join(map(str, VECTORS[0])) + "]"
    search_plan = explain(sent["search"], (vector, K))
    found = graph.query("MATCH (c:\"Chunk\") WHERE c.__id__ IN ['c0', 'c1'] RETURN id(c) AS id")
    seeds = [str(r["id"]) for r in found]
    walk_plan = explain(sent["walk"], (seeds, 20, 2))
    checks = {
        "the HNSW index drives the seed search": "hnsw" in search_plan.lower()
        or "Index Scan" in search_plan,
        "no Seq Scan in the search": "Seq Scan" not in search_plan,
        "the walk reads edges by index": "Index" in walk_plan,
        "no Seq Scan in the walk": "Seq Scan" not in walk_plan,
    }
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if not all(checks.values()):
        print("    search plan:\n      " + search_plan.replace("\n", "\n      "))
        print("    walk plan:\n      " + walk_plan.replace("\n", "\n      "))
    return all(checks.values())


def async_throughput() -> Dict[str, float]:
    """Sixteen retrievals at once, on an engine whose awaiting pools close with the loop."""
    queries = [QUERIES[i % len(QUERIES)] for i in range(64)]

    async def both() -> Dict[str, float]:
        async with AgensEngine(config.url(), max_size=16) as engine:
            retriever = AgensGraphContextRetriever(vectorstore=open_store(engine), k=K, hops=1)
            await retriever.ainvoke(queries[0])
            t0 = time.perf_counter()
            for q in queries:
                await retriever.ainvoke(q)
            serial_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            for at in range(0, len(queries), 16):
                await asyncio.gather(*(retriever.ainvoke(q) for q in queries[at : at + 16]))
            concurrent_s = time.perf_counter() - t0
        return {
            "serial_per_s": len(queries) / serial_s,
            "concurrent_per_s": len(queries) / concurrent_s,
        }

    return asyncio.run(both())


def main() -> None:
    print("loadavg before:", os.getloadavg())
    with AgensEngine(config.url(), max_size=16) as engine:
        graph = AgensGraph(engine, graph_name=GRAPH, create_graph=True)
        try:
            console.section("dataset")
            with console.timer(f"{N_CHUNKS} chunks + {N_CHUNKS * 4} edges + HNSW"):
                store = build_dataset(engine, graph)

            console.section("1. seed-only retriever vs as_retriever")
            raw = store.as_retriever(search_kwargs={"k": K})
            plain = AgensVectorRetriever(vectorstore=store, k=K)
            ms_raw = median_ms(raw.invoke)
            ms_ours = median_ms(plain.invoke)
            console.kv("as_retriever", f"{ms_raw:.2f} ms")
            console.kv("AgensVectorRetriever", f"{ms_ours:.2f} ms  ({ms_ours / ms_raw:.2f}x)")

            console.section("2. one walk from every seed vs one query per seed")
            for hops in (1, 2):
                ctx = AgensGraphContextRetriever(vectorstore=store, k=K, hops=hops)
                ms_one = median_ms(ctx.invoke)
                ms_naive = median_ms(naive_two_step(store, graph, hops))
                console.kv(
                    f"hops={hops} one walk",
                    f"{ms_one:.2f} ms vs {ms_naive:.2f} ms naive ({ms_naive / ms_one:.2f}x)",
                )

            console.section("3. plan shape of the search and the walk")
            plan_shape(AgensGraphContextRetriever(vectorstore=store, k=K, hops=2), graph)
        except BaseException:
            graph.drop_graph()
            raise

    console.section("4. async concurrency (16 at once, hops=1)")
    rates = async_throughput()
    console.kv("serial", f"{rates['serial_per_s']:.0f} retrievals/s")
    console.kv(
        "16-way",
        f"{rates['concurrent_per_s']:.0f} retrievals/s "
        f"({rates['concurrent_per_s'] / rates['serial_per_s']:.2f}x)",
    )

    print("loadavg after:", os.getloadavg())
    with AgensEngine(config.url()) as engine:
        AgensGraph(engine, graph_name=GRAPH).drop_graph()


if __name__ == "__main__":
    main()
