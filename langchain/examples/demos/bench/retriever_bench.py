"""Measured answers to the retriever module's performance claims.

Five questions, each with a number:

  1. Does AgensVectorRetriever cost anything over ``as_retriever()``?
  2. What does the one-statement graph context buy over fetching each seed's
     neighbourhood with its own query?
  3. Does the composed statement actually use the vector index and the
     variable-length-edge machinery, or did the planner fall to a Seq Scan?
  4. How does the awaiting path scale when sixteen retrievals run at once?
  5. Is the binary protocol worth requesting for context rows?

Medians of ``REPS`` warm repetitions over a deterministic synthetic graph
(``N_CHUNKS`` vertices, ~4 edges each, seeded embeddings), so two runs argue
about the same workload. Point ``AGENSGRAPH_URL`` at the server to measure;
without it the demos' configuration is used.

    cd langchain
    .venv/bin/python examples/demos/bench/retriever_bench.py
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
from langchain_core.embeddings import Embeddings
from psycopg.types.json import Jsonb

from _common import config, console
from langchain_agensgraph.engine import AgensEngine
from langchain_agensgraph.retrievers import (
    AgensGraphContextRetriever,
    AgensVectorRetriever,
)
from langchain_agensgraph.retrievers.graph_context import build_expansion_query
from langchain_agensgraph.vectorstores.agensgraph_vector import AgensgraphVector

GRAPH = "retriever_bench"
N_CHUNKS = 2000
DIM = 64
REPS = 15
WARMUP = 3
K = 8


def _url() -> str:
    return os.getenv("AGENSGRAPH_URL") or config.url()


def _vectors() -> List[List[float]]:
    rng = random.Random(42)
    return [[rng.uniform(-1, 1) for _ in range(DIM)] for _ in range(N_CHUNKS)]


TEXTS = [f"chunk {i}" for i in range(N_CHUNKS)]
VECTORS = _vectors()


class SeededEmbeddings(Embeddings):
    """Every ``chunk i`` text embeds to the same seeded vector on every run."""

    def _one(self, text: str) -> List[float]:
        if text.startswith("chunk "):
            return VECTORS[int(text.split()[1])]
        return [0.0] * DIM

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._one(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._one(text)


def build_dataset(engine: AgensEngine) -> AgensgraphVector:
    from hashlib import md5

    store = AgensgraphVector.from_embeddings(
        text_embeddings=list(zip(TEXTS, VECTORS)),
        embedding=SeededEmbeddings(),
        pre_delete_collection=True,
        graph_name=GRAPH,
        url=_url(),
        engine=engine,
    )
    store.query("CREATE ELABEL IF NOT EXISTS relates")
    ids = [md5(t.encode()).hexdigest() for t in TEXTS]
    pairs = []
    for i in range(N_CHUNKS):
        for step in (1, 7, 42, 400):
            pairs.append([ids[i], ids[(i + step) % N_CHUNKS]])
    for at in range(0, len(pairs), 2000):
        store.query(
            "UNWIND %(pairs)s AS p "
            'MATCH (a:"Chunk"), (b:"Chunk") '
            "WHERE a.__id__ = p[0] AND b.__id__ = p[1] "
            "CREATE (a)-[:relates]->(b)",
            params={"pairs": Jsonb(pairs[at : at + 2000])},
        )
    store.query("ANALYZE")
    return store


QUERIES = [f"chunk {i}" for i in random.Random(7).sample(range(N_CHUNKS), REPS)]


def median_ms(fn: Callable[[str], Any]) -> float:
    for q in QUERIES[:WARMUP]:
        fn(q)
    times = []
    for q in QUERIES:
        t0 = time.perf_counter()
        fn(q)
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


def naive_two_step(store: AgensgraphVector, hops: int) -> Callable[[str], Any]:
    """The client-side shape: the seed search, then one query per seed."""
    per_seed = (
        'MATCH (seed:"Chunk") WHERE seed.__id__ = %(sid)s '
        + f"OPTIONAL MATCH (seed)-[rels*1..{hops}]-(peer) "
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
            store.query(per_seed, params={"sid": doc.id, "mcn": 20, "mcr": 20})

    return run


def explain_context_statement(store: AgensgraphVector) -> bool:
    """Plan-shape proof, with the seqscan escape hatch closed."""
    statement, params, _ = store._build_search(
        VECTORS[0], K, retrieval_query=build_expansion_query(hops=2, hybrid=False)
    )
    rendered = statement.as_string()
    params = dict(
        params,
        embedding="[" + ",".join(map(str, VECTORS[0])) + "]",
        max_context_nodes=20,
        max_context_rels=20,
    )
    with psycopg.connect(_url()) as conn:
        cur = conn.cursor()
        cur.execute(f"SET graph_path = {GRAPH}")
        cur.execute("SET enable_seqscan = off")
        cur.execute("EXPLAIN (COSTS OFF) " + rendered, params)
        plan = "\n".join(r[0] for r in cur.fetchall())
    checks = {
        "vector index drives the seeds": "Index Scan" in plan and "vector" in plan,
        "expansion is the VLE machinery": "VLE" in plan,
        "no Seq Scan anywhere": "Seq Scan" not in plan,
    }
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if not all(checks.values()):
        print("    plan:\n      " + plan.replace("\n", "\n      "))
    return all(checks.values())


def async_throughput(retriever: AgensGraphContextRetriever) -> Dict[str, float]:
    queries = [QUERIES[i % len(QUERIES)] for i in range(64)]

    async def serial() -> float:
        t0 = time.perf_counter()
        for q in queries:
            await retriever.ainvoke(q)
        return time.perf_counter() - t0

    async def concurrent() -> float:
        t0 = time.perf_counter()
        for at in range(0, len(queries), 16):
            await asyncio.gather(
                *(retriever.ainvoke(q) for q in queries[at : at + 16])
            )
        return time.perf_counter() - t0

    async def both() -> Dict[str, float]:
        await retriever.ainvoke(queries[0])
        serial_s = await serial()
        concurrent_s = await concurrent()
        return {
            "serial_per_s": len(queries) / serial_s,
            "concurrent_per_s": len(queries) / concurrent_s,
        }

    return asyncio.run(both())


def binary_vs_text(store: AgensgraphVector) -> Dict[str, float]:
    statement, params, _ = store._build_search(
        VECTORS[0], K, retrieval_query=build_expansion_query(hops=2, hybrid=False)
    )
    params = dict(params, max_context_nodes=20, max_context_rels=20)
    out = {}
    for name, binary_ in (("text", False), ("binary", True)):
        for _ in range(WARMUP):
            store.query(statement, params=params, binary_=binary_)
        times = []
        for _ in range(REPS):
            t0 = time.perf_counter()
            store.query(statement, params=params, binary_=binary_)
            times.append((time.perf_counter() - t0) * 1000)
        out[name] = statistics.median(times)
    return out


def main() -> None:
    print("loadavg before:", open("/proc/loadavg").read().strip())
    engine = AgensEngine(_url(), max_size=16)
    console.section("dataset")
    with console.timer(f"{N_CHUNKS} chunks + {N_CHUNKS * 4} edges + HNSW"):
        store = build_dataset(engine)

    console.section("1. seed-only retriever vs as_retriever")
    raw = store.as_retriever(search_kwargs={"k": K})
    plain = AgensVectorRetriever(store=store, k=K)
    ms_raw = median_ms(raw.invoke)
    ms_ours = median_ms(plain.invoke)
    console.kv("as_retriever", f"{ms_raw:.2f} ms")
    console.kv("AgensVectorRetriever", f"{ms_ours:.2f} ms  ({ms_ours / ms_raw:.2f}x)")

    console.section("2. one-statement context vs one query per seed")
    for hops in (1, 2):
        ctx = AgensGraphContextRetriever(store=store, k=K, expand_by_hops=hops)
        ms_one = median_ms(ctx.invoke)
        ms_naive = median_ms(naive_two_step(store, hops))
        console.kv(
            f"hops={hops} one statement",
            f"{ms_one:.2f} ms vs {ms_naive:.2f} ms naive "
            f"({ms_naive / ms_one:.2f}x)",
        )

    console.section("3. plan shape of the composed statement")
    explain_context_statement(store)

    console.section("4. async concurrency (16 at once, hops=1)")
    ctx1 = AgensGraphContextRetriever(store=store, k=K, expand_by_hops=1)
    rates = async_throughput(ctx1)
    console.kv("serial", f"{rates['serial_per_s']:.0f} retrievals/s")
    console.kv(
        "16-way",
        f"{rates['concurrent_per_s']:.0f} retrievals/s "
        f"({rates['concurrent_per_s'] / rates['serial_per_s']:.2f}x)",
    )

    console.section("5. binary vs text decode of context rows")
    decode = binary_vs_text(store)
    console.kv("text", f"{decode['text']:.2f} ms")
    ratio = decode["text"] / decode["binary"]
    console.kv("binary", f"{decode['binary']:.2f} ms ({ratio:.2f}x)")

    print("loadavg after:", open("/proc/loadavg").read().strip())
    store.query("MATCH (n) DETACH DELETE n")
    store.close()
    engine.close()


if __name__ == "__main__":
    main()
