"""arXiv GraphRAG — ingest.

Builds, in ONE AgensGraph graph, a knowledge graph of papers + authors +
categories + years AND a pgvector (HNSW) index over the same Paper nodes:

    (:Paper {id,title,abstract,year})-[:AUTHORED_BY]->(:Author {name})
    (:Paper)-[:IN_CATEGORY]->(:Category {name})
    (:Paper)-[:UPDATED_IN]->(:Year {year})

Data: real arXiv metadata streamed from Hugging Face (no full download).
Scale with ARXIV_LIMIT (default 50000). Everything runs through one shared
AgensEngine connection pool with batched Cypher UNWIND ingest.

    cd langchain
    .venv/bin/python examples/demos/01_arxiv_graphrag/prepare.py
    ARXIV_LIMIT=2000 .venv/bin/python examples/demos/01_arxiv_graphrag/prepare.py   # quick
    ARXIV_RESET=1    .venv/bin/python examples/demos/01_arxiv_graphrag/prepare.py   # rebuild
"""

from __future__ import annotations

import os
import pathlib
import sys
import time
from typing import Any, Dict, Iterable, List, Optional

# Make the demos root importable (dir names like "01_..." aren't valid packages).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import psycopg
from psycopg.types.json import Jsonb

from langchain_agensgraph import AgensgraphVector

from _common import agens, config, console
from _common.datautil import batched, env_int, stream_hf

GRAPH = "arxiv"
DATASET = "UniverseTBD/arxiv-abstracts-large"
MAX_AUTHORS = 20  # cap pathological author lists (some physics papers have 100s)


def _norm(rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Normalize a raw HF record into {id,title,abstract,year,authors,categories}."""
    pid = (rec.get("id") or "").strip()
    title = " ".join((rec.get("title") or "").split())
    abstract = " ".join((rec.get("abstract") or "").split())
    if not pid or not title or not abstract:
        return None

    upd = rec.get("update_date")
    year = getattr(upd, "year", None)
    if year is None:  # some dumps carry a string
        try:
            year = int(str(upd)[:4])
        except (ValueError, TypeError):
            year = 0

    authors: List[str] = []
    for parts in (rec.get("authors_parsed") or [])[:MAX_AUTHORS]:
        name = " ".join(p for p in (parts or []) if p).strip()
        if name:
            authors.append(name)

    categories = [c for c in (rec.get("categories") or "").split() if c]

    return {
        "id": pid,
        "title": title,
        "abstract": abstract,
        "year": int(year),
        "authors": authors,
        "categories": categories,
    }


def _reset(conf: Dict[str, Any]) -> None:
    """Drop the graph (outside the pool) so a rebuild starts clean."""
    with psycopg.connect(**conf, autocommit=True) as conn:
        conn.execute('DROP GRAPH IF EXISTS "%s" CASCADE' % GRAPH)
    print(f"[reset] dropped graph {GRAPH!r}")


# ── batched ingest ──────────────────────────────────────────────────────────

PAPER_Q = """
UNWIND %(rows)s AS row
MERGE (p:"Paper" {id: row.id})
  SET p.title = row.title, p.abstract = row.abstract, p.year = row.year
MERGE (y:"Year" {year: row.year})
MERGE (p)-[:"UPDATED_IN"]->(y)
"""

AUTHOR_Q = """
UNWIND %(rows)s AS row
MERGE (a:"Author" {name: row.name})
MERGE (p:"Paper" {id: row.pid})
MERGE (p)-[:"AUTHORED_BY"]->(a)
"""

CATEGORY_Q = """
UNWIND %(rows)s AS row
MERGE (c:"Category" {name: row.name})
MERGE (p:"Paper" {id: row.pid})
MERGE (p)-[:"IN_CATEGORY"]->(c)
"""


def _ensure_schema(graph) -> None:
    """Labels + property indexes so every MERGE is an index lookup, not a seq scan.

    Without these the MERGE-by-name on Author/Category (and MERGE-by-id on Paper)
    would sequentially scan the label on every row → O(N^2) ingest.
    """
    for vlabel in ("Paper", "Author", "Category", "Year"):
        graph.query(f'CREATE VLABEL IF NOT EXISTS "{vlabel}"')
    for elabel in ("AUTHORED_BY", "IN_CATEGORY", "UPDATED_IN"):
        graph.query(f'CREATE ELABEL IF NOT EXISTS "{elabel}"')
    graph.query('CREATE PROPERTY INDEX IF NOT EXISTS paper_id_idx ON "Paper" (id)')
    graph.query('CREATE PROPERTY INDEX IF NOT EXISTS author_name_idx ON "Author" (name)')
    graph.query('CREATE PROPERTY INDEX IF NOT EXISTS category_name_idx ON "Category" (name)')
    graph.query('CREATE PROPERTY INDEX IF NOT EXISTS year_year_idx ON "Year" (year)')


def _ingest(graph, records: Iterable[Dict[str, Any]], batch_size: int) -> Dict[str, int]:
    """Batched UNWIND ingest. Reports DB-only throughput separately from the
    HF streaming latency (which is network-bound and dominates wall-clock)."""
    n_papers = n_authored = n_incat = 0
    db_seconds = 0.0
    wall_start = time.perf_counter()
    for chunk in batched(records, batch_size):
        t0 = time.perf_counter()
        graph.query(PAPER_Q, {"rows": Jsonb(chunk)})
        author_rows = [{"pid": r["id"], "name": a} for r in chunk for a in r["authors"]]
        if author_rows:
            graph.query(AUTHOR_Q, {"rows": Jsonb(author_rows)})
        cat_rows = [{"pid": r["id"], "name": c} for r in chunk for c in r["categories"]]
        if cat_rows:
            graph.query(CATEGORY_Q, {"rows": Jsonb(cat_rows)})
        db_seconds += time.perf_counter() - t0
        n_papers += len(chunk)
        n_authored += len(author_rows)
        n_incat += len(cat_rows)
        if n_papers % (batch_size * 10) == 0:
            print(f"    ... {n_papers:,} papers")

    wall = time.perf_counter() - wall_start
    edges = n_authored + n_incat + n_papers  # + UPDATED_IN (one per paper)
    print(f"  DB ingest:   {n_papers:,} papers + {edges:,} edges in {db_seconds:.2f}s "
          f"({n_papers / db_seconds:,.0f} papers/s, {edges / db_seconds:,.0f} edges/s)")
    print(f"  wall (incl. HF streaming): {wall:.1f}s")
    return {"papers": n_papers, "authored_by": n_authored, "in_category": n_incat}


def main() -> None:
    limit = env_int("ARXIV_LIMIT", 50000)
    batch_size = env_int("ARXIV_BATCH", 1000)
    config.require_openai_key()

    console.section(f"arXiv GraphRAG — ingest  (ARXIV_LIMIT={limit:,})")

    if os.getenv("ARXIV_RESET"):
        _reset(config.conf())

    graph = agens.make_graph(GRAPH, create=True)
    try:
        console.sub("schema (labels + property indexes)")
        _ensure_schema(graph)

        console.sub("streaming + batched UNWIND ingest")
        records = (n for n in (_norm(r) for r in stream_hf(DATASET, limit=limit)) if n)
        counts = _ingest(graph, records, batch_size)
        console.table(list(counts.items()), headers=["edge/node", "count"])

        # Vector index over the SAME Paper nodes: embed title+abstract in place,
        # build the HNSW (cosine) index. Reuses the shared engine + graph.
        console.sub("embedding Paper abstracts → pgvector HNSW (from_existing_graph)")
        with console.timer("embed + index") as t:
            from _common.models import get_embeddings

            AgensgraphVector.from_existing_graph(
                embedding=get_embeddings(),
                node_label="Paper",
                embedding_node_property="embedding",
                text_node_properties=["title", "abstract"],
                index_name="paper_vec",
                graph_name=GRAPH,
                engine=agens.get_engine(),
            )
        print("  " + t.rate(counts["papers"], "papers embedded"))

        console.sub("done")
        total_nodes = graph.query("MATCH (n) RETURN count(n) AS c")[0]["c"]
        total_edges = graph.query("MATCH ()-[r]->() RETURN count(r) AS c")[0]["c"]
        console.table(
            [["nodes", f"{total_nodes:,}"], ["edges", f"{total_edges:,}"]],
            headers=["total", "count"],
        )
        print("\nNext:  .venv/bin/python examples/demos/01_arxiv_graphrag/query.py")
    finally:
        agens.close()


if __name__ == "__main__":
    main()
