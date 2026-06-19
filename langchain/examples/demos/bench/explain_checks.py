"""EXPLAIN proofs that the demos' hot paths use the right indexes.

Run AFTER 01_arxiv_graphrag/prepare.py has populated the ``arxiv`` graph. Each
check forces ``enable_seqscan = off`` and asserts the expected index appears in
the plan with no ``Seq Scan`` — i.e. the index is genuinely usable, not just
"chosen because the table is tiny".

    cd langchain
    .venv/bin/python examples/demos/bench/explain_checks.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import psycopg

from _common import config, console

GRAPH = "arxiv"


def _explain_cypher(cur, cypher: str, params: dict) -> str:
    cur.execute("SET LOCAL enable_seqscan = off")
    cur.execute("EXPLAIN " + cypher, params)
    plan = "\n".join(r[0] for r in cur.fetchall())
    cur.connection.rollback()
    return plan


def _check(name: str, plan: str, must_have: str) -> bool:
    ok = must_have in plan and "Seq Scan" not in plan
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: uses {must_have!r}, no Seq Scan")
    if not ok:
        print("    plan:\n      " + plan.replace("\n", "\n      "))
    return ok


def main() -> None:
    import json

    conf = config.conf()
    conn = psycopg.connect(**conf)
    cur = conn.cursor()
    cur.execute(f"SET graph_path = {GRAPH}")
    conn.commit()
    ok = True

    console.section("EXPLAIN — graph hot paths (Cypher property indexes)")
    ok &= _check(
        "Paper lookup by id (MERGE/MATCH key)",
        _explain_cypher(cur, 'MATCH (p:"Paper" {id: %(v)s}) RETURN p', {"v": json.dumps("0704.0001")}),
        "paper_id_idx",
    )
    ok &= _check(
        "Author lookup by name (ingest MERGE key)",
        _explain_cypher(cur, 'MATCH (a:"Author" {name: %(v)s}) RETURN a', {"v": json.dumps("Berger E. L.")}),
        "author_name_idx",
    )
    ok &= _check(
        "Category lookup by name",
        _explain_cypher(cur, 'MATCH (c:"Category" {name: %(v)s}) RETURN c', {"v": json.dumps("hep-ph")}),
        "category_name_idx",
    )

    console.section("Vector index — HNSW over Paper embeddings")
    cur.execute(
        "SELECT indexname, indexdef FROM pg_indexes "
        "WHERE schemaname = %s AND indexdef ILIKE '%%hnsw%%'",
        (GRAPH,),
    )
    rows = cur.fetchall()
    if rows:
        for name, ddl in rows:
            print(f"  [PASS] HNSW index present: {name}")
            print(f"         {ddl}")
    else:
        ok = False
        print("  [FAIL] no HNSW index found on graph", GRAPH)

    cur.close()
    conn.close()
    print(f"\n{'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
