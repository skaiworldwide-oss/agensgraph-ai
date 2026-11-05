"""EXPLAIN proofs that the demos' hot paths use the right indexes.

Run AFTER 01_arxiv_graphrag/prepare.py has populated the ``arxiv`` graph. Each check forces
``enable_seqscan = off`` and asserts the expected index appears in the plan with no ``Seq Scan``
— i.e. the index is genuinely usable, not just "chosen because the table is tiny".

    cd langchain
    .venv/bin/python examples/demos/bench/explain_checks.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import psycopg
from agensgraph import Jsonb
from langchain_core.embeddings import DeterministicFakeEmbedding

from _common import agens, config, console
from langchain_agensgraph import AgensgraphVector, log_queries

GRAPH = "arxiv"
STORE_SCHEMA = f"{GRAPH}_store"


def _explain(cur, statement: str, params) -> str:
    cur.execute("SET LOCAL enable_seqscan = off")
    cur.execute("EXPLAIN " + statement, params)
    plan = "\n".join(r[0] for r in cur.fetchall())
    cur.connection.rollback()
    return plan


def _check(name: str, plan: str, must_have: str) -> bool:
    ok = must_have in plan and "Seq Scan" not in plan
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: uses {must_have!r}, no Seq Scan")
    if not ok:
        print("    plan:\n      " + plan.replace("\n", "\n      "))
    return ok


def _key(cur, label: str) -> object:
    cur.execute(f'MATCH (n:"{label}") RETURN n.id LIMIT 1')
    return cur.fetchone()[0]


def search_statement() -> tuple[str, str]:
    """The statement a vector search sends, as the driver reports it, and a query vector."""
    # The store reads its dimension from the vector table; searching by vector embeds nothing.
    store = AgensgraphVector.from_existing_index(
        DeterministicFakeEmbedding(size=1),
        engine=agens.get_engine(),
        graph_name=GRAPH,
        label="Paper",
    )
    vector = [0.1] * int(store.dims)
    with log_queries() as sent:
        store.similarity_search_by_vector(vector, k=5)
    statement = next(r.statement for r in sent if "ORDER BY distance" in r.statement)
    return statement, "[" + ",".join(map(str, vector)) + "]"


def main() -> None:
    conn = psycopg.connect(config.url())
    cur = conn.cursor()
    cur.execute(f'SET graph_path = "{GRAPH}"')
    conn.commit()
    ok = True

    # add_graph_documents gives every label it writes a unique property index on id, named
    # "<label>__id", which each MERGE and MATCH by id is a lookup in.
    console.section("EXPLAIN — graph hot paths (Cypher property indexes)")
    for label in ("Paper", "Author", "Category"):
        key = _key(cur, label)
        ok &= _check(
            f"{label} lookup by id (MERGE/MATCH key)",
            _explain(cur, f'MATCH (n:"{label}" {{id: %s}}) RETURN n', (Jsonb(key),)),
            f"{label}__id",
        )

    console.section("Vector index — HNSW over the Paper embeddings")
    cur.execute(
        "SELECT indexname, indexdef FROM pg_indexes "
        "WHERE schemaname = %s AND indexdef ILIKE '%%hnsw%%'",
        (STORE_SCHEMA,),
    )
    rows = cur.fetchall()
    conn.commit()
    if rows:
        for name, ddl in rows:
            print(f"  [PASS] HNSW index present: {name}")
            print(f"         {ddl}")
        statement, vector = search_statement()
        ok &= _check("similarity search", _explain(cur, statement, (vector, 5)), rows[0][0])
    else:
        ok = False
        print(f"  [FAIL] no HNSW index found in {STORE_SCHEMA}")

    cur.close()
    conn.close()
    agens.close()
    print(f"\n{'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
