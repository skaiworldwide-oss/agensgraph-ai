"""The phase-0 shape matrix (PLAN.md §9.1): 22 statement forms the 0.4 package relies on.

Each test asserts the behaviour the plan records for the server line it runs on. A failure on a
leg means the plan's assumption is wrong there and needs an alternative before code depends on it.
"""

from __future__ import annotations

import subprocess
import uuid

import numpy as np
import psycopg
import pytest
from psycopg.types.json import Jsonb

from .conftest import APP_ROLE, DB, LLM_ROLE, PARTIAL_ROLE, Server
from .support import (
    copy_in,
    create_vector_table,
    edge_copy,
    gid_text,
    label_info,
    load_chunks,
    public_create_granted,
    qi,
    read_only_is_enforced,
    reserve_ids,
    sqlstate,
    vec_literal,
    vertex_copy,
)

RNG_SEED = 20261030

NO_HASH_OR_MERGE_JOINS = (
    "SELECT set_config('enable_hashjoin', 'off', true), set_config('enable_mergejoin', 'off', true)"
)


def _rng() -> np.random.Generator:
    return np.random.default_rng(RNG_SEED)


def _unit(rows: int, dim: int) -> np.ndarray:
    v = _rng().standard_normal((rows, dim)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _store(graph: str) -> str:
    return qi(graph + "_store")


def _grant_app(conn: psycopg.Connection, graph: str, role: str = APP_ROLE) -> None:
    conn.execute(f"CREATE SCHEMA IF NOT EXISTS {_store(graph)}")
    for schema in (qi(graph), _store(graph)):
        conn.execute(f"GRANT USAGE ON SCHEMA {schema} TO {role}")
        conn.execute(
            f"GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA {schema} TO {role}"
        )
        conn.execute(f"GRANT USAGE ON ALL SEQUENCES IN SCHEMA {schema} TO {role}")


# 1 ------------------------------------------------------------------------------------------------
def test_01_reserved_ids_binary_copy(server: Server, graph: str, admin: psycopg.Connection) -> None:
    """Ids reserved via ag_label.labid + the pg_depend sequence; binary COPY into vertex and edge
    labels, including a renamed label; a duplicate __id__ raises 23505."""
    admin.execute('CREATE VLABEL "Chunk"')
    admin.execute('CREATE VLABEL "Old"')
    admin.execute('CREATE ELABEL "REL"')
    admin.execute('CREATE UNIQUE PROPERTY INDEX "Chunk__id" ON "Chunk" (__id__)')
    admin.execute('ALTER VLABEL "Old" RENAME TO "New"')

    for label in ("Chunk", "New"):
        ids = load_chunks(
            admin, graph, [{"__id__": f"{label}-{i}"} for i in range(50)], label=label
        )
        info = label_info(admin, graph, label)
        assert {i >> 48 for i in ids} == {info.labid}
        got = admin.execute(f"MATCH (n:{qi(label)}) RETURN count(n)").fetchone()[0]
        assert int(got) == 50
        first = admin.execute(
            f"MATCH (n:{qi(label)}) WHERE id(n) = %s::graphid RETURN n.__id__",
            (gid_text(ids[0]),),
        ).fetchone()[0]
        assert first == f"{label}-0"

    chunk_ids = [
        r[0]
        for r in admin.execute(
            "SELECT id::text FROM "
            + label_info(admin, graph, "Chunk").table
            + " ORDER BY id LIMIT 2"
        ).fetchall()
    ]
    rel = label_info(admin, graph, "REL")
    edge_ids = reserve_ids(admin, rel, 1)
    start, end = (int(a) << 48 | int(b) for a, b in (t.split(".") for t in chunk_ids))
    copy_in(
        admin,
        f'COPY {rel.table} (id, start, "end", properties) FROM STDIN (FORMAT BINARY)',
        edge_copy([(edge_ids[0], start, end, {"w": 1})]),
    )
    assert (
        int(admin.execute('MATCH (:"Chunk")-[r:"REL"]->(:"Chunk") RETURN count(r)').fetchone()[0])
        == 1
    )

    info = label_info(admin, graph, "Chunk")
    with pytest.raises(psycopg.errors.UniqueViolation):
        copy_in(
            admin,
            f"COPY {info.table} (id, properties) FROM STDIN (FORMAT BINARY)",
            vertex_copy(reserve_ids(admin, info, 1), [{"__id__": "Chunk-0"}]),
        )


# 2 ------------------------------------------------------------------------------------------------
def _upsert(
    conn: psycopg.Connection, graph: str, rows: list[dict], vectors: list[list[float]]
) -> None:
    keys = [r["__id__"] for r in rows]
    conn.execute(NO_HASH_OR_MERGE_JOINS)
    # A guard after MERGE (MERGE … WITH c, row WHERE … SET) creates nothing on every line, so the
    # upsert is a guarded update of existing vertices plus a single-insert create of missing ones.
    conn.execute(
        'UNWIND %s AS row MATCH (c:"Chunk" {__id__: row.__id__}) '
        "WITH c, row WHERE properties(c) <> row SET c = row",
        (Jsonb(rows),),
    )
    conn.execute(
        'UNWIND %s AS row OPTIONAL MATCH (c:"Chunk" {__id__: row.__id__}) '
        'WITH row, c WHERE c IS NULL CREATE (:"Chunk" =row)',
        (Jsonb(rows),),
    )
    conn.execute(
        f"""
        WITH ids AS (UNWIND %s AS k MATCH (c:"Chunk" {{__id__: k}}) RETURN k AS k, id(c) AS id)
        INSERT INTO {_store(graph)}.vec AS v (id, embedding)
        SELECT ids.id, x.emb FROM ids JOIN unnest(%s::text[], %s::vector[]) AS x(k, emb)
          ON ids.k = to_jsonb(x.k)
        ON CONFLICT (id) DO UPDATE SET embedding = EXCLUDED.embedding
          WHERE v.embedding IS DISTINCT FROM EXCLUDED.embedding
        """,
        (Jsonb(keys), keys, [vec_literal(v) for v in vectors]),
    )


def _xact_writes(conn: psycopg.Connection, graph: str) -> dict[str, tuple[int, int]]:
    """Pending (inserted, updated) counts per table. They accumulate until the backend flushes its
    statistics, so callers compare readings taken inside one transaction."""
    rows = conn.execute(
        "SELECT relname, n_tup_ins, n_tup_upd FROM pg_stat_xact_user_tables "
        "WHERE schemaname IN (%s, %s) AND relname IN ('Chunk', 'vec')",
        (graph, graph + "_store"),
    ).fetchall()
    counts = {"Chunk": (0, 0), "vec": (0, 0)}
    counts.update({r[0]: (r[1], r[2]) for r in rows})
    return counts


def _upsert_writes(
    conn: psycopg.Connection, graph: str, rows: list[dict], vectors: list[list[float]]
) -> dict[str, tuple[int, int]]:
    """Run the upsert in a transaction and return what it wrote per table."""
    conn.execute("BEGIN")
    before = _xact_writes(conn, graph)
    _upsert(conn, graph, rows, vectors)
    after = _xact_writes(conn, graph)
    conn.execute("COMMIT")
    return {t: (after[t][0] - before[t][0], after[t][1] - before[t][1]) for t in after}


def test_02_caller_id_upsert_pipeline(
    server: Server, graph: str, admin: psycopg.Connection
) -> None:
    """The caller-id upsert in one pipeline; a no-op re-add writes nothing to either table."""
    admin.execute('CREATE VLABEL "Chunk"')
    admin.execute('CREATE UNIQUE PROPERTY INDEX "Chunk__id" ON "Chunk" (__id__)')
    create_vector_table(admin, graph, 4)
    rows = [{"__id__": f"k{i}", "text": f"t{i}", "m": i} for i in range(5)]
    vectors = [[float(i), 1.0, 0.0, 0.5] for i in range(5)]

    with admin.pipeline():
        _upsert(admin, graph, rows, vectors)
    assert int(admin.execute('MATCH (c:"Chunk") RETURN count(c)').fetchone()[0]) == 5
    assert admin.execute(f"SELECT count(*) FROM {_store(graph)}.vec").fetchone()[0] == 5

    unchanged = _upsert_writes(admin, graph, rows, vectors)
    assert unchanged == {"Chunk": (0, 0), "vec": (0, 0)}, unchanged

    rows[2] = {**rows[2], "text": "changed"}
    vectors[2] = [9.0, 9.0, 9.0, 9.0]
    rows.append({"__id__": "k5", "text": "t5", "m": 5})
    vectors.append([5.0, 1.0, 0.0, 0.5])
    changed = _upsert_writes(admin, graph, rows, vectors)
    assert changed == {"Chunk": (1, 1), "vec": (1, 1)}, changed
    assert (
        admin.execute("MATCH (c:\"Chunk\" {__id__: 'k2'}) RETURN c.text").fetchone()[0] == "changed"
    )


# 3 ------------------------------------------------------------------------------------------------
KNN = """
SELECT t.id FROM (
  SELECT v.id, v.embedding <=> (SELECT %s::vector
           WHERE set_config('hnsw.ef_search', %s, true) IS NOT NULL
             AND set_config('hnsw.iterative_scan', %s, true) IS NOT NULL) AS d
  FROM {vec} v JOIN {label} c ON c.id = v.id
  WHERE c.properties->>'source' = %s
  ORDER BY d LIMIT 10) t
"""

KNN_UNFILTERED = """
SELECT v.id FROM {vec} v
ORDER BY v.embedding <=>
         (SELECT %s::vector WHERE set_config('hnsw.ef_search', %s, true) IS NOT NULL)
LIMIT 10
"""


def test_03_initplan_settings(server: Server, graph: str, admin: psycopg.Connection) -> None:
    """Per-search pgvector settings set inside the statement apply, prepared and generic, and do
    not leak to the next statement."""
    admin.execute('CREATE VLABEL "Chunk"')
    vec = create_vector_table(admin, graph, 16)
    vectors = _unit(3000, 16)
    load_chunks(admin, graph, [{"source": f"s{i % 10}"} for i in range(3000)], vectors)
    admin.execute(f"CREATE INDEX ON {vec} USING hnsw (embedding vector_cosine_ops)")
    admin.execute(f"ANALYZE {vec}")
    admin.execute(f'ANALYZE {qi(graph)}."Chunk"')
    # At 3k rows the planner would rather sort a filtered label scan; the package's real stores
    # are large enough to prefer the ordered HNSW scan, which this forces.
    admin.execute("SET enable_sort = off")
    q = vec_literal(vectors[7] + 0.01)
    label = f'{qi(graph)}."Chunk"'

    for mode in ("auto", "force_generic_plan"):
        admin.execute(f"SET plan_cache_mode = {mode}")
        for _ in range(6):
            got = admin.execute(KNN_UNFILTERED.format(vec=vec), (q, "3"), prepare=True).fetchall()
            assert len(got) == 3
            assert admin.execute("SHOW hnsw.ef_search").fetchone()[0] == "40"
            off = admin.execute(
                KNN.format(vec=vec, label=label), (q, "40", "off", "s3"), prepare=True
            ).fetchall()
            strict = admin.execute(
                KNN.format(vec=vec, label=label),
                (q, "40", "strict_order", "s3"),
                prepare=True,
            ).fetchall()
            assert len(off) < 10, "the filtered search did not use the HNSW index"
            assert len(strict) == 10


# 4 ------------------------------------------------------------------------------------------------
def test_04_filter_indexes_chosen(server: Server, graph: str, admin: psycopg.Connection) -> None:
    """A SQL btree on a declared key and a GIN jsonb_path_ops catch-all on the label are chosen."""
    admin.execute('CREATE VLABEL "Chunk"')
    load_chunks(
        admin,
        graph,
        [{"bucket": f"b{i % 1000}", "source": f"s{i % 10}"} for i in range(20000)],
    )
    admin.execute(f"CREATE INDEX chunk_bucket ON {qi(graph)}.\"Chunk\" ((properties->'bucket'))")
    admin.execute(
        f'CREATE INDEX chunk_gin ON {qi(graph)}."Chunk" USING gin (properties jsonb_path_ops)'
    )
    admin.execute(f'ANALYZE {qi(graph)}."Chunk"')
    by_key = "\n".join(
        r[0]
        for r in admin.execute(
            f"EXPLAIN SELECT id FROM {qi(graph)}.\"Chunk\" WHERE properties->'bucket' = %s::jsonb",
            ('"b7"',),
        ).fetchall()
    )
    by_gin = "\n".join(
        r[0]
        for r in admin.execute(
            f'EXPLAIN SELECT id FROM {qi(graph)}."Chunk" WHERE properties @> %s::jsonb',
            ('{"bucket": "b7"}',),
        ).fetchall()
    )
    assert "chunk_bucket" in by_key, by_key
    assert "chunk_gin" in by_gin, by_gin


# 5 ------------------------------------------------------------------------------------------------
def test_05_fts_table_rrf(server: Server, graph: str, admin: psycopg.Connection) -> None:
    """A stored-tsvector side table, and RRF over top-n vector and top-n keyword hits."""
    admin.execute('CREATE VLABEL "Chunk"')
    vec = create_vector_table(admin, graph, 8)
    store = _store(graph)
    admin.execute(
        f'CREATE TABLE {store}.fts (id graphid PRIMARY KEY REFERENCES {qi(graph)}."Chunk" (id) '
        "ON DELETE CASCADE, tsv tsvector NOT NULL)"
    )
    vectors = _unit(200, 8)
    texts = [f"ordinary text number {i}" for i in range(200)]
    texts[42] = "a rare zymurgy passage"
    ids = load_chunks(admin, graph, [{"text": t} for t in texts], vectors)
    admin.execute(
        f"INSERT INTO {store}.fts SELECT id, to_tsvector('english', properties->>'text') "
        f'FROM {qi(graph)}."Chunk"'
    )
    admin.execute(f"CREATE INDEX ON {store}.fts USING gin (tsv)")
    q = vec_literal(vectors[42])
    top = admin.execute(
        f"""
        WITH vh AS (SELECT v.id, row_number() OVER (ORDER BY v.embedding <=> %s::vector, v.id) AS r
                    FROM {vec} v ORDER BY v.embedding <=> %s::vector, v.id LIMIT 20),
             kh AS (SELECT f.id, row_number() OVER (ORDER BY ts_rank_cd(f.tsv, q) DESC, f.id) AS r
                    FROM {store}.fts f, websearch_to_tsquery('english', %s) q
                    WHERE f.tsv @@ q ORDER BY ts_rank_cd(f.tsv, q) DESC, f.id LIMIT 20)
        SELECT coalesce(vh.id, kh.id)::text,
               coalesce(1.0 / (60 + vh.r), 0) + coalesce(1.0 / (60 + kh.r), 0) AS score
        FROM vh FULL OUTER JOIN kh ON vh.id = kh.id
        ORDER BY score DESC, 1 LIMIT 5
        """,
        (q, q, "zymurgy"),
    ).fetchall()
    assert top[0][0] == gid_text(ids[42])


# 6 ------------------------------------------------------------------------------------------------
GRAPH_CONTEXT = """
SELECT s.id::text AS seed, x.rel, x.other::text
FROM (SELECT id FROM {g}."Person" ORDER BY id LIMIT 3) s,
LATERAL (SELECT e.tableoid::regclass::text AS rel, e."end" AS other
         FROM {g}.ag_edge e WHERE e.start = s.id LIMIT 20) x
ORDER BY 1, 2, 3
"""


def test_06_graph_context_privileges(server: Server, graph: str, admin: psycopg.Connection) -> None:
    """Graph context over ag_edge with tableoid, as owner, as app_rw, and as a role missing one edge
    label's grant: only 2.18.6+ refuses that role; has_table_privilege names the label
    everywhere."""
    for stmt in (
        'CREATE VLABEL "Person"',
        'CREATE VLABEL "Org"',
        'CREATE ELABEL "KNOWS"',
        'CREATE ELABEL "WORKS_AT"',
    ):
        admin.execute(stmt)
    admin.execute(
        "CREATE (a:\"Person\" {name: 'a'}), (b:\"Person\" {name: 'b'}), (o:\"Org\" {name: 'o'}), "
        '(a)-[:"KNOWS"]->(b), (a)-[:"WORKS_AT"]->(o), (b)-[:"WORKS_AT"]->(o)'
    )
    sql = GRAPH_CONTEXT.format(g=qi(graph))
    owner_rows = admin.execute(sql).fetchall()
    assert {r[1].split(".")[-1].strip('"') for r in owner_rows} == {"KNOWS", "WORKS_AT"}

    _grant_app(admin, graph)
    _grant_app(admin, graph, PARTIAL_ROLE)
    admin.execute(f'REVOKE SELECT ON {qi(graph)}."WORKS_AT" FROM {PARTIAL_ROLE}')
    assert server.connect(user=APP_ROLE).execute(sql).fetchall() == owner_rows

    partial = server.connect(user=PARTIAL_ROLE)
    missing = partial.execute(
        "SELECT l.labname FROM ag_label l JOIN ag_graph g ON g.oid = l.graphid "
        "WHERE g.graphname = %s AND NOT has_table_privilege(l.relid, 'SELECT')",
        (graph,),
    ).fetchall()
    assert [m[0] for m in missing] == ["WORKS_AT"]
    if server.at_least(2, 18, 6):
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            partial.execute(sql)
    else:
        assert partial.execute(sql).fetchall() == owner_rows


# 7 ------------------------------------------------------------------------------------------------
def test_07_retrieval_bridge(server: Server, graph: str, admin: psycopg.Connection) -> None:
    """Ranked SQL handed to Cypher as a jsonb list, then OPTIONAL MATCH, ORDER BY and LIMIT."""
    admin.execute('CREATE VLABEL "Chunk"')
    admin.execute('CREATE ELABEL "REL"')
    admin.execute('CREATE UNIQUE PROPERTY INDEX "Chunk__id" ON "Chunk" (__id__)')
    vec = create_vector_table(admin, graph, 8)
    vectors = _unit(100, 8)
    load_chunks(admin, graph, [{"__id__": f"c{i}"} for i in range(100)], vectors)
    admin.execute(
        'UNWIND %s AS p MATCH (a:"Chunk" {__id__: p.a}), (b:"Chunk" {__id__: p.b}) '
        'CREATE (a)-[:"REL"]->(b)',
        (Jsonb([{"a": f"c{i}", "b": f"c{i + 1}"} for i in range(99)]),),
    )
    q = vectors[5]
    rows = admin.execute(
        f"""
        UNWIND [1] AS a
        WITH (SELECT jsonb_agg(jsonb_build_object('id', h.id::text, 'score', h.score)
                               ORDER BY h.score DESC)
              FROM (SELECT v.id, 1 - (v.embedding <=> %s::vector) AS score
                    FROM {vec} v ORDER BY v.embedding <=> %s::vector LIMIT %s) h) AS hits
        UNWIND hits AS hit
        WITH hit LIMIT %s
        MATCH (node:"Chunk") WHERE id(node) = (hit->>'id')::graphid
        OPTIONAL MATCH (node)-[r:"REL"]->(m:"Chunk")
        WITH node, hit, count(m) AS deg
        RETURN node.__id__ AS k, (hit->>'score')::float8 AS score, deg
        ORDER BY score DESC LIMIT %s
        """,
        (vec_literal(q), vec_literal(q), 5, 5, 5),
    ).fetchall()
    expected = [f"c{i}" for i in np.argsort(-(vectors @ q))[:5]]
    assert [r[0] for r in rows] == expected
    assert [float(r[1]) for r in rows] == sorted((float(r[1]) for r in rows), reverse=True)


# 8 ------------------------------------------------------------------------------------------------
def _chunks_with_edges(admin: psycopg.Connection, graph: str) -> str:
    admin.execute('CREATE VLABEL "Chunk"')
    admin.execute('CREATE VLABEL "Entity"')
    admin.execute('CREATE ELABEL "MENTIONS"')
    admin.execute('CREATE UNIQUE PROPERTY INDEX "Chunk__id" ON "Chunk" (__id__)')
    vec = create_vector_table(admin, graph, 4)
    load_chunks(admin, graph, [{"__id__": f"c{i}"} for i in range(300)], _unit(300, 4))
    admin.execute(
        'UNWIND %s AS n CREATE (:"Entity" {name: n})',
        (Jsonb([f"e{i}" for i in range(50)]),),
    )
    _mention(admin, "c", 300)
    return vec


def _mention(admin: psycopg.Connection, prefix: str, n: int) -> None:
    pairs = [{"c": f"{prefix}{i}", "e": f"e{i % 50}"} for i in range(n)]
    admin.execute(
        'UNWIND %s AS p MATCH (c:"Chunk" {__id__: p.c}), (e:"Entity" {name: p.e}) '
        'CREATE (c)-[:"MENTIONS"]->(e)',
        (Jsonb(pairs),),
    )


def test_08_clear_paths(server: Server, graph: str, admin: psycopg.Connection) -> None:
    """clear() on ≤ 2.16: edges first, then either VACUUM + DELETE, or DROP VLABEL CASCADE with the
    label, its __id__ index and the vector FK recreated in one transaction."""
    vec = _chunks_with_edges(admin, graph)
    g = qi(graph)
    # owned label: edges first, drop and recreate in one transaction
    admin.execute("BEGIN")
    admin.execute(f"TRUNCATE {vec}")
    admin.execute('MATCH (:"Chunk")-[r]-() DELETE r')
    admin.execute('DROP VLABEL "Chunk" CASCADE')
    admin.execute('CREATE VLABEL "Chunk"')
    admin.execute('CREATE UNIQUE PROPERTY INDEX "Chunk__id" ON "Chunk" (__id__)')
    admin.execute(
        f"ALTER TABLE {vec} ADD CONSTRAINT vec_fk FOREIGN KEY (id) "
        f'REFERENCES {g}."Chunk" (id) ON DELETE CASCADE'
    )
    admin.execute("COMMIT")
    assert int(admin.execute('MATCH (c:"Chunk") RETURN count(c)').fetchone()[0]) == 0
    assert int(admin.execute('MATCH ()-[r:"MENTIONS"]->() RETURN count(r)').fetchone()[0]) == 0
    assert int(admin.execute('MATCH (e:"Entity") RETURN count(e)').fetchone()[0]) == 50

    # shared label: edges first, VACUUM sent alone, then DELETE
    load_chunks(admin, graph, [{"__id__": f"d{i}"} for i in range(100)], _unit(100, 4))
    _mention(admin, "d", 100)
    admin.execute('MATCH (:"Chunk")-[r]-() DELETE r')
    admin.execute(f'VACUUM {g}."MENTIONS"')
    admin.execute('MATCH (c:"Chunk") DELETE c')
    assert admin.execute(f"SELECT count(*) FROM {vec}").fetchone()[0] == 0

    # VACUUM must never ride in a pipeline: PG ≥ 15 refuses it there; PG 14.6 does not
    refused = False
    try:
        with admin.pipeline():
            admin.execute("SELECT 1")
            admin.execute(f'VACUUM {g}."MENTIONS"')
    except psycopg.Error as exc:
        refused = sqlstate(exc) == "25001"
    assert refused or server.pg_major == 14


# 9 ------------------------------------------------------------------------------------------------
def test_09_edge_vector_cascade(server: Server, graph: str, admin: psycopg.Connection) -> None:
    """An edge-vector table FK-cascaded to an edge label empties when an endpoint is DETACH
    DELETEd."""
    admin.execute('CREATE VLABEL "Chunk"')
    admin.execute('CREATE ELABEL "REL"')
    admin.execute(f'CREATE UNIQUE INDEX CONCURRENTLY rel_id ON {qi(graph)}."REL" (id)')
    store = _store(graph)
    admin.execute(f"CREATE SCHEMA {store}")
    admin.execute(
        f'CREATE TABLE {store}.evec (id graphid PRIMARY KEY REFERENCES {qi(graph)}."REL" (id) '
        "ON DELETE CASCADE, embedding vector(4) NOT NULL)"
    )
    admin.execute('CREATE (:"Chunk" {__id__: \'a\'})-[:"REL"]->(:"Chunk" {__id__: \'b\'})')
    admin.execute(f"INSERT INTO {store}.evec SELECT id, '[1,0,0,0]' FROM {qi(graph)}.\"REL\"")
    admin.execute("MATCH (a:\"Chunk\" {__id__: 'a'}) DETACH DELETE a")
    assert admin.execute(f"SELECT count(*) FROM {store}.evec").fetchone()[0] == 0


# 10 -----------------------------------------------------------------------------------------------
def _append(conn: psycopg.Connection, graph: str, sid: str, inc: str, seqs: range) -> None:
    msgs = [{"seq": s, "type": "human" if s % 2 == 0 else "ai"} for s in seqs]
    with conn.pipeline():
        conn.execute(
            f"INSERT INTO {_store(graph)}.chat_message SELECT %s, (m->>'seq')::int, m "
            "FROM jsonb_array_elements(%s) m",
            (sid, Jsonb(msgs)),
        )
        conn.execute(
            'MERGE (s:"ChatSession" {session_id: %s, incarnation: %s}) SET s.updated_at = %s',
            (Jsonb(sid), Jsonb(inc), Jsonb("2026-10-30T00:00:00Z")),
        )
        conn.execute(
            'UNWIND %s AS m CREATE (:"ChatMessage" {session_id: %s, seq: m.seq, '
            "incarnation: %s, type: m.type})",
            (Jsonb(msgs), Jsonb(sid), Jsonb(inc)),
        )


def test_10_projection_lifecycle(server: Server, graph: str, admin: psycopg.Connection) -> None:
    """Projection append in the table's pipeline, tombstone, re-create under a new incarnation,
    then purge only the tombstoned vertices."""
    admin.execute(f"CREATE SCHEMA {_store(graph)}")
    admin.execute(
        f"CREATE TABLE {_store(graph)}.chat_message (session_id text, seq int, message jsonb, "
        "PRIMARY KEY (session_id, seq))"
    )
    admin.execute('CREATE VLABEL "ChatSession"')
    admin.execute('CREATE VLABEL "ChatMessage"')
    admin.execute(
        'CREATE UNIQUE PROPERTY INDEX "ChatSession_key" ON "ChatSession" (session_id, incarnation)'
    )
    admin.execute(
        'CREATE UNIQUE PROPERTY INDEX "ChatMessage_key" '
        'ON "ChatMessage" (session_id, seq, incarnation)'
    )
    inc1, inc2 = str(uuid.uuid4()), str(uuid.uuid4())
    _append(admin, graph, "s1", inc1, range(0, 4))

    admin.execute(f"DELETE FROM {_store(graph)}.chat_message WHERE session_id = 's1'")
    admin.execute(
        'MATCH (m:"ChatMessage") WHERE m.session_id = %s AND m.incarnation = %s '
        "SET m._deleted_at = %s, m.type = NULL",
        (Jsonb("s1"), Jsonb(inc1), Jsonb("t1")),
    )
    _append(admin, graph, "s1", inc2, range(0, 4))

    active = admin.execute(
        'MATCH (m:"ChatMessage") WHERE m.session_id = %s AND m._deleted_at IS NULL RETURN count(m)',
        (Jsonb("s1"),),
    ).fetchone()[0]
    assert int(active) == 4
    admin.execute('MATCH (m:"ChatMessage") WHERE m._deleted_at IS NOT NULL DETACH DELETE m')
    remaining = admin.execute(
        'MATCH (m:"ChatMessage") RETURN m.incarnation AS i, count(m) AS n'
    ).fetchall()
    assert [(r[0], int(r[1])) for r in remaining] == [(inc2, 4)]


# 11 -----------------------------------------------------------------------------------------------
def test_11_drop_property_index_on_sql_index(
    server: Server, graph: str, admin: psycopg.Connection
) -> None:
    """DROP PROPERTY INDEX removes an index created with plain SQL on a label table."""
    admin.execute('CREATE VLABEL "Chunk"')
    admin.execute(f"CREATE INDEX chunk_k ON {qi(graph)}.\"Chunk\" ((properties->'k'))")
    admin.execute("DROP PROPERTY INDEX chunk_k")
    assert admin.execute("SELECT to_regclass(%s)", (f"{qi(graph)}.chunk_k",)).fetchone()[0] is None


# 12 -----------------------------------------------------------------------------------------------
def test_12_graph_documents_pipeline(server: Server, graph: str, admin: psycopg.Connection) -> None:
    """add_graph_documents' pipeline is idempotent, and a unique (start, end) index stops
    duplicates."""
    admin.execute('CREATE VLABEL "Person"')
    admin.execute('CREATE ELABEL "KNOWS"')
    admin.execute('CREATE UNIQUE PROPERTY INDEX "Person_id" ON "Person" (id)')
    admin.execute(f'CREATE UNIQUE INDEX knows_pair ON {qi(graph)}."KNOWS" (start, "end")')
    people = [{"id": f"p{i}", "props": {"name": f"n{i}", "age": i}} for i in range(10)]
    edges = [{"s": f"p{i}", "t": f"p{(i + 1) % 10}"} for i in range(10)]
    for _ in range(2):
        with admin.pipeline():
            admin.execute(NO_HASH_OR_MERGE_JOINS)
            admin.execute(
                'UNWIND %s AS r MERGE (n:"Person" {id: r.id}) SET n += r.props',
                (Jsonb(people),),
            )
            admin.execute(
                'UNWIND %s AS e MATCH (a:"Person" {id: e.s}), (b:"Person" {id: e.t}) '
                'MERGE (a)-[:"KNOWS"]->(b)',
                (Jsonb(edges),),
            )
    assert int(admin.execute('MATCH (n:"Person") RETURN count(n)').fetchone()[0]) == 10
    assert int(admin.execute('MATCH ()-[r:"KNOWS"]->() RETURN count(r)').fetchone()[0]) == 10
    assert admin.execute("MATCH (n:\"Person\" {id: 'p3'}) RETURN n.age").fetchone()[0] == 3
    with pytest.raises(psycopg.errors.UniqueViolation):
        admin.execute(
            'MATCH (a:"Person" {id: \'p0\'}), (b:"Person" {id: \'p1\'}) CREATE (a)-[:"KNOWS"]->(b)'
        )


# 13 -----------------------------------------------------------------------------------------------
def test_13_identity_survives_dump(server: Server, graph: str, admin: psycopg.Connection) -> None:
    """The graph-identity UUID, stored as the probe label's single identity vertex, survives
    pg_dump/pg_restore; a schema COMMENT change does not touch it.

    A COMMENT ON VLABEL does not survive: pg_dump writes each label table's pg_class comment
    ("base table for graph label …") as COMMENT ON VLABEL, replacing the label's own comment.
    """
    if not server.container:
        pytest.skip("set AGENSGRAPH_COMPAT_DOCKER to run the server's own pg_dump")
    ident = str(uuid.uuid4())
    admin.execute('CREATE VLABEL "__agensgraph_ai__"')
    admin.execute('CREATE (:"__agensgraph_ai__" {identity: %s})', (Jsonb(ident),))
    admin.execute(f"COMMENT ON VLABEL \"__agensgraph_ai__\" IS '{ident}'")
    admin.execute(f"CREATE SCHEMA {_store(graph)}")
    admin.execute(f"COMMENT ON SCHEMA {_store(graph)} IS 'before'")
    probe = f"SELECT properties->>'identity' FROM {qi(graph)}.\"__agensgraph_ai__\""
    label_comment = (
        "SELECT obj_description(l.oid, 'ag_label') FROM ag_label l "
        "JOIN ag_graph g ON g.oid = l.graphid "
        "WHERE g.graphname = %s AND l.labname = '__agensgraph_ai__'"
    )
    assert admin.execute(probe).fetchall() == [(ident,)]
    admin.execute(f"COMMENT ON SCHEMA {_store(graph)} IS 'after'")
    assert admin.execute(probe).fetchall() == [(ident,)]

    restored = f"{DB}_restore"
    dump = f"/tmp/{graph}.dump"
    with server.connect(dbname="postgres") as pg:
        pg.execute(f"DROP DATABASE IF EXISTS {restored}")
        pg.execute(f"CREATE DATABASE {restored}")
    try:
        for argv in (
            ["pg_dump", "-U", server.superuser, "-Fc", "-d", DB, "-f", dump],
            ["pg_restore", "-U", server.superuser, "-d", restored, dump],
        ):
            subprocess.run(
                ["docker", "exec", server.container, *argv],
                check=True,
                capture_output=True,
            )
        with server.connect(dbname=restored) as conn:
            assert conn.execute(probe).fetchall() == [(ident,)]
            assert conn.execute(label_comment, (graph,)).fetchone()[0] != ident
    finally:
        subprocess.run(["docker", "exec", server.container, "rm", "-f", dump], check=False)
        with server.connect(dbname="postgres") as pg:
            pg.execute(f"DROP DATABASE IF EXISTS {restored} WITH (FORCE)")


# 14 -----------------------------------------------------------------------------------------------
def test_14_implicit_pipeline_scope(server: Server, graph: str, admin: psycopg.Connection) -> None:
    """In one implicit-transaction pipeline, set_config(…, true) and an xact advisory lock taken by
    one statement are still in force for the next, and both end with the pipeline."""
    key = 7_340_123_456
    with admin.pipeline():
        admin.execute("SELECT set_config('statement_timeout', '4321', true)")
        setting = admin.execute("SELECT current_setting('statement_timeout')")
        admin.execute("SELECT pg_advisory_xact_lock(%s)", (key,))
        held = admin.execute(
            "SELECT count(*) FROM pg_locks WHERE locktype = 'advisory' "
            "AND pid = pg_backend_pid() AND granted"
        )
    assert setting.fetchone()[0] == "4321ms"
    assert held.fetchone()[0] == 1
    assert admin.execute("SELECT current_setting('statement_timeout')").fetchone()[0] != "4321ms"
    assert (
        admin.execute(
            "SELECT count(*) FROM pg_locks WHERE locktype = 'advisory' AND pid = pg_backend_pid()"
        ).fetchone()[0]
        == 0
    )


# 15 -----------------------------------------------------------------------------------------------
def test_15_read_only_probe(server: Server, graph: str, admin: psycopg.Connection) -> None:
    """The driver's read_only_is_enforced() probe, run as the model role: not enforced below
    2.16."""
    admin.execute('CREATE VLABEL "__agensgraph_ai__"')
    admin.execute('CREATE (:"__agensgraph_ai__" {identity: %s})', (Jsonb(str(uuid.uuid4())),))
    admin.execute(f"GRANT USAGE ON SCHEMA {qi(graph)} TO {LLM_ROLE}")
    admin.execute(f'GRANT SELECT ON {qi(graph)}."__agensgraph_ai__" TO {LLM_ROLE}')
    llm = server.connect(user=LLM_ROLE)
    assert read_only_is_enforced(llm, graph) == server.at_least(2, 16)
    assert admin.execute('MATCH (p:"__agensgraph_ai__") RETURN p.probe').fetchall() == [(None,)]


# 16 -----------------------------------------------------------------------------------------------
def test_16_read_bridge_forms(server: Server, graph: str, admin: psycopg.Connection) -> None:
    """Agent-state rows read next to the knowledge graph, in each form the README documents."""
    store = _store(graph)
    admin.execute(f"CREATE SCHEMA {store}")
    admin.execute(
        f"CREATE TABLE {store}.chat_message (session_id text, seq int, message jsonb, "
        "PRIMARY KEY (session_id, seq))"
    )
    admin.execute(f"""INSERT INTO {store}.chat_message VALUES
        ('s1', 0, '{{"content": "Who founded Acme?", "mentions": ["Acme"]}}'),
        ('s1', 1, '{{"content": "Ada founded Acme.", "mentions": ["Ada", "Acme"]}}')""")
    admin.execute('CREATE VLABEL "Entity"')
    admin.execute(
        "CREATE (:\"Entity\" {name: 'Ada'}), (:\"Entity\" {name: 'Acme'}), "
        "(:\"Entity\" {name: 'Blue'})"
    )
    sub = (
        f"(SELECT jsonb_agg(m.message ORDER BY m.seq) FROM {store}.chat_message m "
        "WHERE m.session_id = 's1')"
    )
    forms = {
        "match_with": f'MATCH (e:"Entity") WITH e, {sub} AS ms UNWIND ms AS msg '
        "WITH e, msg WHERE e.name IN msg.mentions RETURN DISTINCT e.name AS name",
        "unwind_literal_with": f"UNWIND [1] AS a WITH {sub} AS ms UNWIND ms AS msg "
        'MATCH (e:"Entity") WHERE e.name IN msg.mentions RETURN DISTINCT e.name AS name',
        "where_in": 'MATCH (e:"Entity") WHERE e.name IN '
        "(SELECT jsonb_array_elements(m.message->'mentions') "
        f"FROM {store}.chat_message m WHERE m.session_id = 's1') RETURN e.name AS name",
        "sql_join": f"SELECT DISTINCT e.properties->'name' FROM {store}.chat_message m "
        f"JOIN {qi(graph)}.\"Entity\" e ON m.message->'mentions' ? (e.properties->>'name') "
        "WHERE m.session_id = 's1'",
        "cypher_in_from": "SELECT DISTINCT t.name "
        'FROM (MATCH (e:"Entity") RETURN e.name AS name) AS t '
        f"JOIN {store}.chat_message m ON m.message->'mentions' @> jsonb_build_array(t.name) "
        "WHERE m.session_id = 's1'",
    }
    for name, q in forms.items():
        got = sorted(str(r[0]).strip('"') for r in admin.execute(q).fetchall())
        assert got == ["Acme", "Ada"], name


# 17 -----------------------------------------------------------------------------------------------
WRAP_STATEMENTS = [
    'MATCH (n:"Person") RETURN n.name AS name ORDER BY name',
    'MATCH (a:"Person")-[:"KNOWS"]->(b:"Person") RETURN a.name AS a, b.name AS b ORDER BY a, b',
    'MATCH (n:"Person") WHERE n.age > 2 RETURN count(n) AS c',
    "OPTIONAL MATCH (n:\"Person\" {name: 'nobody'}) RETURN n IS NULL AS missing",
    "UNWIND [3, 1, 2] AS x RETURN x ORDER BY x DESC",
    'MATCH (n:"Person") RETURN n.name AS name, n.age AS age ORDER BY age DESC LIMIT 2',
    'MATCH p = (:"Person")-[:"KNOWS"*1..2]->(:"Person") '
    "RETURN length(p) AS len ORDER BY len LIMIT 5",
]


def _wrapped(conn: psycopg.Connection, text: str, max_rows: int) -> list[tuple]:
    with conn.pipeline():
        conn.execute("BEGIN READ ONLY")
        conn.execute("SELECT set_config('statement_timeout', '5000', true)")
        conn.execute(f"DECLARE c NO SCROLL CURSOR FOR SELECT * FROM ({text}) AS t")
        rows = conn.execute(f"FETCH {max_rows + 1} FROM c")
        conn.execute("ROLLBACK")
        conn.execute("SELECT pg_advisory_unlock_all()")
    return rows.fetchall()


def test_17_cursor_wrap(server: Server, graph: str, admin: psycopg.Connection) -> None:
    """The one-round-trip cursor wrap returns the bare statement's rows in order; WITH-first text
    and write clauses are refused by the wrap's grammar."""
    admin.execute('CREATE VLABEL "Person"')
    admin.execute('CREATE ELABEL "KNOWS"')
    admin.execute(
        'UNWIND %s AS p CREATE (:"Person" {name: p.name, age: p.age})',
        (Jsonb([{"name": f"p{i}", "age": i} for i in range(1, 7)]),),
    )
    admin.execute(
        'UNWIND %s AS i MATCH (a:"Person" {age: i}), (b:"Person" {age: i + 1}) '
        'CREATE (a)-[:"KNOWS"]->(b)',
        (Jsonb(list(range(1, 6))),),
    )
    for text in WRAP_STATEMENTS:
        assert _wrapped(admin, text, 100) == admin.execute(text).fetchall(), text
    for text in ("WITH 1 AS x RETURN x", 'MATCH (n:"Person") SET n.x = 1 RETURN n'):
        with pytest.raises(psycopg.errors.SyntaxError):
            _wrapped(admin, text, 10)
        admin.execute("ROLLBACK")


# 18 -----------------------------------------------------------------------------------------------
def test_18_unique_cic_and_invalid_cleanup(
    server: Server, graph: str, admin: psycopg.Connection
) -> None:
    """A unique CREATE INDEX CONCURRENTLY over duplicates raises 23505 and leaves an INVALID index.
    A property-expression index is dropped with DROP PROPERTY INDEX (plain DROP INDEX refuses it
    before 2.18.6); a column index such as (start, "end") with DROP INDEX CONCURRENTLY."""
    admin.execute('CREATE VLABEL "Chunk"')
    admin.execute('CREATE ELABEL "REL"')
    load_chunks(admin, graph, [{"__id__": "dup"}, {"__id__": "dup"}, {"__id__": "x"}])
    admin.execute(
        'MATCH (a:"Chunk" {__id__: \'x\'}), (b:"Chunk" {__id__: \'dup\'}) CREATE (a)-[:"REL"]->(b)'
    )
    g = qi(graph)
    for name, target in (
        ("chunk_uid", f"{g}.\"Chunk\" ((properties->'__id__'))"),
        ("rel_pair", f'{g}."REL" (start, "end")'),
    ):
        if name == "rel_pair":
            admin.execute(
                "MATCH (a:\"Chunk\" {__id__: 'x'}), (b:\"Chunk\") WHERE b.__id__ = 'dup' "
                'CREATE (a)-[:"REL"]->(b)'
            )
        with pytest.raises(psycopg.errors.UniqueViolation):
            admin.execute(f"CREATE UNIQUE INDEX CONCURRENTLY {name} ON {target}")
        valid = admin.execute(
            "SELECT indisvalid FROM pg_index WHERE indexrelid = to_regclass(%s)",
            (f"{g}.{name}",),
        ).fetchone()
        assert valid == (False,)
        if name == "chunk_uid":
            admin.execute(f"DROP PROPERTY INDEX {name}")
        else:
            admin.execute(f"DROP INDEX CONCURRENTLY {g}.{name}")
        assert admin.execute("SELECT to_regclass(%s)", (f"{g}.{name}",)).fetchone()[0] is None


# 19 -----------------------------------------------------------------------------------------------
def test_19_role_graph_path(server: Server, graph: str, admin: psycopg.Connection) -> None:
    """ALTER ROLE … IN DATABASE … SET graph_path gives a role its graph on connect."""
    admin.execute(f"GRANT USAGE ON SCHEMA {qi(graph)} TO {APP_ROLE}")
    admin.execute(f"ALTER ROLE {APP_ROLE} IN DATABASE {DB} SET graph_path = {qi(graph)}")
    app = server.connect(user=APP_ROLE)
    assert app.execute("SHOW graph_path").fetchone()[0] == graph


# 20 -----------------------------------------------------------------------------------------------
BUILD_FN = """
CREATE FUNCTION {s}.agensgraph_build_index(p_name text) RETURNS void
LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog, pg_temp AS $fn$
DECLARE r record; ext name;
BEGIN
  SELECT * INTO r FROM {s}.agensgraph_ai_objects WHERE name = p_name;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'unknown index %', p_name USING ERRCODE = '42704';
  END IF;
  IF r.opclass NOT IN ('vector_cosine_ops', 'vector_l2_ops', 'vector_ip_ops',
                       'halfvec_cosine_ops') THEN
    RAISE EXCEPTION 'opclass % not allowed', r.opclass USING ERRCODE = '42501';
  END IF;
  SELECT n.nspname INTO ext FROM pg_extension e JOIN pg_namespace n ON n.oid = e.extnamespace
  WHERE e.extname = 'vector';
  EXECUTE format('CREATE INDEX IF NOT EXISTS %I ON %I.%I USING hnsw (embedding %I.%s)',
                 r.name, r.schema_name, r.table_name, ext, r.opclass);
END $fn$
"""

ANALYZE_FN = """
CREATE FUNCTION {s}.agensgraph_analyze(p_name text) RETURNS void
LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog, pg_temp AS $fn$
DECLARE r record;
BEGIN
  SELECT * INTO r FROM {s}.agensgraph_ai_objects WHERE name = p_name;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'unknown object %', p_name USING ERRCODE = '42704';
  END IF;
  EXECUTE format('ANALYZE %I.%I', r.schema_name, r.table_name);
END $fn$
"""


def test_20_security_definer_owner_functions(
    server: Server, graph: str, admin: psycopg.Connection
) -> None:
    """app_rw builds and analyzes through owner functions; unknown names and a temp-table spoof
    are refused."""
    admin.execute('CREATE VLABEL "Chunk"')
    vec = create_vector_table(admin, graph, 4)
    load_chunks(admin, graph, [{"i": i} for i in range(50)], _unit(50, 4))
    s = _store(graph)
    admin.execute(
        f"CREATE TABLE {s}.agensgraph_ai_objects (name text PRIMARY KEY, schema_name text, "
        "table_name text, opclass text)"
    )
    admin.execute(
        f"INSERT INTO {s}.agensgraph_ai_objects "
        "VALUES ('vec_hnsw', %s, 'vec', 'vector_cosine_ops')",
        (graph + "_store",),
    )
    for fn in (BUILD_FN, ANALYZE_FN):
        admin.execute(fn.format(s=s))
    for fn in ("agensgraph_build_index(text)", "agensgraph_analyze(text)"):
        admin.execute(f"REVOKE ALL ON FUNCTION {s}.{fn} FROM PUBLIC")
        admin.execute(f"GRANT EXECUTE ON FUNCTION {s}.{fn} TO {APP_ROLE}")
    admin.execute(f"GRANT USAGE ON SCHEMA {s} TO {APP_ROLE}")

    app = server.connect(user=APP_ROLE)
    app.execute(f"SELECT {s}.agensgraph_build_index('vec_hnsw')")
    app.execute(f"SELECT {s}.agensgraph_analyze('vec_hnsw')")
    owner = admin.execute(
        "SELECT pg_get_userbyid(c.relowner) FROM pg_class c WHERE c.oid = to_regclass(%s)",
        (f"{s}.vec_hnsw",),
    ).fetchone()
    assert owner == (server.superuser,)
    assert (
        admin.execute(
            "SELECT last_analyze IS NOT NULL OR last_autoanalyze IS NOT NULL "
            "FROM pg_stat_user_tables WHERE relid = %s::regclass",
            (vec,),
        ).fetchone()
        is not None
    )

    with pytest.raises(psycopg.Error) as unknown:
        app.execute(f"SELECT {s}.agensgraph_build_index('nope')")
    assert sqlstate(unknown.value) == "42704"
    app.execute(
        "CREATE TEMP TABLE agensgraph_ai_objects "
        "(name text, schema_name text, table_name text, opclass text)"
    )
    app.execute(
        "INSERT INTO agensgraph_ai_objects "
        "VALUES ('evil', 'pg_catalog', 'pg_class', 'vector_cosine_ops')"
    )
    with pytest.raises(psycopg.Error) as spoof:
        app.execute(f"SELECT {s}.agensgraph_build_index('evil')")
    assert sqlstate(spoof.value) == "42704"
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        server.connect(user=LLM_ROLE).execute(f"SELECT {s}.agensgraph_build_index('vec_hnsw')")


# 21 -----------------------------------------------------------------------------------------------
def test_21_detach_delete_needs_edge_delete_grant(
    server: Server, graph: str, admin: psycopg.Connection
) -> None:
    """DETACH DELETE as app_rw without DELETE on one touched edge label: refused from 2.17.

    2.16 checks privileges for explicit Cypher writes but still removes a detached vertex's edges
    through its internal edge scan without checking DELETE on the edge label.
    """
    admin.execute('CREATE VLABEL "Chunk"')
    admin.execute('CREATE VLABEL "Entity"')
    admin.execute('CREATE ELABEL "MENTIONS"')
    admin.execute('CREATE (:"Chunk" {__id__: \'a\'})-[:"MENTIONS"]->(:"Entity" {name: \'e\'})')
    _grant_app(admin, graph)
    admin.execute(f'REVOKE DELETE ON {qi(graph)}."MENTIONS" FROM {APP_ROLE}')
    app = server.connect(user=APP_ROLE, graph=graph)
    stmt = "MATCH (c:\"Chunk\" {__id__: 'a'}) DETACH DELETE c"
    if server.at_least(2, 17):
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            app.execute(stmt)
    else:
        app.execute(stmt)
        assert int(admin.execute('MATCH (c:"Chunk") RETURN count(c)').fetchone()[0]) == 0


# 22 -----------------------------------------------------------------------------------------------
def test_22_public_create_check(server: Server, graph: str, admin: psycopg.Connection) -> None:
    """The nspacl check runs on every line; a fresh PG14 grants PUBLIC CREATE on public, PG15+
    does not."""
    assert public_create_granted(admin) == (server.pg_major < 15)
