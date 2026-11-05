"""Plain-psycopg helpers for the compatibility shapes.

The shapes test statement forms the 0.4 package will send, on every server line, before any
package code depends on them. They use psycopg directly, so what they show is the server's
behaviour and not the driver's.
"""

from __future__ import annotations

import json
import re
import struct
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import psycopg

COPY_SIGNATURE = b"PGCOPY\n\xff\r\n\0" + b"\0" * 8
COPY_TRAILER = b"\xff\xff"
LOCID_BITS = 48


def qi(name: str) -> str:
    """Quote an identifier."""
    return '"' + name.replace('"', '""') + '"'


def gid(labid: int, locid: int) -> int:
    """A graphid as its packed 64-bit value."""
    return (labid << LOCID_BITS) | locid


def gid_text(value: int) -> str:
    """A packed graphid in the server's text form, ``labid.locid``."""
    return f"{value >> LOCID_BITS}.{value & ((1 << LOCID_BITS) - 1)}"


def parse_version(text: str) -> tuple[int, ...]:
    """``2.18.6.1``, ``2.16`` or ``2.15.0-devel`` as a tuple of ints."""
    return tuple(int(p) for p in re.findall(r"\d+", text.split("-")[0]))


def vec_literal(values: Iterable[float]) -> str:
    return "[" + ",".join(f"{v:.7g}" for v in values) + "]"


@dataclass(frozen=True)
class LabelInfo:
    graph: str
    label: str
    labid: int
    relation: str
    sequence: str

    @property
    def table(self) -> str:
        return f"{qi(self.graph)}.{qi(self.label)}"


def label_info(conn: psycopg.Connection, graph: str, label: str) -> LabelInfo:
    """The label's id and the sequence its id default draws from, found through pg_depend.

    The default itself names the label (``graph_labid('g."L"')``), which goes stale after a
    rename on 2.14.1 and 2.15.0, so ids are never reserved by evaluating it.
    """
    row = conn.execute(
        """
        SELECT l.labid, l.relid::regclass::text, s.oid::regclass::text
        FROM ag_label l
        JOIN ag_graph g ON g.oid = l.graphid
        JOIN pg_attribute a ON a.attrelid = l.relid AND a.attname = 'id'
        JOIN pg_attrdef ad ON ad.adrelid = l.relid AND ad.adnum = a.attnum
        JOIN pg_depend d ON d.classid = 'pg_attrdef'::regclass AND d.objid = ad.oid
                        AND d.refclassid = 'pg_class'::regclass
        JOIN pg_class s ON s.oid = d.refobjid AND s.relkind = 'S'
        WHERE g.graphname = %s AND l.labname = %s
        """,
        (graph, label),
    ).fetchone()
    assert row is not None, f"label {graph}.{label} not found"
    return LabelInfo(graph, label, row[0], row[1], row[2])


def reserve_ids(conn: psycopg.Connection, info: LabelInfo, n: int) -> list[int]:
    """Reserve n graphids for a label in one round trip."""
    rows = conn.execute(
        "SELECT nextval(%s::regclass) FROM generate_series(1, %s)", (info.sequence, n)
    ).fetchall()
    return [gid(info.labid, r[0]) for r in rows]


def _jsonb_field(doc: object) -> bytes:
    body = json.dumps(doc, separators=(",", ":")).encode()
    return struct.pack(">iB", len(body) + 1, 1) + body


def _gid_field(value: int) -> bytes:
    return struct.pack(">iQ", 8, value)


def vertex_copy(ids: Sequence[int], props: Sequence[dict]) -> bytes:
    """A binary COPY stream for ``(id, properties)``."""
    out = bytearray(COPY_SIGNATURE)
    for i, p in zip(ids, props, strict=True):
        out += struct.pack(">h", 2) + _gid_field(i) + _jsonb_field(p)
    return bytes(out + COPY_TRAILER)


def edge_copy(rows: Sequence[tuple[int, int, int, dict]]) -> bytes:
    """A binary COPY stream for ``(id, start, "end", properties)``."""
    out = bytearray(COPY_SIGNATURE)
    for i, s, e, p in rows:
        out += (
            struct.pack(">h", 4) + _gid_field(i) + _gid_field(s) + _gid_field(e) + _jsonb_field(p)
        )
    return bytes(out + COPY_TRAILER)


def vector_copy(ids: Sequence[int], vectors: np.ndarray) -> bytes:
    """A binary COPY stream for ``(id, embedding vector)``."""
    n, dim = vectors.shape
    dt = np.dtype(
        [
            ("nf", ">i2"),
            ("l1", ">i4"),
            ("gid", ">u8"),
            ("l2", ">i4"),
            ("dim", ">u2"),
            ("unused", ">u2"),
            ("v", ">f4", (dim,)),
        ]
    )
    a = np.empty(n, dtype=dt)
    a["nf"], a["l1"], a["gid"] = 2, 8, np.asarray(ids, dtype=np.uint64)
    a["l2"], a["dim"], a["unused"], a["v"] = 4 + 4 * dim, dim, 0, vectors
    return COPY_SIGNATURE + a.tobytes() + COPY_TRAILER


def copy_in(conn: psycopg.Connection, statement: str, payload: bytes) -> None:
    with conn.cursor() as cur, cur.copy(statement) as cp:
        cp.write(payload)


def load_chunks(
    conn: psycopg.Connection,
    graph: str,
    props: Sequence[dict],
    vectors: np.ndarray | None = None,
    *,
    label: str = "Chunk",
    store: str | None = None,
) -> list[int]:
    """Reserve ids, COPY the vertices and, when given, their vectors. Returns the ids."""
    info = label_info(conn, graph, label)
    ids = reserve_ids(conn, info, len(props))
    copy_in(
        conn,
        f"COPY {info.table} (id, properties) FROM STDIN (FORMAT BINARY)",
        vertex_copy(ids, props),
    )
    if vectors is not None:
        copy_in(
            conn,
            f"COPY {qi(store or graph + '_store')}.vec (id, embedding) FROM STDIN (FORMAT BINARY)",
            vector_copy(ids, vectors),
        )
    return ids


def create_vector_table(
    conn: psycopg.Connection,
    graph: str,
    dim: int,
    *,
    label: str = "Chunk",
    name: str = "vec",
) -> str:
    """The companion schema and an FK-cascaded vector table for a label."""
    store = qi(graph + "_store")
    conn.execute(f"CREATE SCHEMA IF NOT EXISTS {store}")
    conn.execute(
        f"CREATE TABLE {store}.{qi(name)} (id graphid PRIMARY KEY "
        f"REFERENCES {qi(graph)}.{qi(label)} (id) ON DELETE CASCADE, "
        f"embedding vector({dim}) NOT NULL) "
        f"WITH (toast_tuple_target = 8160)"
    )
    return f"{store}.{qi(name)}"


def sqlstate(exc: BaseException) -> str | None:
    return getattr(exc, "sqlstate", None)


def read_only_is_enforced(conn: psycopg.Connection, graph: str) -> bool:
    """The driver's probe (PLAN.md A2): a zero-row Cypher SET on the probe label, read only.

    The label holds the identity vertex, so the probe filters it out and touches no row. Success
    means the server let a Cypher write through a READ ONLY transaction. 25006 or 42501 means it
    checked. Any other error is treated as not enforced, so callers fail safe.
    """
    conn.execute("BEGIN READ ONLY")
    try:
        conn.execute(f"SET LOCAL graph_path = {qi(graph)}")
        conn.execute('MATCH (p:"__agensgraph_ai__") WHERE p.identity IS NULL SET p.probe = 1')
    except psycopg.Error as exc:
        return sqlstate(exc) in ("25006", "42501")
    finally:
        conn.execute("ROLLBACK")
    return False


def public_create_granted(conn: psycopg.Connection) -> bool:
    """Whether PUBLIC holds CREATE on schema public (PLAN.md §2.4)."""
    return conn.execute(
        """
        SELECT EXISTS (
            SELECT 1 FROM pg_namespace n,
                 aclexplode(coalesce(n.nspacl, acldefault('n', n.nspowner))) a
            WHERE n.nspname = 'public' AND a.grantee = 0 AND a.privilege_type = 'CREATE')
        """
    ).fetchone()[0]
