"""What each server does, held against what the driver says it does (PLAN.md §5).

The package chooses between its own ways of doing a thing by asking the driver, never by reading a
version. These tests hold the driver to its answers: on each leg, every feature in
``agensgraph.capabilities.FEATURES`` is observed on the server, and a feature the driver says is
there must be there, while one it says is missing must be missing. A new row in the driver's table
fails ``test_every_feature_has_a_probe`` until it is given a probe here.

Every probe runs in a transaction that is rolled back, so the graph the probes share stays as built.
"""

from __future__ import annotations

import secrets
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field

import agensgraph
import psycopg
import pytest
from agensgraph.capabilities import FEATURES
from psycopg.types.json import Jsonb

from .conftest import DB, Server
from .support import parse_version, qi

GRAPH_ROWS = (
    "CREATE VLABEL v",
    "CREATE VLABEL m",
    "CREATE ELABEL e",
    "CREATE ELABEL other",
    "CREATE (:v {k: 1, j: 3, l: [1, null, 2], emb: [1, 2, 3]})",
    "CREATE (:v {k: 2, j: 2})",
    "CREATE (:v {k: 3, j: 1})",
    "MATCH (a:v {k: 1}), (b:v {k: 2}) CREATE (a)-[:e]->(b)",
    "MATCH (b:v {k: 2}), (c:v {k: 3}) CREATE (b)-[:e]->(c)",
    # In this order: where max() and min() over jsonb go wrong, they give the first and last rows.
    "CREATE (:m {x: 1})",
    "CREATE (:m {x: 3})",
    "CREATE (:m {x: 2})",
    "CREATE (:m {x: 20})",
)


@dataclass
class Probing:
    """A connection to a graph holding ``GRAPH_ROWS``, and ways to try statements on it."""

    conn: psycopg.Connection
    graph: str
    notices: list[str] = field(default_factory=list)

    @contextmanager
    def rolled_back(self, *, read_only: bool = False) -> Iterator[psycopg.Connection]:
        with self.conn.transaction(force_rollback=True):
            if read_only:
                self.conn.execute("SET TRANSACTION READ ONLY")
            yield self.conn

    def rows(
        self, statement: str, params: Sequence[object] | None = None, *, read_only: bool = False
    ) -> list[tuple[object, ...]] | None:
        """What a statement returns, or None if the server refuses it."""
        try:
            with self.rolled_back(read_only=read_only) as conn:
                cur = conn.execute(statement, params)
                return cur.fetchall() if cur.description else []
        except psycopg.Error:
            return None

    def runs(self, statement: str, *, read_only: bool = False) -> bool:
        return self.rows(statement, read_only=read_only) is not None

    def runs_as(self, grants: Sequence[str], statement: str) -> bool:
        """Whether a role holding only ``grants`` (``{role}`` and ``{graph}`` filled in) may run
        the statement. The role exists only inside the rolled-back transaction."""
        role = f"lcag_cap_{secrets.token_hex(4)}"
        try:
            with self.rolled_back() as conn:
                conn.execute(f"CREATE ROLE {role} NOLOGIN")
                conn.execute(f"GRANT USAGE ON SCHEMA {qi(self.graph)} TO {role}")
                for grant in grants:
                    conn.execute(grant.format(role=role, graph=qi(self.graph)))
                conn.execute(f"SET LOCAL ROLE {role}")
                conn.execute(statement)
                return True
        except psycopg.Error:
            return False

    def setting_exists(self, name: str) -> bool:
        row = self.conn.execute("SELECT current_setting(%s, true)", (name,)).fetchone()
        return row is not None and row[0] is not None


Probe = Callable[[Probing], bool]
PROBES: dict[str, Probe] = {}


def probe(feature: str) -> Callable[[Probe], Probe]:
    def register(observe: Probe) -> Probe:
        PROBES[feature] = observe
        return observe

    return register


# Safety ------------------------------------------------------------------------------------------


@probe("cypher_write_enforced")
def _read_only_refuses_a_cypher_write(p: Probing) -> bool:
    return not p.runs("MATCH (n:v) WHERE n.k > 100 SET n.a = 1", read_only=True)


@probe("detach_checks_edge_grants")
def _detach_needs_delete_on_the_edge_label(p: Probing) -> bool:
    grants = (
        "GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA {graph} TO {role}",
        "REVOKE DELETE ON {graph}.e FROM {role}",
    )
    return not p.runs_as(grants, "MATCH (n:v {k: 1}) DETACH DELETE n")


@probe("unlabelled_read_checks_label_grants")
def _unlabelled_read_needs_every_label(p: Probing) -> bool:
    grants = (
        "GRANT SELECT ON ALL TABLES IN SCHEMA {graph} TO {role}",
        "REVOKE SELECT ON {graph}.other FROM {role}",
    )
    return not p.runs_as(grants, "MATCH ()-[r]->() RETURN count(*)")


# Statement forms ---------------------------------------------------------------------------------


@probe("order_by_unprojected")
def _order_by_unprojected(p: Probing) -> bool:
    return p.runs("MATCH (n:v) RETURN n.k AS k ORDER BY n.j")


@probe("in_parameter_list")
def _in_parameter_list(p: Probing) -> bool:
    return p.rows("MATCH (n:v) WHERE n.k IN %s RETURN count(*)", (Jsonb([1, 2]),)) == [(2,)]


@probe("cypher_typmod_cast")
def _typmod_cast_keeps_its_dimension(p: Probing) -> bool:
    # 2.16 takes the cast but drops the dimension, so a wrong one is not refused.
    return p.runs("MATCH (n:v {k: 1}) RETURN n.emb::vector(3)") and not p.runs(
        "MATCH (n:v {k: 1}) RETURN n.emb::vector(2)"
    )


@probe("cypher_vector_cast")
def _vector_cast(p: Probing) -> bool:
    return p.runs("MATCH (n:v {k: 1}) RETURN n.emb::vector")


@probe("cypher_sql_functions")
def _sql_functions(p: Probing) -> bool:
    return p.runs(
        "MATCH ()-[r:e]->() RETURN start_id(r), end_id(r), range(1, 2), split('a,b', ','), "
        "jsonb_build_object('a', 1) LIMIT 1"
    )


@probe("range_in_unwind")
def _range_in_unwind(p: Probing) -> bool:
    return p.rows("UNWIND range(1, 3) AS x RETURN x") == [(1,), (2,), (3,)]


@probe("missing_label_read_empty")
def _missing_label_read(p: Probing) -> bool:
    return p.rows("MATCH (n:absent) RETURN n") == []


@probe("missing_label_write_noop")
def _missing_label_write(p: Probing) -> bool:
    return p.runs("MATCH (n:absent) SET n.a = 1")


@probe("merge_creates_label")
def _merge_creates_label(p: Probing) -> bool:
    return p.runs("MERGE (:fresh {k: 1})")


@probe("collect_drops_nulls")
def _collect_drops_nulls(p: Probing) -> bool:
    empty = p.rows("MATCH (n:v) WHERE n.k > 100 RETURN collect(n.k)")
    nulls = p.rows("MATCH (n:v) RETURN collect(n.absent)")
    return empty == nulls == [([],)]


@probe("fast_vertex_delete")
def _vertex_delete_skips_unrelated_edge_labels(p: Probing) -> bool:
    scans = (
        "SELECT seq_scan FROM pg_stat_xact_user_tables WHERE schemaname = %s AND relname = 'other'"
    )
    with p.rolled_back() as conn:
        conn.execute("CREATE (:v {k: 99})")
        before = conn.execute(scans, (p.graph,)).fetchone()
        conn.execute("MATCH (n:v {k: 99}) DETACH DELETE n")
        after = conn.execute(scans, (p.graph,)).fetchone()
    return before == after


@probe("element_comparison")
def _element_comparison(p: Probing) -> bool:
    return p.rows("MATCH (a:v), (b:v) WHERE a < b RETURN count(*)") == [(3,)]


@probe("element_array_negative_index")
def _negative_index(p: Probing) -> bool:
    return p.rows("MATCH p = (a:v {k: 1})-[:e]->(b) RETURN id(nodes(p)[-1]) = id(b)") == [(True,)]


@probe("list_filter_drops_json_null")
def _list_filter(p: Probing) -> bool:
    return p.rows("MATCH (n:v {k: 1}) RETURN [x IN n.l WHERE x IS NOT NULL]") == [([1, 2],)]


@probe("size_of_list")
def _size_of_list(p: Probing) -> bool:
    return p.rows("RETURN size([1, 2, 3])") == [(3,)]


@probe("aggregate_slice_in_with")
def _aggregate_slice(p: Probing) -> bool:
    return p.rows("UNWIND [1, 2, 3] AS x WITH collect(x)[0..2] AS l RETURN l") == [([1, 2],)]


@probe("gql_clauses")
def _gql_offset(p: Probing) -> bool:
    return p.runs("MATCH (n:v) RETURN n.k AS k OFFSET 1")


@probe("property_promotion")
def _promotion_catalog(p: Probing) -> bool:
    return p.rows("SELECT 1 WHERE to_regclass('pg_catalog.ag_label_property') IS NOT NULL") == [
        (1,)
    ]


@probe("endpoint_elision")
def _elision_setting(p: Probing) -> bool:
    return p.setting_exists("enable_graph_endpoint_elision")


@probe("text_cast_sort_in_with")
def _text_cast_sort(p: Probing) -> bool:
    return p.runs("MATCH (n:v) WITH n.k::text AS t ORDER BY t RETURN t")


# Server state ------------------------------------------------------------------------------------


@probe("graph_dml_setting")
def _graph_dml_setting(p: Probing) -> bool:
    return p.setting_exists("enable_graph_dml")


@probe("graphmeta_notices")
def _regather_notices(p: Probing) -> bool:
    p.notices.clear()
    with p.rolled_back() as conn:
        conn.execute("CREATE (:v {k: 60})-[:e]->(:v {k: 61})")
        conn.execute("SELECT regather_graphmeta()")
    return bool(p.notices)


@probe("graphmeta_catalog")
def _graphmeta_valid_column(p: Probing) -> bool:
    return p.rows(
        "SELECT 1 FROM pg_attribute WHERE attrelid = 'pg_catalog.ag_graph'::regclass "
        "AND attname = 'graphmeta_valid' AND NOT attisdropped"
    ) == [(1,)]


@probe("graph_path_reported")
def _graph_path_parameter(p: Probing) -> bool:
    return p.conn.info.parameter_status("graph_path") == p.graph


@probe("write_completes_under_limit")
def _write_under_limit(p: Probing) -> bool:
    with p.rolled_back() as conn:
        conn.execute("MATCH (n:v) CREATE (:w {k: n.k}) RETURN n.k LIMIT 1").fetchall()
        written = conn.execute("MATCH (n:w) RETURN count(*)").fetchone()
    return written == (3,)


@probe("write_counters_fresh")
def _counters_after_a_returning_write(p: Probing) -> bool:
    with p.rolled_back() as conn:
        conn.execute("CREATE (:v {k: 50}), (:v {k: 51})")
        conn.execute("MATCH (n:v {k: 50}) SET n.q = 1 RETURN n.k").fetchall()
        counters = conn.execute("SELECT * FROM get_last_graph_write_stats()").fetchone()
    assert counters is not None
    return counters[0] == 0


# Known bugs: the probe says whether the bug is there --------------------------------------------


@probe("jsonb_minmax_wrong")
def _jsonb_minmax(p: Probing) -> bool:
    return p.rows("MATCH (n:m) RETURN max(n.x), min(n.x)") != [(20, 1)]


@probe("vle_unnamed_wrong")
def _unnamed_vle(p: Probing) -> bool:
    unnamed = p.rows("MATCH (a:v {k: 1})-[:e*1..2]->(c) RETURN count(*)")
    named = p.rows("MATCH (a:v {k: 1})-[r:e*1..2]->(c) RETURN count(*)")
    return unnamed != named


# -------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def probing(server: Server) -> Iterator[Probing]:
    graph = f"capprobe_{secrets.token_hex(3)}"
    with psycopg.connect(server.conninfo, dbname=DB, autocommit=True) as conn:
        found = Probing(conn, graph)
        conn.add_notice_handler(lambda diag: found.notices.append(diag.message_primary or ""))
        conn.execute(f"CREATE GRAPH {qi(graph)}")
        conn.execute(f"SET graph_path = {qi(graph)}")
        for statement in GRAPH_ROWS:
            conn.execute(statement)
        try:
            yield found
        finally:
            conn.execute(f"DROP GRAPH IF EXISTS {qi(graph)} CASCADE")


def test_every_feature_has_a_probe() -> None:
    assert sorted(PROBES) == sorted(FEATURES)


@pytest.mark.parametrize("feature", sorted(PROBES))
def test_the_server_does_what_the_driver_says(probing: Probing, feature: str) -> None:
    capabilities = agensgraph.Capabilities.of(probing.conn)
    expected = capabilities.has(feature)
    assert PROBES[feature](probing) is expected, (
        f"the driver says {capabilities.reported} {'has' if expected else 'lacks'} {feature}"
    )


def test_pgvector_is_at_the_floor(server: Server, probing: Probing) -> None:
    """D10: pgvector 0.8.0 or later, for iterative index scans; 0.8.1 on PostgreSQL 18."""
    row = probing.conn.execute(
        "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
    ).fetchone()
    assert row is not None
    floor = (0, 8, 1) if server.pg_major >= 18 else (0, 8, 0)
    assert parse_version(row[0]) >= floor
