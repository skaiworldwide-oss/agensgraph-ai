"""Fixtures for the phase-0 compatibility shapes (PLAN.md §9.1).

Each shape runs once per server leg named in ``AGENSGRAPH_COMPAT_DSNS``, a comma-separated list of
``label=conninfo`` pairs whose conninfo reaches a superuser on that server::

    AGENSGRAPH_COMPAT_DSNS="2.14=postgresql://postgres:agens@127.0.0.1:55414/postgres,..."

``AGENSGRAPH_COMPAT_DOCKER`` optionally maps the same labels to container names
(``2.14=agensgraph-test-ag2141-1``) for the shape that needs the server's own ``pg_dump``. Without
servers the shapes skip, unless ``REQUIRE_SERVER=1``.

Every leg gets a fresh ``lcag_compat`` database and the ``lcag_*`` roles; every shape gets its own
graph, dropped with its companion schema afterwards.
"""

from __future__ import annotations

import itertools
import os
import secrets
from collections.abc import Iterator
from dataclasses import dataclass, field

import psycopg
import pytest

from .support import parse_version, qi

DB = "lcag_compat"
ROLE_PASSWORD = "lcag"
APP_ROLE = "lcag_app_rw"
LLM_ROLE = "lcag_llm_ro"
PARTIAL_ROLE = "lcag_partial"
ROLES = (APP_ROLE, LLM_ROLE, PARTIAL_ROLE)


def _pairs(var: str) -> dict[str, str]:
    raw = os.environ.get(var, "")
    return dict(p.split("=", 1) for p in raw.split(",") if "=" in p)


LEGS = _pairs("AGENSGRAPH_COMPAT_DSNS")
CONTAINERS = _pairs("AGENSGRAPH_COMPAT_DOCKER")
_graph_counter = itertools.count(1)


@dataclass
class Server:
    label: str
    conninfo: str
    container: str | None
    version: tuple[int, ...] = ()
    pg_major: int = 0
    superuser: str = ""
    _open: list[psycopg.Connection] = field(default_factory=list)

    def connect(
        self,
        *,
        dbname: str = DB,
        user: str | None = None,
        graph: str | None = None,
        autocommit: bool = True,
    ) -> psycopg.Connection:
        kw: dict = {"dbname": dbname, "autocommit": autocommit}
        if user:
            kw.update(user=user, password=ROLE_PASSWORD)
        conn = psycopg.connect(self.conninfo, **kw)
        if graph:
            conn.execute(f"SET graph_path = {qi(graph)}")
        self._open.append(conn)
        return conn

    def at_least(self, *version: int) -> bool:
        return self.version >= version

    def close_all(self) -> None:
        for conn in self._open:
            conn.close()
        self._open.clear()


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "server" in metafunc.fixturenames:
        labels = sorted(LEGS) or ["<none>"]
        metafunc.parametrize("server", labels, indirect=True, scope="session")


@pytest.fixture(scope="session")
def server(request: pytest.FixtureRequest) -> Iterator[Server]:
    label = request.param
    if label not in LEGS:
        if os.environ.get("REQUIRE_SERVER") == "1":
            pytest.fail("REQUIRE_SERVER=1 but AGENSGRAPH_COMPAT_DSNS names no servers")
        pytest.skip("set AGENSGRAPH_COMPAT_DSNS to run the compatibility shapes")
    srv = Server(label, LEGS[label], CONTAINERS.get(label))
    with psycopg.connect(srv.conninfo, dbname="postgres", autocommit=True) as admin:
        srv.version = parse_version(admin.info.parameter_status("agversion") or "0")
        srv.pg_major = admin.info.server_version // 10000
        srv.superuser = admin.info.user
        admin.execute(f"DROP DATABASE IF EXISTS {DB}")
        admin.execute(f"CREATE DATABASE {DB}")
        for role in ROLES:
            admin.execute(
                f"DO $$ BEGIN IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '{role}') THEN "
                f"CREATE ROLE {role} LOGIN PASSWORD '{ROLE_PASSWORD}'; END IF; END $$"
            )
    with srv.connect() as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    yield srv
    srv.close_all()
    with psycopg.connect(srv.conninfo, dbname="postgres", autocommit=True) as admin:
        admin.execute(f"DROP DATABASE IF EXISTS {DB} WITH (FORCE)")
        for role in ROLES:
            admin.execute(f"DROP ROLE IF EXISTS {role}")


@pytest.fixture
def graph(server: Server) -> Iterator[str]:
    """A fresh graph for one shape; dropped with its companion schema afterwards."""
    name = f"shp{next(_graph_counter)}_{secrets.token_hex(3)}"
    with server.connect() as conn:
        conn.execute(f"CREATE GRAPH {qi(name)}")
    try:
        yield name
    finally:
        server.close_all()
        with server.connect() as conn:
            conn.execute(f"DROP GRAPH IF EXISTS {qi(name)} CASCADE")
            conn.execute(f"DROP SCHEMA IF EXISTS {qi(name + '_store')} CASCADE")
            for role in ROLES:
                conn.execute(f"ALTER ROLE {role} IN DATABASE {DB} RESET ALL")


@pytest.fixture
def admin(server: Server, graph: str) -> psycopg.Connection:
    """A superuser connection with graph_path set to the shape's graph."""
    return server.connect(graph=graph)
