"""What every test of the package stands on (PLAN.md §9.3).

A server test reaches its server through ``AGENSGRAPH_TEST_DSN``, a superuser's conninfo; one run
tests one release. Without it a test marked ``server`` skips, and with ``REQUIRE_SERVER=1`` a
missing server fails the run instead.

Each worker gets a database of its own, ``lcag_test_<worker>``, made from template1 and so holding
vector, and dropped when the run ends. ``AGENSGRAPH_HOST``, ``AGENSGRAPH_PORT``,
``AGENSGRAPH_USER``, ``AGENSGRAPH_PASSWORD``, ``AGENSGRAPH_DB`` and ``AGENSGRAPH_URL`` are pointed
at it as well, for the tests that read those. A test asking for ``graph_name`` gets a graph of its
own, dropped with its companion schema afterwards.

When the run ends a worker's database must hold nothing a test made: no graph, no companion schema,
no owner function. Anything left fails the run.

``SKIP_LEDGER`` names a TOML file of the skip reasons each release may give; with it set, a skip for
any other reason fails the run.
"""

from __future__ import annotations

import itertools
import os
import secrets
import tomllib
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import quote

import agensgraph
import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo

ADMIN_DSN = os.environ.get("AGENSGRAPH_TEST_DSN", "")
REQUIRE_SERVER = os.environ.get("REQUIRE_SERVER") == "1"
OWNER_FUNCTIONS = ("agensgraph_build_index", "agensgraph_analyze")

_graphs = itertools.count(1)
_worker_dsn = ""
_leg = "offline"
_skips: dict[str, str] = {}


def _worker() -> str:
    return os.environ.get("PYTEST_XDIST_WORKER", "main")


def _database() -> str:
    return f"lcag_test_{_worker()}"


def leg_of(capabilities: agensgraph.Capabilities) -> str:
    """The leg a server is, as the skip ledger names it: ``2.14`` … ``2.17``, ``2.18.4``."""
    release = capabilities.release
    if release[:2] >= (2, 18):
        return ".".join(str(part) for part in release[:3])
    return ".".join(str(part) for part in release[:2])


def _controls_workers(config: pytest.Config) -> bool:
    """Whether this process only hands tests to xdist workers, and runs none itself."""
    return bool(getattr(config.option, "numprocesses", None)) and not hasattr(config, "workerinput")


def pytest_configure(config: pytest.Config) -> None:
    global _worker_dsn, _leg
    config.addinivalue_line("markers", "server: needs a server, through AGENSGRAPH_TEST_DSN")
    if not ADMIN_DSN:
        return
    database = _database()
    with psycopg.connect(ADMIN_DSN, dbname="postgres", autocommit=True) as admin:
        _leg = leg_of(agensgraph.Capabilities(admin.info.parameter_status("agversion") or ""))
        if _controls_workers(config):
            return
        admin.execute(
            sql.SQL("drop database if exists {} with (force)").format(sql.Identifier(database))
        )
        admin.execute(sql.SQL("create database {}").format(sql.Identifier(database)))
    _worker_dsn = make_conninfo(ADMIN_DSN, dbname=database)
    parts = conninfo_to_dict(_worker_dsn)
    host, port = str(parts.get("host", "localhost")), str(parts.get("port", 5432))
    user, password = str(parts.get("user", "")), str(parts.get("password", ""))
    os.environ["AGENSGRAPH_HOST"] = host
    os.environ["AGENSGRAPH_PORT"] = port
    os.environ["AGENSGRAPH_USER"] = user
    os.environ["AGENSGRAPH_PASSWORD"] = password
    os.environ["AGENSGRAPH_DB"] = database
    os.environ["AGENSGRAPH_URL"] = (
        f"postgresql://{quote(user, safe='')}:{quote(password, safe='')}@{host}:{port}/{database}"
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if ADMIN_DSN:
        return
    wanting = [item for item in items if item.get_closest_marker("server")]
    if wanting and REQUIRE_SERVER:
        raise pytest.UsageError("REQUIRE_SERVER=1 but AGENSGRAPH_TEST_DSN is not set")
    skip = pytest.mark.skip(reason="set AGENSGRAPH_TEST_DSN to run the server tests")
    for item in wanting:
        item.add_marker(skip)


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    if report.skipped and isinstance(report.longrepr, tuple):
        _skips[report.nodeid] = str(report.longrepr[2]).removeprefix("Skipped: ")


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    problems = _unexpected_skips(session.config)
    if problems:
        reporter = session.config.pluginmanager.get_plugin("terminalreporter")
        for problem in problems:
            if reporter is not None:
                reporter.write_line(problem, red=True)
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


def pytest_unconfigure(config: pytest.Config) -> None:
    if not _worker_dsn:
        return
    with psycopg.connect(ADMIN_DSN, dbname="postgres", autocommit=True) as admin:
        admin.execute(
            sql.SQL("drop database if exists {} with (force)").format(sql.Identifier(_database()))
        )


@pytest.fixture(scope="session", autouse=True)
def _nothing_left_behind() -> Iterator[None]:
    """Fails the run, as an error after its last test, if the tests left anything behind."""
    yield
    problems = _leftovers()
    if problems:
        pytest.fail("\n".join(problems), pytrace=False)


def _leftovers() -> list[str]:
    """What the tests left in this worker's database: graphs, companion schemas, owner functions."""
    if not _worker_dsn:
        return []
    with psycopg.connect(_worker_dsn, autocommit=True) as conn:
        graphs = [row[0] for row in conn.execute("select graphname::text from ag_graph")]
        schemas = [
            row[0]
            for row in conn.execute(
                r"select nspname::text from pg_namespace where nspname like '%\_store'"
            )
        ]
        functions = [
            row[0]
            for row in conn.execute(
                "select proname::text from pg_proc where proname = any(%s)",
                (list(OWNER_FUNCTIONS),),
            )
        ]
    return [
        f"left behind in {_database()}: {kind} {', '.join(names)}"
        for kind, names in (("graphs", graphs), ("schemas", schemas), ("functions", functions))
        if names
    ]


def _unexpected_skips(config: pytest.Config) -> list[str]:
    ledger = os.environ.get("SKIP_LEDGER")
    if not ledger or getattr(config, "workerinput", None) is not None:
        return []
    allowed = tomllib.loads(Path(ledger).read_text())
    reasons = set(allowed.get("all", {}).get("reasons", [])) | set(
        allowed.get(_leg, {}).get("reasons", [])
    )
    return [
        f"skipped for a reason the ledger does not allow on {_leg}: {nodeid}: {reason}"
        for nodeid, reason in sorted(_skips.items())
        if reason not in reasons
    ]


@pytest.fixture(scope="session")
def dsn() -> str:
    """A superuser's conninfo on this worker's own database."""
    if not _worker_dsn:
        pytest.skip("set AGENSGRAPH_TEST_DSN to run the server tests")
    return _worker_dsn


@pytest.fixture(scope="session")
def leg() -> str:
    """The release this run tests, as the skip ledger names it."""
    return _leg


@pytest.fixture
def graph_name(dsn: str) -> Iterator[str]:
    """A name no other test uses; its graph and companion schema are dropped afterwards."""
    name = f"t{next(_graphs)}_{secrets.token_hex(3)}"
    try:
        yield name
    finally:
        with psycopg.connect(dsn, autocommit=True) as conn:
            conn.execute(sql.SQL("drop graph if exists {} cascade").format(sql.Identifier(name)))
            conn.execute(
                sql.SQL("drop schema if exists {} cascade").format(sql.Identifier(f"{name}_store"))
            )
