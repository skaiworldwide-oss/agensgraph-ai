"""The round-trip counter counts what the protocol makes a round trip, and nothing else."""

from __future__ import annotations

from collections.abc import Iterator

import agensgraph
import psycopg
import pytest

from tests.wire import Relay, statements

pytestmark = pytest.mark.server


@pytest.fixture
def relay(dsn: str) -> Iterator[Relay]:
    counter = Relay(dsn)
    try:
        yield counter
    finally:
        counter.close()


@pytest.fixture
def conn(relay: Relay) -> Iterator[psycopg.Connection]:
    with psycopg.connect(relay.conninfo, autocommit=True) as connection:
        relay.reset()
        yield connection


def test_one_statement_is_one(relay: Relay, conn: psycopg.Connection) -> None:
    conn.execute("select 1").fetchall()
    assert relay.round_trips == 1


def test_a_pipeline_is_one(relay: Relay, conn: psycopg.Connection) -> None:
    with conn.pipeline():
        for i in range(10):
            conn.execute("select %s::int", (i,))
    assert relay.round_trips == 1


def test_a_copy_is_two(relay: Relay, conn: psycopg.Connection) -> None:
    conn.execute("create temp table t (n int)")
    relay.reset()
    with conn.cursor().copy("copy t from stdin") as copy:
        for i in range(1000):
            copy.write_row((i,))
    assert relay.round_trips == 2


def test_a_large_answer_is_still_one(relay: Relay, conn: psycopg.Connection) -> None:
    rows = conn.execute("select repeat('x', 1000) from generate_series(1, 2000)").fetchall()
    assert len(rows) == 2000
    assert relay.round_trips == 1


def test_the_driver_reports_its_statements(relay: Relay) -> None:
    with agensgraph.connect(relay.conninfo, autocommit=True) as conn, statements() as sent:
        conn.execute("select   1").fetchall()
        conn.execute_query("select %s::int AS n", (2,))
    assert sent == ["select 1", "select %s::int AS n"]
