"""What a read is allowed to do, and what it leaves behind.

The refusal under test is the server's, not a reading of the statement, so each payload below
is a different way of writing that a keyword check would have to recognise separately.
"""

from __future__ import annotations

import psycopg
import pytest
from agensgraph.errors import ConfigurationError

from mcp_agensgraph_common.connection import check_role_cannot_run_programs, run_query

from .conftest import ALLOW_SERVER_PROGRAMS


def refusals(graphname: str) -> dict[str, str]:
    """Ways of writing a write that a reading of the statement would have to know separately.

    The SQL ones name the label's own table, because a label *is* a table -- which is the
    reason a boundary made of Cypher keywords is not one.
    """
    label = f'"{graphname}"."Person"'
    return {
        "cypher create": 'CREATE (:"Person" {name: \'Mallory\'})',
        "cypher set": 'MATCH (n:"Person") SET n.age = 0',
        "cypher delete": 'MATCH (n:"Person") DETACH DELETE n',
        "gql insert": 'INSERT (:"Person" {name: \'Mallory\'})',
        "sql truncate": f"TRUNCATE {label}",
        "sql create table": "CREATE TABLE mcp_common_it_marker (i int)",
        "sql drop": f"DROP TABLE {label}",
        "a write after a comment": '-- a read, honestly\nCREATE (:"Person" {name: \'M\'})',
    }


@pytest.mark.parametrize("spelling", list(refusals("g")))
async def test_a_read_only_transaction_refuses_a_write(seeded, graphname, spelling):
    """25006 is `read_only_sql_transaction`, and it is the server saying so."""
    with pytest.raises(psycopg.Error) as raised:
        await run_query(
            seeded,
            refusals(graphname)[spelling],
            read_only=True,
            allow_server_programs=ALLOW_SERVER_PROGRAMS,
        )
    assert raised.value.sqlstate == "25006"


@pytest.mark.parametrize(
    "statement",
    [
        "INSERT INTO {label} (properties) VALUES ('{{}}'::jsonb)",
        "UPDATE {label} SET properties = '{{}}'::jsonb",
        "DELETE FROM {label}",
    ],
    ids=["insert", "update", "delete"],
)
async def test_plain_sql_dml_on_a_label_never_reaches_the_read_only_check(
    seeded, graphname, statement
):
    """The server refuses it earlier, and for a different reason.

    `enable_graph_dml` is off by default and only a superuser may turn it on, so writing to a
    label's table as a table is refused whatever the transaction is -- with a message and no
    SQLSTATE, rather than the 25006 every other spelling gets. Worth a test of its own because
    the difference is in what a caller is told, not in what lands.
    """
    with pytest.raises(Exception, match="enable_graph_dml"):
        await run_query(
            seeded,
            statement.format(label=f'"{graphname}"."Person"'),
            read_only=True,
            allow_server_programs=ALLOW_SERVER_PROGRAMS,
        )
    rows = await run_query(seeded, 'MATCH (n:"Person") RETURN count(*) AS n', read_only=True,
                           allow_server_programs=ALLOW_SERVER_PROGRAMS)
    assert rows == [{"n": 3}]


async def test_a_refused_write_leaves_the_graph_as_it_was(seeded):
    before = await run_query(seeded, 'MATCH (n:"Person") RETURN count(*) AS n', read_only=True,
                             allow_server_programs=ALLOW_SERVER_PROGRAMS)
    with pytest.raises(psycopg.Error):
        await run_query(seeded, 'CREATE (:"Person" {name: \'Mallory\'})', read_only=True,
                        allow_server_programs=ALLOW_SERVER_PROGRAMS)
    after = await run_query(seeded, 'MATCH (n:"Person") RETURN count(*) AS n', read_only=True,
                            allow_server_programs=ALLOW_SERVER_PROGRAMS)
    assert before == after == [{"n": 3}]


async def test_a_setting_a_read_made_does_not_reach_the_next_caller(dsn, graphname):
    """A read-only transaction cannot write, but it can `SET`, and a pool lends the connection on.

    Committed, `search_path` was read back by the following call on the same connection. The
    block ends by rolling back, so it is not.
    """
    from mcp_agensgraph_common.connection import create_pool

    one = create_pool(dsn, graphname, min_size=1, max_size=1, read_timeout=30)
    await one.open()
    await one.wait()
    try:
        before = await run_query(one, "SHOW search_path", read_only=True,
                                 allow_server_programs=ALLOW_SERVER_PROGRAMS)
        await run_query(one, "SET search_path = 'left_behind'", read_only=True,
                        allow_server_programs=ALLOW_SERVER_PROGRAMS)
        after = await run_query(one, "SHOW search_path", read_only=True,
                                allow_server_programs=ALLOW_SERVER_PROGRAMS)
        assert after == before
        assert "left_behind" not in str(after)
    finally:
        await one.close()


async def test_a_read_is_bounded_by_the_pools_own_timeout(dsn, graphname):
    """The limit travels in the connection's options, so it is in force before the first call."""
    from mcp_agensgraph_common.connection import create_pool

    brief = create_pool(dsn, graphname, min_size=1, max_size=1, read_timeout=1)
    await brief.open()
    await brief.wait()
    try:
        with pytest.raises(psycopg.Error) as raised:
            await run_query(brief, "RETURN pg_sleep(5) AS slept", read_only=True,
                            allow_server_programs=ALLOW_SERVER_PROGRAMS)
        assert raised.value.sqlstate == "57014"
    finally:
        await brief.close()


async def test_a_role_that_can_run_a_command_on_the_host_is_refused_at_startup(pool):
    """`COPY ... TO PROGRAM` is not a write, so a read-only transaction has nothing to refuse.

    What stops it is not holding the privilege, so a server connected as a role that does is
    refused before it serves anything.
    """
    async with pool.connection() as conn:
        privileged = await conn.can_run_server_programs()
        await conn.rollback()
    if not privileged:
        pytest.skip("this role cannot run a command on the server's host, which is the point")

    with pytest.raises(RuntimeError, match="COPY"):
        await check_role_cannot_run_programs(pool)
    # And the read tools refuse to open at all, rather than claiming a boundary they do not have.
    with pytest.raises(ConfigurationError):
        await run_query(pool, "RETURN 1 AS one", read_only=True)


async def test_accepting_it_out_loud_is_what_lets_the_server_start(pool):
    await check_role_cannot_run_programs(pool, allow_server_programs=True)
    assert await run_query(pool, "RETURN 1 AS one", read_only=True,
                           allow_server_programs=True) == [{"one": 1}]


async def test_the_graph_is_the_pools_and_survives_a_rollback(seeded):
    """Selecting a graph is a statement inside the transaction, so a rollback undoes it.

    The pool selects it as each connection is made, and a read ends by rolling back -- so this
    is the case that would leave the next call with no graph path at all.
    """
    for _ in range(3):
        rows = await run_query(
            seeded,
            'MATCH (n:"Person") RETURN count(*) AS n',
            read_only=True,
            allow_server_programs=ALLOW_SERVER_PROGRAMS,
        )
        assert rows == [{"n": 3}]
