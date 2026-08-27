'''
Copyright (c) 2025, SKAI Worldwide Co., Ltd.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

import os
import uuid

import psycopg
import pytest
from psycopg import sql
from agensgraph.errors import ConfigurationError
from llama_index.core.graph_stores.types import EntityNode

from llama_index_agensgraph.graph_stores.agensgraph import AgensPropertyGraphStore
from llama_index_agensgraph.graph_stores.agensgraph.utils import AgensQueryException

agens_db = os.environ.get("AGENS_DB")
agens_user = os.environ.get("AGENS_USER")
agens_password = os.environ.get("AGENS_PASSWORD")
agens_host = os.environ.get("AGENS_HOST") or "localhost"
agens_port = os.environ.get("AGENS_PORT") or 5432

pytestmark = pytest.mark.skipif(
    not (agens_db and agens_user and agens_password),
    reason="Requires AGENS_DB, AGENS_USER and AGENS_PASSWORD environment variables.",
)

GRAPH = "test_read_only"
READER = "li_test_reader"


def _conf(**over):
    return {
        "dbname": agens_db,
        "user": agens_user,
        "password": agens_password,
        "host": agens_host,
        "port": agens_port,
        **over,
    }


@pytest.fixture(scope="module")
def reader_store():
    """A store on a role the server will not let write.

    A read-only transaction is not a boundary for a role that may run a command on
    the server's host, and the driver says so rather than pretending otherwise --
    so these have to run as a role that holds nothing.
    """
    owner = AgensPropertyGraphStore(GRAPH, conf=_conf(), create=True)
    owner.structured_query("MATCH (n) DETACH DELETE n")
    owner.upsert_nodes(
        [EntityNode(name="Alice", label="Person", properties={"note": "on DELETE"})]
    )
    password = uuid.uuid4().hex
    with psycopg.connect(autocommit=True, **_conf()) as admin:
        admin.execute(
            f"DO $$ BEGIN "
            f"  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '{READER}') "
            f"  THEN CREATE ROLE {READER} LOGIN; END IF; END $$"
        )
        # A utility statement takes no parameter, so the password is quoted in.
        admin.execute(
            sql.SQL("ALTER ROLE {} PASSWORD {}").format(
                sql.Identifier(READER), sql.Literal(password)
            )
        )
        if admin.execute(
            "SELECT rolsuper OR pg_has_role(%s, 'pg_execute_server_program', 'member')"
            " FROM pg_roles WHERE rolname = %s",
            (READER, READER),
        ).fetchone()[0]:
            pytest.skip(f"{READER} may run server programs")
        # Given every privilege it would need to write, so that what refuses a
        # write here is the transaction and not the grant. Without this the role
        # is refused for lacking permission and the guard is never exercised.
        for schema in (GRAPH, "public", "pg_catalog"):
            admin.execute(f'GRANT USAGE ON SCHEMA "{schema}" TO {READER}')
            admin.execute(
                f'GRANT SELECT, INSERT, UPDATE, DELETE, TRUNCATE '
                f'ON ALL TABLES IN SCHEMA "{schema}" TO {READER}'
            )
            admin.execute(
                f'GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA "{schema}" '
                f'TO {READER}'
            )
    return AgensPropertyGraphStore(
        GRAPH,
        conf=_conf(user=READER, password=password),
        create=False,
        create_indexes=False,
        refresh_schema=False,
    )


def test_read_runs(reader_store):
    with reader_store.read_only():
        rows = reader_store.structured_query(
            'MATCH (n:"Person") RETURN n.name AS name'
        )
    assert [r["name"] for r in rows] == ["Alice"]


def test_a_read_whose_text_says_delete_is_not_a_write(reader_store):
    """The keyword list this replaces refused this one."""
    with reader_store.read_only():
        rows = reader_store.structured_query(
            """MATCH (n:"Person") WHERE n.note = 'on DELETE' RETURN n.name AS name"""
        )
    assert [r["name"] for r in rows] == ["Alice"]


@pytest.mark.parametrize(
    ("what", "statement"),
    [
        ("cypher create", '''CREATE (:"Person" {id: 'x'})'''),
        ("cypher delete", 'MATCH (n) DETACH DELETE n'),
        ("sql truncate", f'TRUNCATE {GRAPH}."Person"'),
        ("sql alter", f'ALTER TABLE {GRAPH}."Person" ADD COLUMN c int'),
        ("sql grant", f"GRANT ALL ON SCHEMA {GRAPH} TO PUBLIC"),
        ("sql insert", f'INSERT INTO {GRAPH}."Person" VALUES (default, default)'),
    ],
)
def test_writes_are_refused(reader_store, what, statement):
    """None of these is Cypher, and the keyword list allowed four of them."""
    with pytest.raises(AgensQueryException):
        with reader_store.read_only():
            reader_store.structured_query(statement)


def test_a_write_after_a_semicolon_is_refused(reader_store):
    """Sent with no parameters the whole string runs and only the first result
    comes back, so the write would land looking like the read."""
    with pytest.raises(ValueError):
        with reader_store.read_only():
            reader_store.structured_query(
                'MATCH (n) RETURN 1 AS one; CREATE (:"Person" {id: \'sneak\'})'
            )


def test_the_graph_is_unchanged(reader_store):
    rows = reader_store.structured_query('MATCH (n:"Person") RETURN count(*) AS c')
    assert rows[0]["c"] == 1


def test_the_block_ends(reader_store):
    """Outside it, a statement is committed again rather than rolled back."""
    with reader_store.read_only():
        reader_store.structured_query("RETURN 1 AS one")
    owner = AgensPropertyGraphStore(GRAPH, conf=_conf(), create=False)
    owner.upsert_nodes([EntityNode(name="Bob", label="Person")])
    assert (
        owner.structured_query('MATCH (n:"Person") RETURN count(*) AS c')[0]["c"] == 2
    )
    owner.structured_query('MATCH (n:"Person") WHERE n.name = \'Bob\' DETACH DELETE n')


def test_a_role_that_may_run_server_programs_is_refused():
    """Refused rather than left to find out: COPY ... TO PROGRAM takes rows out
    rather than putting any in, so a read-only transaction does not stop it."""
    store = AgensPropertyGraphStore(
        GRAPH, conf=_conf(), create=False, create_indexes=False, refresh_schema=False
    )
    with psycopg.connect(autocommit=True, **_conf()) as conn:
        privileged = conn.execute(
            "SELECT rolsuper OR pg_has_role(current_user,"
            " 'pg_execute_server_program', 'member') FROM pg_roles"
            " WHERE rolname = current_user"
        ).fetchone()[0]
    if not privileged:
        pytest.skip("the test role may not run server programs")
    with pytest.raises(ConfigurationError):
        with store.read_only():
            store.structured_query("RETURN 1 AS one")
    # ...and it can be accepted deliberately, which still refuses the write.
    with store.read_only(allow_server_programs=True):
        assert store.structured_query("RETURN 1 AS one")[0]["one"] == 1
        with pytest.raises(AgensQueryException):
            store.structured_query('CREATE (:"Person" {id: \'x\'})')


def test_one_callers_read_only_block_does_not_refuse_anothers_write(reader_store):
    """The depth was an attribute of the store, so it was shared by everyone
    using it: while one thread was inside a read-only block, every other thread's
    write came back 25006, and on a store with no engine both died on the
    transaction nesting."""
    import threading

    owner = AgensPropertyGraphStore(GRAPH, conf=_conf(), create=False)
    # Declared up front: the first write to a label is DDL, and that waits for
    # the open transaction below whatever this test is measuring.
    owner.upsert_nodes([EntityNode(name="warm", label="Person")])

    outcome = {}
    started = threading.Event()

    def other_thread():
        started.set()
        try:
            owner.upsert_nodes([EntityNode(name="fromB", label="Person")])
            outcome["b"] = "wrote"
        except Exception as exc:  # noqa: BLE001 -- recorded, not raised
            outcome["b"] = f"{type(exc).__name__}"

    writer = threading.Thread(target=other_thread)
    with owner.read_only(allow_server_programs=True):
        owner.structured_query("RETURN 1 AS one")
        writer.start()
        started.wait(5)
        writer.join(30)
    assert outcome.get("b") == "wrote", outcome
    owner.delete(ids=["fromB", "warm"])


def test_a_nested_block_asking_for_the_safe_default_gets_it(reader_store):
    """The flag was recorded only on the outermost entry, so a caller asking for
    the safe default inside somebody else's permissive block inherited the
    permission -- skipping the refusal that is the whole boundary against
    COPY ... TO PROGRAM."""
    store = AgensPropertyGraphStore(
        GRAPH, conf=_conf(), create=False, create_indexes=False, refresh_schema=False
    )
    with psycopg.connect(autocommit=True, **_conf()) as conn:
        privileged = conn.execute(
            "SELECT rolsuper OR pg_has_role(current_user,"
            " 'pg_execute_server_program', 'member') FROM pg_roles"
            " WHERE rolname = current_user"
        ).fetchone()[0]
    if not privileged:
        pytest.skip("the test role may not run server programs")

    with store.read_only(allow_server_programs=True):
        store.structured_query("RETURN 1 AS one")
        with pytest.raises(ConfigurationError):
            with store.read_only():
                store.structured_query("RETURN 1 AS one")
        # and the outer block is still usable after the inner one was refused
        assert store.structured_query("RETURN 2 AS two")[0]["two"] == 2
