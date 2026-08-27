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

import multiprocessing as mp
import os

import agensgraph
import pytest

agens_db = os.environ.get("AGENS_DB")
agens_user = os.environ.get("AGENS_USER")
agens_password = os.environ.get("AGENS_PASSWORD")
agens_host = os.environ.get("AGENS_HOST") or "localhost"
agens_port = os.environ.get("AGENS_PORT") or 5432

pytestmark = pytest.mark.skipif(
    not (agens_db and agens_user and agens_password),
    reason="Requires AGENS_DB, AGENS_USER and AGENS_PASSWORD environment variables.",
)

GRAPH = "test_concurrency"
WRITERS = 6
KEYS = 20


def _conf():
    return {
        "dbname": agens_db, "user": agens_user, "password": agens_password,
        "host": agens_host, "port": agens_port,
    }


def _writer(worker, queue):
    """Built and written from scratch, as a separate process would."""
    from llama_index.core.graph_stores.types import EntityNode

    from llama_index_agensgraph.graph_stores.agensgraph import AgensPropertyGraphStore

    failures = []
    try:
        store = AgensPropertyGraphStore(GRAPH, conf=_conf(), create=True)
        for key in range(KEYS):
            try:
                store.upsert_nodes(
                    [
                        EntityNode(
                            label="Racer", name=f"k{key}", properties={"by": worker}
                        )
                    ]
                )
            except Exception as exc:  # noqa: BLE001 -- reported, not raised
                failures.append(
                    getattr(getattr(exc, "__cause__", None), "sqlstate", None)
                    or type(exc).__name__
                )
    except Exception as exc:  # noqa: BLE001
        failures.append(f"construction: {type(exc).__name__}: {exc}")
    queue.put(failures)


def test_many_writers_build_the_store_and_write_shared_keys():
    """Both halves of this used to fail.

    Constructing the store was a run of CREATE OR REPLACE statements, so two
    processes doing it at once rewrote the same catalog rows and one lost with
    "tuple concurrently updated"; only two of eight got through. Then a shared
    key was a race nothing retried, and one refusal left the connection unable to
    run anything else, so 95 statements in 200 failed.
    """
    conn = agensgraph.Connection.connect(autocommit=True, **_conf())
    conn.execute(f"DROP GRAPH IF EXISTS {GRAPH} CASCADE")
    conn.close()

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    processes = [ctx.Process(target=_writer, args=(i, queue)) for i in range(WRITERS)]
    for process in processes:
        process.start()
    reported = [queue.get() for _ in processes]
    for process in processes:
        process.join(120)

    failures = [f for batch in reported for f in batch]
    assert failures == [], failures

    conn = agensgraph.Connection.connect(autocommit=True, **_conf())
    conn.graph(GRAPH)
    count = conn.execute('MATCH (n:"Racer") RETURN count(*)').fetchone()[0]
    conn.close()
    # One element per key, not one per writer: a label carries its own uniqueness
    # on id, and a constraint on the parent does not reach a child.
    assert count == KEYS


def test_no_advisory_lock_is_held_between_calls():
    """Declaring a label takes the graph's advisory lock, which lasts as long as
    the transaction holding it.

    On a connection that is not in autocommit a transaction is already open, so
    ``with conn.transaction():`` opens a *savepoint* -- and releasing a savepoint
    ends neither the transaction nor the lock. The store kept it, and the next one
    to touch that graph waited for this one to be garbage collected: the suite
    stopped dead rather than failing.

    This asserts the invariant -- nothing still holds it once the calls that took
    it have returned. It is a guard, not a reproduction: the failure was found by
    the suite stopping dead rather than failing, and most call sequences commit
    soon enough afterwards to release the lock by accident, which is exactly why
    it was invisible until two stores wanted the same graph.
    """
    from llama_index.core.graph_stores.types import EntityNode

    from llama_index_agensgraph.graph_stores.agensgraph import AgensPropertyGraphStore

    conn = agensgraph.Connection.connect(autocommit=True, **_conf())
    conn.execute("DROP GRAPH IF EXISTS test_lock_release CASCADE")
    conn.close()

    store = AgensPropertyGraphStore("test_lock_release", conf=_conf(), create=True)
    store.upsert_nodes([EntityNode(name="a", label="Held")])
    # Nothing between this and the check: a later statement on the same
    # connection would commit and end the lock by accident, which is what makes
    # this hard to see in ordinary use and catastrophic when it happens.
    store._ensure_element_labels(["Fresh"])

    watcher = agensgraph.Connection.connect(autocommit=True, **_conf())
    try:
        held = watcher.execute(
            "SELECT count(*) FROM pg_locks WHERE locktype = 'advisory'"
            " AND objid = (SELECT oid FROM ag_graph WHERE graphname = %s)",
            ("test_lock_release",),
        ).fetchone()[0]
    finally:
        watcher.close()
    store.close()
    assert held == 0, (
        f"{held} advisory lock(s) on the graph are still held after the calls "
        "that took them returned"
    )
