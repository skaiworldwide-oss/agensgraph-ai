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
