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

import asyncio
import os

import pytest
from llama_index.core.graph_stores.types import EntityNode, PropertyGraphStore, Relation

from llama_index_agensgraph.graph_stores.agensgraph import AgensPropertyGraphStore

agens_db = os.environ.get("AGENS_DB")
agens_user = os.environ.get("AGENS_USER")
agens_password = os.environ.get("AGENS_PASSWORD")
agens_host = os.environ.get("AGENS_HOST") or "localhost"
agens_port = os.environ.get("AGENS_PORT") or 5432

pytestmark = pytest.mark.skipif(
    not (agens_db and agens_user and agens_password),
    reason="Requires AGENS_DB, AGENS_USER and AGENS_PASSWORD environment variables.",
)

GRAPH = "test_async_parity"

# Every async method the contract declares. The base class answers most of them by
# calling the synchronous one, which holds the event loop for the whole round trip
# -- so an async retriever did a real async vector query and then blocked on the
# more expensive call after it.
ASYNC_METHODS = [
    "adelete",
    "adelete_llama_nodes",
    "aget",
    "aget_llama_nodes",
    "aget_rel_map",
    "aget_schema",
    "aget_schema_str",
    "aget_triplets",
    "astructured_query",
    "aupsert_nodes",
    "aupsert_relations",
    "avector_query",
]


def _conf():
    return {
        "dbname": agens_db,
        "user": agens_user,
        "password": agens_password,
        "host": agens_host,
        "port": agens_port,
    }


@pytest.fixture(scope="module")
def store():
    s = AgensPropertyGraphStore(GRAPH, conf=_conf(), create=True)
    s.structured_query("MATCH (n) DETACH DELETE n")
    nodes = [EntityNode(name=f"n{i}", label="P", properties={"rank": i}) for i in range(30)]
    s.upsert_nodes(nodes)
    s.upsert_relations(
        [
            Relation(source_id=nodes[i].id, target_id=nodes[(i * 7 + 1) % 30].id,
                     label="KNOWS")
            for i in range(30)
        ]
    )
    return s


def test_every_async_method_is_this_class_and_not_the_base(store):
    falling_through = [
        name for name in ASYNC_METHODS if name not in type(store).__dict__
    ]
    assert falling_through == []


@pytest.mark.asyncio
async def test_the_event_loop_keeps_running_during_a_rel_map(store):
    """The base class ran twelve of these with the loop doing nothing else at all."""
    seeds = store.get()[:10]
    ticks = 0

    async def tick():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.001)
            ticks += 1

    ticker = asyncio.create_task(tick())
    try:
        await asyncio.gather(
            *[store.aget_rel_map(seeds, depth=2, limit=30) for _ in range(12)]
        )
    finally:
        ticker.cancel()
    assert ticks > 0, "the loop did nothing else while those were in flight"


@pytest.mark.asyncio
async def test_the_base_class_would_have_blocked_it(store):
    """What the override is for, stated as a measurement rather than a claim."""
    seeds = store.get()[:10]
    ticks = 0

    async def tick():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.001)
            ticks += 1

    ticker = asyncio.create_task(tick())
    try:
        await asyncio.gather(
            *[PropertyGraphStore.aget_rel_map(store, seeds, 2, 30) for _ in range(12)]
        )
    finally:
        ticker.cancel()
    assert ticks == 0


@pytest.mark.asyncio
async def test_async_answers_match_sync(store):
    seeds = store.get()[:5]
    ids = [n.id for n in seeds]

    assert {n.name for n in await store.aget(ids=ids)} == {
        n.name for n in store.get(ids=ids)
    }
    assert len(await store.aget_triplets(ids=ids)) == len(store.get_triplets(ids=ids))
    assert len(await store.aget_rel_map(seeds, depth=2)) == len(
        store.get_rel_map(seeds, depth=2)
    )
    assert await store.aget_schema() == store.get_schema()
    assert await store.aget_schema_str() == store.get_schema_str()


@pytest.mark.asyncio
async def test_adelete_removes_what_delete_would(store):
    doomed = EntityNode(name="doomed", label="P")
    await store.aupsert_nodes([doomed])
    assert [n.name for n in await store.aget(ids=[doomed.id])] == ["doomed"]
    await store.adelete(ids=[doomed.id])
    assert await store.aget(ids=[doomed.id]) == []
