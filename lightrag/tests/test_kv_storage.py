"""
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
"""

import pytest
import pytest_asyncio

from lightrag_agensgraph.kg.agensgraph_kv_impl import AgensgraphKVStorage

from conftest import requires_agens

pytestmark = [requires_agens, pytest.mark.asyncio]


def _kv(namespace="full_docs", workspace=""):
    return AgensgraphKVStorage(
        namespace=namespace, workspace=workspace, global_config={}, embedding_func=None
    )


@pytest_asyncio.fixture
async def kv():
    store = _kv()
    await store.initialize()
    await store.drop()
    try:
        yield store
    finally:
        await store.drop()
        await store.finalize()


async def test_crud_and_injected_fields(kv):
    assert await kv.is_empty() is True
    await kv.upsert({"d1": {"content": "hello"}, "d2": {"content": "world"}})
    rec = await kv.get_by_id("d1")
    assert rec["content"] == "hello" and rec["_id"] == "d1"
    assert "create_time" in rec and "update_time" in rec
    assert await kv.is_empty() is False
    assert sorted((await kv.get_all()).keys()) == ["d1", "d2"]
    await kv.delete(["d1"])
    assert await kv.get_by_id("d1") is None


async def test_get_by_ids_order_and_none(kv):
    await kv.upsert({"a": {"v": 1}, "b": {"v": 2}})
    got = await kv.get_by_ids(["b", "ghost", "a"])
    assert [g["_id"] if g else None for g in got] == ["b", None, "a"]


async def test_filter_keys(kv):
    await kv.upsert({"a": {"v": 1}})
    assert await kv.filter_keys({"a", "b", "c"}) == {"b", "c"}


async def test_namespace_isolation(kv):
    await kv.upsert({"shared": {"v": "docs"}})
    cache = _kv(namespace="llm_response_cache")
    await cache.initialize()
    try:
        await cache.upsert({"shared": {"v": "cache"}})
        assert (await kv.get_by_id("shared"))["v"] == "docs"
        assert (await cache.get_by_id("shared"))["v"] == "cache"
    finally:
        await cache.drop()
        await cache.finalize()


async def test_workspace_isolation(kv):
    await kv.upsert({"d1": {"content": "tenantA"}})
    other = _kv(workspace="tenantB")
    await other.initialize()
    try:
        assert await other.get_by_id("d1") is None
    finally:
        await other.drop()
        await other.finalize()
