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

from lightrag.base import DocProcessingStatus, DocStatus

from lightrag_agensgraph.kg.agensgraph_docstatus_impl import AgensgraphDocStatusStorage

from conftest import requires_agens

pytestmark = [requires_agens, pytest.mark.asyncio]


def _rec(status, fp, ch, n=1, track=None):
    return {
        "content_summary": "s",
        "content_length": n,
        "file_path": fp,
        "status": status,
        "created_at": f"2026-01-{n:02d}",
        "updated_at": f"2026-02-{n:02d}",
        "content_hash": ch,
        "track_id": track,
    }


@pytest_asyncio.fixture
async def ds():
    store = AgensgraphDocStatusStorage(
        namespace="doc_status", workspace="", global_config={}, embedding_func=None
    )
    await store.initialize()
    await store.drop()
    await store.upsert(
        {
            "d1": _rec("processed", "/docs/a.txt", "h1", n=1, track="t1"),
            "d2": _rec("pending", "/docs/b.txt", "h2", n=2, track="t1"),
            "d3": _rec("processed", "/x/c.txt", "h3", n=3),
        }
    )
    try:
        yield store
    finally:
        await store.drop()
        await store.finalize()


async def test_status_counts(ds):
    counts = await ds.get_status_counts()
    assert counts["processed"] == 2 and counts["pending"] == 1
    assert (await ds.get_all_status_counts())["all"] == 3


async def test_by_status_and_track(ds):
    by_status = await ds.get_docs_by_status(DocStatus.PROCESSED)
    assert sorted(by_status) == ["d1", "d3"]
    assert all(isinstance(v, DocProcessingStatus) for v in by_status.values())
    assert sorted(await ds.get_docs_by_track_id("t1")) == ["d1", "d2"]


async def test_pagination_and_sort(ds):
    rows, total = await ds.get_docs_paginated(
        sort_field="updated_at", sort_direction="desc"
    )
    assert total == 3
    assert [i for i, _ in rows] == ["d3", "d2", "d1"]  # updated_at desc
    rows2, total2 = await ds.get_docs_paginated(status_filter=DocStatus.PROCESSED)
    assert total2 == 2 and sorted(i for i, _ in rows2) == ["d1", "d3"]


async def test_pagination_sort_whitelist_is_injection_safe(ds):
    rows, total = await ds.get_docs_paginated(
        sort_field="updated_at; DROP TABLE x", sort_direction="weird"
    )
    assert total == 3 and len(rows) == 3  # falls back to safe defaults


async def test_single_doc_lookups(ds):
    assert (await ds.get_doc_by_file_path("/docs/a.txt"))["content_hash"] == "h1"
    assert (await ds.get_doc_by_file_basename("c.txt"))[0] == "d3"
    assert (await ds.get_doc_by_content_hash("h2"))[0] == "d2"
    assert await ds.get_doc_by_file_path("/nope") is None


async def test_filter_keys_and_delete(ds):
    assert await ds.filter_keys({"d1", "zzz"}) == {"zzz"}
    await ds.delete(["d1"])
    assert await ds.get_by_id("d1") is None
