# Copyright (c) 2025, SKAI Worldwide Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The document status store, including the scheduling API LightRAG 1.5.6 drives it through."""

import pytest
import pytest_asyncio
from lightrag.base import (
    CURSOR_END,
    CURSOR_START,
    DocProcessingStatus,
    DocStatus,
    SourceAbsent,
    SourceConflict,
    SourceUnique,
)
from lightrag.exceptions import SourceConflictRepairCASError, StorageRecordNotFoundError

from lightrag_agensgraph.kg.agensgraph_docstatus_impl import AgensgraphDocStatusStorage

pytestmark = pytest.mark.asyncio


def _rec(status, fp, ch, n=1, track=None, **more):
    rec = {
        "content_summary": "s",
        "content_length": n,
        "file_path": fp,
        "status": status,
        "created_at": f"2026-01-{n:02d}T00:00:00+00:00",
        "updated_at": f"2026-02-{n:02d}T00:00:00+00:00",
        "content_hash": ch,
        "track_id": track,
    }
    rec.update(more)
    return rec


@pytest_asyncio.fixture
async def ds():
    store = AgensgraphDocStatusStorage(
        namespace="doc_status", workspace="", global_config={}, embedding_func=None
    )
    await store.initialize()
    await store.drop()
    await store.upsert(
        {
            "d1": _rec("processed", "a.txt", "h1", n=1, track="t1"),
            "d2": _rec("pending", "b.txt", "h2", n=2, track="t1"),
            "d3": _rec("processed", "c.txt", "h3", n=3),
        }
    )
    try:
        yield store
    finally:
        await store.drop()
        await store.finalize()


async def test_status_counts(ds):
    counts = await ds.get_status_counts()
    assert counts["processed"] == 2 and counts["pending"] == 1 and counts["failed"] == 0
    assert (await ds.get_all_status_counts())["all"] == 3
    assert await ds.count_docs_by_statuses([DocStatus.PENDING, DocStatus.PROCESSING]) == 1


async def test_by_status_and_track(ds):
    by_status = await ds.get_docs_by_status(DocStatus.PROCESSED)
    assert set(by_status) == {"d1", "d3"}
    assert all(isinstance(v, DocProcessingStatus) for v in by_status.values())
    assert set(await ds.get_docs_by_statuses([DocStatus.PENDING], strict=True)) == {"d2"}
    assert set(await ds.get_docs_by_track_id("t1")) == {"d1", "d2"}


async def test_a_record_round_trips_with_its_timestamps(ds):
    rec = await ds.get_by_id("d1")
    assert rec["status"] == "processed" and rec["file_path"] == "a.txt"
    assert rec["created_at"] == "2026-01-01T00:00:00+00:00"
    assert rec["chunks_list"] == [] and rec["metadata"] == {}
    assert await ds.get_by_id_strict("nobody") is None
    assert type(ds).supports_strict_point_reads is True
    # LightRAG writes microsecond timestamps with a zone; they come back as written.
    await ds.upsert(
        {"d9": _rec("pending", "z.txt", "h9", n=9, created_at="2026-03-01T12:34:56.789012+00:00")}
    )
    assert (await ds.get_by_id("d9"))["created_at"] == "2026-03-01T12:34:56.789012+00:00"


async def test_created_at_is_kept_when_a_document_is_written_again(ds):
    await ds.upsert({"d1": _rec("processing", "a.txt", "h1", n=5, track="t1")})
    rec = await ds.get_by_id("d1")
    assert rec["status"] == "processing" and rec["updated_at"] == "2026-02-05T00:00:00+00:00"
    assert rec["created_at"] == "2026-01-01T00:00:00+00:00"


async def test_pagination_and_sort(ds):
    rows, total = await ds.get_docs_paginated(
        page=1, page_size=10, sort_field="created_at", sort_direction="asc"
    )
    assert total == 3 and [r[0] for r in rows] == ["d1", "d2", "d3"]
    rows, total = await ds.get_docs_paginated(status_filter=DocStatus.PROCESSED, page_size=10)
    assert total == 2 and {r[0] for r in rows} == {"d1", "d3"}


async def test_pagination_sort_whitelist_is_injection_safe(ds):
    rows, total = await ds.get_docs_paginated(sort_field="id; DROP TABLE x", page_size=10)
    assert total == 3 and len(rows) == 3  # fell back to updated_at


async def test_single_doc_lookups(ds):
    assert (await ds.get_doc_by_file_path("a.txt"))["content_hash"] == "h1"
    assert (await ds.get_doc_by_file_basename("c.txt"))[0] == "d3"
    assert (await ds.get_doc_by_content_hash("h2"))[0] == "d2"
    assert await ds.get_doc_by_content_hash("nope") is None
    assert await ds.get_doc_by_content_hash("") is None


async def test_content_hash_lookup_skips_the_asking_document_and_records_pointing_at_it(ds):
    # d1 holds h1. dup-1 says its content belongs to d1; it is not a second holder.
    await ds.upsert(
        {
            "dup-1": _rec(
                "failed", "a.txt", "h1", n=4, metadata={"is_duplicate": True, "original_doc_id": "d1"}
            ),
            "d7": _rec("pending", "e.txt", "h1", n=7),
        }
    )
    assert (await ds.get_doc_by_content_hash("h1"))[0] == "d1"  # earliest by (created_at, id)
    assert (await ds.get_doc_by_content_hash("h1", exclude_doc_id="d1"))[0] == "d7"
    await ds.delete(["d7"])
    assert await ds.get_doc_by_content_hash("h1", exclude_doc_id="d1") is None


async def test_filter_keys_and_delete(ds):
    assert await ds.filter_keys({"d1", "ghost"}) == {"ghost"}
    await ds.delete(["d1"])
    assert await ds.get_by_id("d1") is None
    assert await ds.is_empty() is False


async def test_update_fields_touches_only_what_it_is_given(ds):
    await ds.update_doc_status_fields(
        "d2", {"status": DocStatus.PROCESSING, "metadata": {"kg_write_state": "pre_graph"}}
    )
    rec = await ds.get_by_id("d2")
    assert rec["status"] == "processing" and rec["metadata"] == {"kg_write_state": "pre_graph"}
    assert rec["file_path"] == "b.txt" and rec["created_at"] == "2026-01-02T00:00:00+00:00"
    with pytest.raises(ValueError):
        await ds.update_doc_status_fields("d2", {"created_at": "2027-01-01T00:00:00+00:00"})
    with pytest.raises(ValueError):
        await ds.update_doc_status_fields("d2", {"not_a_column": 1})
    with pytest.raises(StorageRecordNotFoundError):
        await ds.update_doc_status_fields("ghost", {"status": "failed"})
    await ds.update_doc_status_fields("ghost", {"status": "failed"}, missing_ok=True)


async def test_scheduling_pages_sweep_mixed_statuses_in_created_order(ds):
    await ds.upsert(
        {f"p{i}": _rec("pending" if i % 2 else "failed", f"p{i}.txt", f"hp{i}", n=10 + i) for i in range(7)}
    )
    seen, position, pages = [], CURSOR_START, 0
    while True:
        page = await ds.get_docs_by_statuses_page(
            [DocStatus.PENDING, DocStatus.FAILED], limit=3, position=position, strict=True
        )
        seen.extend(page.docs)
        pages += 1
        if page.next_position is CURSOR_END:
            break
        position = page.next_position
    # d2 (pending, created 2026-01-02) first, then p0..p6 in order; each page holds at most 3.
    assert seen == ["d2"] + [f"p{i}" for i in range(7)]
    assert pages == 3
    record = (await ds.get_docs_by_ids(["p1", "ghost"], strict=True))["p1"]
    assert record.status is DocStatus.PENDING and record.file_path == "p1.txt"
    assert record.has_custom_chunk_journal is False
    full = await ds.get_full_docs_by_ids(["p1", "d1"], strict=True)
    assert set(full) == {"p1", "d1"} and full["d1"].content_hash == "h1"


async def test_a_row_without_created_at_is_reached_skipped_and_raised_under_strict(ds):
    # Only an external edit leaves a row without created_at; the sweep must still get past it.
    await ds._run("UPDATE lightrag_doc_status SET created_at = NULL WHERE id = 'd2'")
    page = await ds.get_docs_by_statuses_page([DocStatus.PENDING, DocStatus.PROCESSED], limit=1)
    assert page.docs == {} and page.next_position is not CURSOR_END  # consumed, not returned
    page2 = await ds.get_docs_by_statuses_page(
        [DocStatus.PENDING, DocStatus.PROCESSED], limit=1, position=page.next_position
    )
    assert list(page2.docs) == ["d1"]
    with pytest.raises(TypeError):
        await ds.get_docs_by_statuses_page([DocStatus.PENDING], limit=5, strict=True)


async def test_source_resolution_and_conflict_repair(ds):
    assert isinstance(await ds.resolve_doc_source_strict("nothing.txt"), SourceAbsent)
    assert isinstance(await ds.resolve_doc_source_strict("unknown_source"), SourceAbsent)
    unique = await ds.resolve_doc_source_strict("a.txt")
    assert isinstance(unique, SourceUnique) and unique.doc_id == "d1"
    await ds.upsert(
        {"d1b": _rec("pending", "a.txt", "h1b", n=6), "d1c": _rec("pending", "a.txt", "h1c", n=8)}
    )
    conflict = await ds.resolve_doc_source_strict("a.txt")
    assert isinstance(conflict, SourceConflict) and conflict.candidate_count == 3
    listing = await ds.list_source_conflicts_page(limit=10)
    assert [c.canonical_source_key for c in listing.conflicts] == ["a.txt"]
    assert listing.conflicts[0].sample_doc_ids == ("d1", "d1b", "d1c") and listing.next_position is CURSOR_END

    plan = await ds.repair_source_conflict(
        "a.txt",
        primary_doc_id="d1",
        expected_candidate_count=0,
        expected_candidate_fingerprint="",
        dry_run=True,
    )
    assert (
        plan.committed is False
        and plan.candidate_count == 3
        and set(plan.demoted_sample_doc_ids) == {"d1b", "d1c"}
    )
    with pytest.raises(SourceConflictRepairCASError):
        await ds.repair_source_conflict(
            "a.txt",
            primary_doc_id="d1",
            expected_candidate_count=2,
            expected_candidate_fingerprint=plan.fingerprint,
            dry_run=False,
        )
    with pytest.raises(ValueError):
        await ds.repair_source_conflict(
            "a.txt",
            primary_doc_id="d3",
            expected_candidate_count=3,
            expected_candidate_fingerprint=plan.fingerprint,
            dry_run=False,
        )
    done = await ds.repair_source_conflict(
        "a.txt",
        primary_doc_id="d1",
        expected_candidate_count=3,
        expected_candidate_fingerprint=plan.fingerprint,
        dry_run=False,
    )
    assert done.committed is True
    demoted = await ds.get_by_id("d1b")
    assert demoted["metadata"] == {"is_duplicate": True, "original_doc_id": "d1"}
    assert isinstance(await ds.resolve_doc_source_strict("a.txt"), SourceUnique)
    assert (await ds.list_source_conflicts_page(limit=10)).conflicts == ()
