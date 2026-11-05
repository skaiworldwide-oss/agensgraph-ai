# NOTICE: the tables, statements and delta-history queries in this module are ported from
# langgraph-checkpoint-postgres 3.2.0 (langgraph/checkpoint/postgres/base.py), whose license
# follows.
#
# MIT License
#
# Copyright (c) 2024 LangChain, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""The statements a checkpointer sends, written for its tables in a graph's companion schema.

Three tables hold a thread's state, every one keyed first by ``thread_id``: ``checkpoints``, one row
per checkpoint, its channel values that are JSON scalars kept inline in the ``checkpoint`` column;
``checkpoint_blobs``, every other channel value, serialized, once per channel and version, so a
checkpoint shares the values it did not change with its parent; and ``checkpoint_writes``, the
writes tasks made against a checkpoint, which are its pending writes.

Two more hold the queryable channels: ``checkpoint_items``, the items of each declared channel, and
``checkpoint_threads``, for each thread and namespace the checkpoint those items reflect.
``checkpoint_migrations`` records which entries of :attr:`Tables.migrations` have been applied, by
their position.

Statements name every table with its schema, so nothing depends on the session's ``search_path``.
A blob goes to the server as a binary parameter (``%b``), which saves the text rendering's escaping
on both sides. ``jsonb`` comes back as text, decoded by the saver itself, so how the driver is set
to read numbers does not change what a checkpoint holds.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property

from langgraph.checkpoint.serde.types import TASKS

from .._core.names import literal, qualified

DELTA_PAGE_SIZE = 1024
"""Ancestors read per statement while rebuilding a channel's history."""


@dataclass(frozen=True)
class Tables:
    """One checkpointer's tables in ``schema``, and the statements that use them."""

    schema: str

    # -- names -----------------------------------------------------------------------------

    @cached_property
    def checkpoints(self) -> str:
        return qualified(self.schema, "checkpoints")

    @cached_property
    def blobs(self) -> str:
        return qualified(self.schema, "checkpoint_blobs")

    @cached_property
    def writes(self) -> str:
        return qualified(self.schema, "checkpoint_writes")

    @cached_property
    def items(self) -> str:
        return qualified(self.schema, "checkpoint_items")

    @cached_property
    def threads(self) -> str:
        return qualified(self.schema, "checkpoint_threads")

    # -- setup -----------------------------------------------------------------------------

    @cached_property
    def migrations(self) -> tuple[str, ...]:
        """Append-only: an entry's position is the number recorded once it is applied.

        Plain ``CREATE INDEX``, since setup runs them in a transaction.
        """
        return (
            f"CREATE TABLE IF NOT EXISTS {self.checkpoints} ("
            "thread_id TEXT NOT NULL, "
            "checkpoint_ns TEXT NOT NULL DEFAULT '', "
            "checkpoint_id TEXT NOT NULL, "
            "parent_checkpoint_id TEXT, "
            "type TEXT, "
            "checkpoint JSONB NOT NULL, "
            "metadata JSONB NOT NULL DEFAULT '{}', "
            "PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id))",
            f"CREATE TABLE IF NOT EXISTS {self.blobs} ("
            "thread_id TEXT NOT NULL, "
            "checkpoint_ns TEXT NOT NULL DEFAULT '', "
            "channel TEXT NOT NULL, "
            "version TEXT NOT NULL, "
            "type TEXT NOT NULL, "
            "blob BYTEA, "
            "PRIMARY KEY (thread_id, checkpoint_ns, channel, version))",
            f"CREATE TABLE IF NOT EXISTS {self.writes} ("
            "thread_id TEXT NOT NULL, "
            "checkpoint_ns TEXT NOT NULL DEFAULT '', "
            "checkpoint_id TEXT NOT NULL, "
            "task_id TEXT NOT NULL, "
            "task_path TEXT NOT NULL DEFAULT '', "
            "idx INTEGER NOT NULL, "
            "channel TEXT NOT NULL, "
            "type TEXT, "
            "blob BYTEA NOT NULL, "
            "PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx))",
            "CREATE INDEX IF NOT EXISTS checkpoints_run_id_idx "
            f"ON {self.checkpoints} ((metadata ->> 'run_id'))",
            f"CREATE TABLE IF NOT EXISTS {self.items} ("
            "thread_id TEXT NOT NULL, "
            "checkpoint_ns TEXT NOT NULL, "
            "channel TEXT NOT NULL, "
            "item_key TEXT NOT NULL, "
            "position INTEGER NOT NULL, "
            "value JSONB NOT NULL, "
            "PRIMARY KEY (thread_id, checkpoint_ns, channel, item_key))",
            f"CREATE TABLE IF NOT EXISTS {self.threads} ("
            "thread_id TEXT NOT NULL, "
            "checkpoint_ns TEXT NOT NULL, "
            "checkpoint_id TEXT NOT NULL, "
            "PRIMARY KEY (thread_id, checkpoint_ns))",
        )

    # -- reads -----------------------------------------------------------------------------

    @cached_property
    def select(self) -> str:
        """Checkpoints with their blobs and pending writes, each as an array of ``bytea`` arrays;
        a ``WHERE`` follows."""
        return (
            "select thread_id, checkpoint::text as checkpoint, checkpoint_ns, checkpoint_id, "
            "parent_checkpoint_id, metadata::text as metadata, "
            "(select array_agg(array[bl.channel::bytea, bl.type::bytea, bl.blob]) "
            "from jsonb_each_text(checkpoint -> 'channel_versions') "
            f"inner join {self.blobs} bl "
            "on bl.thread_id = checkpoints.thread_id "
            "and bl.checkpoint_ns = checkpoints.checkpoint_ns "
            "and bl.channel = jsonb_each_text.key "
            "and bl.version = jsonb_each_text.value"
            ") as channel_values, "
            "(select array_agg(array[cw.task_id::text::bytea, cw.channel::bytea, "
            "cw.type::bytea, cw.blob, convert_to(cw.task_path, 'UTF8'), cw.idx::text::bytea]) "
            f"from {self.writes} cw "
            "where cw.thread_id = checkpoints.thread_id "
            "and cw.checkpoint_ns = checkpoints.checkpoint_ns "
            "and cw.checkpoint_id = checkpoints.checkpoint_id"
            ") as pending_writes "
            f"from {self.checkpoints} "
        )

    @cached_property
    def select_pending_sends(self) -> str:
        """The sends each named checkpoint holds, which a checkpoint older than format 4 reads
        from its parent; for threads ``= any(%s)`` and checkpoints ``= any(%s)``."""
        return (
            "select thread_id, checkpoint_id, "
            "array_agg(array[type::bytea, blob] order by task_path, task_id, idx) as sends "
            f"from {self.writes} "
            "where thread_id = any(%s) and checkpoint_id = any(%s) "
            f"and channel = {literal(TASKS)} "
            "group by thread_id, checkpoint_id"
        )

    @cached_property
    def latest_id(self) -> str:
        return (
            f"SELECT checkpoint_id FROM {self.checkpoints} "
            "WHERE thread_id = %s AND checkpoint_ns = %s ORDER BY checkpoint_id DESC LIMIT 1"
        )

    # -- writes ----------------------------------------------------------------------------

    @cached_property
    def upsert_blobs(self) -> str:
        """A thread's new blobs, from arrays of channel, version, type and blob; one already
        stored under the version is kept."""
        return (
            f"INSERT INTO {self.blobs} (thread_id, checkpoint_ns, channel, version, type, blob) "
            "SELECT %s, %s, b.channel, b.version, b.type, b.blob "
            "FROM unnest(%s::text[], %s::text[], %s::text[], %b::bytea[]) "
            "AS b (channel, version, type, blob) "
            "ON CONFLICT (thread_id, checkpoint_ns, channel, version) DO NOTHING"
        )

    @cached_property
    def upsert_checkpoint(self) -> str:
        return (
            f"INSERT INTO {self.checkpoints} (thread_id, checkpoint_ns, checkpoint_id, "
            "parent_checkpoint_id, checkpoint, metadata) "
            "VALUES (%s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (thread_id, checkpoint_ns, checkpoint_id) "
            "DO UPDATE SET checkpoint = EXCLUDED.checkpoint, metadata = EXCLUDED.metadata"
        )

    def _put_writes(self, conflict: str) -> str:
        return (
            f"INSERT INTO {self.writes} (thread_id, checkpoint_ns, checkpoint_id, task_id, "
            "task_path, idx, channel, type, blob) "
            "SELECT %s, %s, %s, %s, %s, w.idx, w.channel, w.type, w.blob "
            "FROM unnest(%s::integer[], %s::text[], %s::text[], %b::bytea[]) "
            "AS w (idx, channel, type, blob) "
            f"ON CONFLICT (thread_id, checkpoint_ns, checkpoint_id, task_id, idx) {conflict}"
        )

    @cached_property
    def upsert_writes(self) -> str:
        """A task's writes, from arrays of index, channel, type and blob, replacing any stored
        under the same index."""
        return self._put_writes(
            "DO UPDATE SET channel = EXCLUDED.channel, type = EXCLUDED.type, blob = EXCLUDED.blob"
        )

    @cached_property
    def insert_writes(self) -> str:
        """A task's writes, keeping any already stored under the same index."""
        return self._put_writes("DO NOTHING")

    # -- queryable channels ------------------------------------------------------------------

    @cached_property
    def advance_pointer(self) -> str:
        """Move the pointer of thread ``%s`` and namespace ``%s`` from the parent ``%s`` to the
        checkpoint ``%s``; a row when it moved."""
        return (
            f"UPDATE {self.threads} SET checkpoint_id = %s "
            "WHERE thread_id = %s AND checkpoint_ns = %s AND checkpoint_id = %s RETURNING 1"
        )

    @cached_property
    def claim_pointer(self) -> str:
        """Point thread ``%s`` and namespace ``%s`` at the checkpoint ``%s`` unless it points at a
        later one; a row when it points at this one. Checkpoint ids sort in the order they were
        made."""
        return (
            f"INSERT INTO {self.threads} AS t (thread_id, checkpoint_ns, checkpoint_id) "
            "VALUES (%s, %s, %s) ON CONFLICT (thread_id, checkpoint_ns) "
            "DO UPDATE SET checkpoint_id = EXCLUDED.checkpoint_id "
            "WHERE t.checkpoint_id <= EXCLUDED.checkpoint_id RETURNING 1"
        )

    @cached_property
    def sync_items(self) -> str:
        """Make a channel's items the ones given, from arrays of key, position and value: delete
        the others, write the new and the changed. Nothing is written unless the pointer is at
        ``%(checkpoint)s``, so a checkpoint that lost the pointer to a later one changes
        nothing."""
        return (
            f"WITH owner AS (SELECT 1 FROM {self.threads} WHERE thread_id = %(thread)s "
            "AND checkpoint_ns = %(ns)s AND checkpoint_id = %(checkpoint)s), "
            f"removed AS (DELETE FROM {self.items} x WHERE x.thread_id = %(thread)s "
            "AND x.checkpoint_ns = %(ns)s AND x.channel = %(channel)s "
            "AND x.item_key <> ALL(%(keys)s::text[]) AND EXISTS (SELECT 1 FROM owner)) "
            f"INSERT INTO {self.items} AS x "
            "(thread_id, checkpoint_ns, channel, item_key, position, value) "
            "SELECT %(thread)s, %(ns)s, %(channel)s, i.item_key, i.position, i.value "
            "FROM unnest(%(keys)s::text[], %(positions)s::integer[], %(values)s::jsonb[]) "
            "AS i (item_key, position, value) WHERE EXISTS (SELECT 1 FROM owner) "
            "ON CONFLICT (thread_id, checkpoint_ns, channel, item_key) "
            "DO UPDATE SET position = EXCLUDED.position, value = EXCLUDED.value "
            "WHERE x.position IS DISTINCT FROM EXCLUDED.position "
            "OR x.value IS DISTINCT FROM EXCLUDED.value"
        )

    def subquery(self, channel: str, element: str, thread: str, namespace: str) -> str:
        """A channel's items as one ``jsonb`` list in position order, each item ``element``."""
        return (
            f"(SELECT coalesce(jsonb_agg({element} ORDER BY i.position), '[]') "
            f"FROM {self.items} i WHERE i.thread_id = {thread} AND i.checkpoint_ns = {namespace} "
            f"AND i.channel = {literal(channel)})"
        )

    # -- deletes and copies ----------------------------------------------------------------

    @cached_property
    def delete_threads(self) -> tuple[str, ...]:
        """Each table's rows for the threads ``= ANY(%s)``."""
        return tuple(
            f"DELETE FROM {table} WHERE thread_id = ANY(%s)"
            for table in (self.checkpoints, self.blobs, self.writes, self.items, self.threads)
        )

    @cached_property
    def delete_for_runs(self) -> str:
        """The checkpoints whose metadata names a run in ``%s``, with their writes and the blobs
        no other checkpoint references; returns each thread and namespace whose pointer was at a
        deleted checkpoint, whose pointer and items are deleted too.

        A child of a deleted checkpoint is left with no parent. A statement does not see its own
        deletes, so the check for other references leaves the deleted checkpoints out by name.
        """
        return (
            f"WITH deleted AS (DELETE FROM {self.checkpoints} "
            "WHERE metadata ->> 'run_id' = ANY(%s) "
            "RETURNING thread_id, checkpoint_ns, checkpoint_id, "
            "checkpoint -> 'channel_versions' AS versions), "
            f"writes AS (DELETE FROM {self.writes} w USING deleted d "
            "WHERE w.thread_id = d.thread_id AND w.checkpoint_ns = d.checkpoint_ns "
            "AND w.checkpoint_id = d.checkpoint_id), "
            f"orphans AS (UPDATE {self.checkpoints} c SET parent_checkpoint_id = NULL "
            "FROM deleted d WHERE c.thread_id = d.thread_id AND c.checkpoint_ns = d.checkpoint_ns "
            "AND c.parent_checkpoint_id = d.checkpoint_id AND NOT EXISTS ("
            "SELECT 1 FROM deleted e WHERE e.thread_id = c.thread_id "
            "AND e.checkpoint_ns = c.checkpoint_ns AND e.checkpoint_id = c.checkpoint_id)), "
            "freed AS (SELECT DISTINCT d.thread_id, d.checkpoint_ns, v.key AS channel, "
            "v.value AS version FROM deleted d CROSS JOIN LATERAL jsonb_each_text(d.versions) v), "
            f"blobs AS (DELETE FROM {self.blobs} b USING freed f "
            "WHERE b.thread_id = f.thread_id AND b.checkpoint_ns = f.checkpoint_ns "
            "AND b.channel = f.channel AND b.version = f.version AND NOT EXISTS ("
            f"SELECT 1 FROM {self.checkpoints} c WHERE c.thread_id = f.thread_id "
            "AND c.checkpoint_ns = f.checkpoint_ns "
            "AND c.checkpoint -> 'channel_versions' ->> f.channel = f.version "
            "AND NOT EXISTS (SELECT 1 FROM deleted e WHERE e.thread_id = c.thread_id "
            "AND e.checkpoint_ns = c.checkpoint_ns AND e.checkpoint_id = c.checkpoint_id))), "
            f"stale AS (DELETE FROM {self.threads} t USING deleted d "
            "WHERE t.thread_id = d.thread_id AND t.checkpoint_ns = d.checkpoint_ns "
            "AND t.checkpoint_id = d.checkpoint_id RETURNING t.thread_id, t.checkpoint_ns), "
            f"cleared AS (DELETE FROM {self.items} i USING stale s "
            "WHERE i.thread_id = s.thread_id AND i.checkpoint_ns = s.checkpoint_ns) "
            "SELECT thread_id, checkpoint_ns FROM stale"
        )

    @cached_property
    def copy_thread(self) -> str:
        """Every row of thread ``%(source)s`` copied to thread ``%(target)s``, unless the target
        holds a checkpoint already; returns whether it did. A row the target holds without a
        checkpoint is kept, or for an item or pointer replaced."""
        unless = "AND NOT (SELECT taken FROM target)"
        return (
            "WITH target AS (SELECT EXISTS ("
            f"SELECT 1 FROM {self.checkpoints} WHERE thread_id = %(target)s) AS taken), "
            f"copied AS (INSERT INTO {self.checkpoints} (thread_id, checkpoint_ns, "
            "checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata) "
            "SELECT %(target)s, checkpoint_ns, checkpoint_id, parent_checkpoint_id, type, "
            f"checkpoint, metadata FROM {self.checkpoints} "
            f"WHERE thread_id = %(source)s {unless}), "
            f"copied_blobs AS (INSERT INTO {self.blobs} (thread_id, checkpoint_ns, channel, "
            "version, type, blob) "
            "SELECT %(target)s, checkpoint_ns, channel, version, type, blob "
            f"FROM {self.blobs} WHERE thread_id = %(source)s {unless} "
            "ON CONFLICT (thread_id, checkpoint_ns, channel, version) DO NOTHING), "
            f"copied_writes AS (INSERT INTO {self.writes} (thread_id, checkpoint_ns, "
            "checkpoint_id, task_id, task_path, idx, channel, type, blob) "
            "SELECT %(target)s, checkpoint_ns, checkpoint_id, task_id, task_path, idx, channel, "
            f"type, blob FROM {self.writes} WHERE thread_id = %(source)s {unless} "
            "ON CONFLICT (thread_id, checkpoint_ns, checkpoint_id, task_id, idx) DO NOTHING), "
            f"copied_items AS (INSERT INTO {self.items} (thread_id, checkpoint_ns, channel, "
            "item_key, position, value) "
            "SELECT %(target)s, checkpoint_ns, channel, item_key, position, value "
            f"FROM {self.items} WHERE thread_id = %(source)s {unless} "
            "ON CONFLICT (thread_id, checkpoint_ns, channel, item_key) "
            "DO UPDATE SET position = EXCLUDED.position, value = EXCLUDED.value), "
            f"copied_threads AS (INSERT INTO {self.threads} (thread_id, checkpoint_ns, "
            "checkpoint_id) "
            "SELECT %(target)s, checkpoint_ns, checkpoint_id "
            f"FROM {self.threads} WHERE thread_id = %(source)s {unless} "
            "ON CONFLICT (thread_id, checkpoint_ns) "
            "DO UPDATE SET checkpoint_id = EXCLUDED.checkpoint_id) "
            "SELECT taken FROM target"
        )

    # -- prune -----------------------------------------------------------------------------
    #
    # The prune statements never join two sets of a thread's rows: a join is planned from
    # per-thread row estimates, which are worst for the few huge threads worth pruning, and can
    # go quadratic. They look rows up by primary key instead (``LATERAL ... LIMIT 1``,
    # ``= ANY``).

    @cached_property
    def prune_delta_threads(self) -> str:
        """The threads, of ``= ANY(%s)`` given twice, that use a ``DeltaChannel``.

        One leaves update counters in checkpoint metadata, or a snapshot: an inline ``true``
        marker with a blob behind it, which a plain ``True`` value never has. LangGraph.js inlines
        no values, but marks the threads its DeltaChannel graphs start with
        ``delta_writes_versioned``.
        """
        return (
            f"SELECT thread_id FROM {self.checkpoints} "
            "WHERE thread_id = ANY(%s) AND ("
            "metadata ? 'counters_since_delta_snapshot' "
            "OR metadata ? 'delta_writes_versioned') "
            "UNION "
            f"SELECT c.thread_id FROM {self.checkpoints} c "
            "CROSS JOIN LATERAL jsonb_each(c.checkpoint -> 'channel_values') AS v "
            "CROSS JOIN LATERAL ("
            f"SELECT 1 FROM {self.blobs} b "
            "WHERE b.thread_id = c.thread_id "
            "AND b.checkpoint_ns = c.checkpoint_ns "
            "AND b.channel = v.key "
            "AND b.version = c.checkpoint -> 'channel_versions' ->> v.key "
            "LIMIT 1) AS snapshot "
            "WHERE c.thread_id = ANY(%s) AND v.value = 'true'::jsonb"
        )

    @cached_property
    def prune_heads(self) -> str:
        """Per namespace of the thread ``%s`` (given four times): the latest checkpoint, the
        parent it reads its pending sends from when it predates format 4, and the root graph's
        checkpoint its subgraph call started from, with whether that one is saved."""
        return (
            "SELECT h.checkpoint_ns, h.checkpoint_id AS head, "
            "CASE WHEN (c.checkpoint ->> 'v')::int < 4 AND EXISTS ("
            f"SELECT 1 FROM {self.writes} w "
            "WHERE w.thread_id = %s "
            "AND w.checkpoint_ns = h.checkpoint_ns "
            "AND w.checkpoint_id = c.parent_checkpoint_id "
            f"AND w.channel = {literal(TASKS)}"
            ") THEN c.parent_checkpoint_id END AS sends_parent, "
            "c.metadata -> 'parents' ->> '' AS root, "
            "EXISTS ("
            f"SELECT 1 FROM {self.checkpoints} r "
            "WHERE r.thread_id = %s AND r.checkpoint_ns = '' "
            "AND r.checkpoint_id = c.metadata -> 'parents' ->> ''"
            ") AS root_saved "
            "FROM ("
            "SELECT DISTINCT ON (checkpoint_ns) checkpoint_ns, checkpoint_id "
            f"FROM {self.checkpoints} "
            "WHERE thread_id = %s "
            "ORDER BY checkpoint_ns DESC, checkpoint_id DESC"
            ") AS h "
            "CROSS JOIN LATERAL ("
            f"SELECT checkpoint, parent_checkpoint_id, metadata FROM {self.checkpoints} "
            "WHERE thread_id = %s "
            "AND checkpoint_ns = h.checkpoint_ns "
            "AND checkpoint_id = h.checkpoint_id"
            ") AS c"
        )

    @cached_property
    def prune_unkept(self) -> str:
        return (
            f"SELECT checkpoint_id FROM {self.checkpoints} "
            "WHERE thread_id = %s AND checkpoint_ns = %s AND checkpoint_id <> ALL(%s)"
        )

    @cached_property
    def prune_writes(self) -> str:
        """The writes of the checkpoints deleted.

        A run saves a superstep's writes in the background, where they can commit before their
        checkpoint does, so they go by the checkpoints deleted.
        """
        return (
            f"DELETE FROM {self.writes} "
            "WHERE thread_id = %s AND checkpoint_ns = %s AND checkpoint_id = ANY(%s)"
        )

    @cached_property
    def prune_checkpoints(self) -> str:
        """Every checkpoint of a namespace but the kept, returning the blobs they referenced."""
        return (
            f"WITH deleted AS (DELETE FROM {self.checkpoints} "
            "WHERE thread_id = %s AND checkpoint_ns = %s AND checkpoint_id <> ALL(%s) "
            "RETURNING checkpoint -> 'channel_versions' AS versions) "
            "SELECT DISTINCT v.key AS channel, v.value AS version "
            "FROM deleted CROSS JOIN LATERAL jsonb_each_text(deleted.versions) AS v"
        )

    @cached_property
    def prune_parents(self) -> str:
        """Kept checkpoints whose parent was deleted are left with none."""
        return (
            f"UPDATE {self.checkpoints} SET parent_checkpoint_id = NULL "
            "WHERE thread_id = %s AND checkpoint_ns = %s "
            "AND checkpoint_id = ANY(%s) AND parent_checkpoint_id <> ALL(%s)"
        )

    @cached_property
    def prune_referenced_blobs(self) -> str:
        return (
            "SELECT DISTINCT v.key AS channel, v.value AS version "
            f"FROM {self.checkpoints} c "
            "CROSS JOIN LATERAL jsonb_each_text(c.checkpoint -> 'channel_versions') AS v "
            "WHERE c.thread_id = %s AND c.checkpoint_ns = %s"
        )

    @cached_property
    def prune_blobs(self) -> str:
        return (
            f"DELETE FROM {self.blobs} "
            "WHERE thread_id = %s AND checkpoint_ns = %s AND channel = %s AND version = ANY(%s)"
        )

    @cached_property
    def prune_subgraph_calls(self) -> tuple[str, ...]:
        """Each table's rows for the namespaces ``= ANY(%s)`` of the thread ``%s``."""
        return tuple(
            f"DELETE FROM {table} WHERE thread_id = %s AND checkpoint_ns = ANY(%s)"
            for table in (self.checkpoints, self.writes, self.blobs, self.items, self.threads)
        )

    # -- delta channel history ---------------------------------------------------------------

    def delta_stage1(self, channels: Sequence[str], *, include_cursor: bool = False) -> str:
        """One page of a thread's checkpoints, newest first from a cursor, with each channel's
        version (``ver_i``), whether a blob is stored for it (``hb_i``), and its inline value as
        JSON text (``inline_i``, null when the checkpoint holds none).

        Parameters: four per channel -- the version lookup, the blob's channel, the version the
        blob must match, the inline lookup -- then the thread, the namespace, the cursor and the
        page size. The first page passes the target as the cursor with ``include_cursor`` so it
        opens with the target's own row, whose parent starts the walk; each later page continues
        below the oldest row read. A checkpoint's ancestors have smaller ids, so no row newer than
        the target is part of its chain.
        """
        columns = [
            f"checkpoint -> 'channel_versions' ->> %s AS ver_{i}, "
            f"EXISTS (SELECT 1 FROM {self.blobs} b{i} "
            f"WHERE b{i}.thread_id = checkpoints.thread_id "
            f"AND b{i}.checkpoint_ns = checkpoints.checkpoint_ns "
            f"AND b{i}.channel = %s "
            f"AND b{i}.version = checkpoint -> 'channel_versions' ->> %s "
            f"AND b{i}.type <> 'empty') AS hb_{i}, "
            f"(checkpoint -> 'channel_values' -> %s)::text AS inline_{i}"
            for i in range(len(channels))
        ]
        return (
            "SELECT checkpoint_id, parent_checkpoint_id, "
            + ", ".join(columns)
            + f" FROM {self.checkpoints} WHERE thread_id = %s AND checkpoint_ns = %s"
            + f" AND checkpoint_id {'<=' if include_cursor else '<'} %s"
            + " ORDER BY checkpoint_id DESC LIMIT %s"
        )

    def delta_stage2(self, *, with_chain: int, with_seed: int) -> str:
        """The writes along each channel's chain and each channel's seed blob, as one
        ``UNION ALL`` of a branch per channel; empty when there are neither.

        Parameters, in order: for each channel with a chain, the thread, the namespace, the
        channel and its chain's checkpoint ids; then for each channel with a seed, the thread,
        the namespace, the channel and the seed's version. A branch per channel reads only that
        channel's own chain, where one filter over every channel and every chain would read the
        longest chain's checkpoints for all of them.
        """
        branches = [
            "SELECT 'w'::text AS _kind, checkpoint_id, channel, type, blob, task_id, task_path, "
            "idx, NULL::text AS version "
            f"FROM {self.writes} "
            "WHERE thread_id = %s AND checkpoint_ns = %s AND channel = %s "
            "AND checkpoint_id = ANY(%s)"
        ] * with_chain
        branches += [
            "SELECT 'b'::text AS _kind, NULL::text AS checkpoint_id, channel, type, blob, "
            "NULL::text AS task_id, NULL::text AS task_path, NULL::int AS idx, version "
            f"FROM {self.blobs} "
            "WHERE thread_id = %s AND checkpoint_ns = %s AND channel = %s AND version = %s"
        ] * with_seed
        return " UNION ALL ".join(branches)


__all__ = ["DELTA_PAGE_SIZE", "Tables"]
