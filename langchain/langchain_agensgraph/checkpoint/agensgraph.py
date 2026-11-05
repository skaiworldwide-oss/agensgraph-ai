# NOTICE: AgensSaver is ported from PostgresSaver and AsyncPostgresSaver of
# langgraph-checkpoint-postgres 3.2.0 (langgraph/checkpoint/postgres/__init__.py, aio.py and
# base.py), whose license follows.
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

"""A LangGraph checkpointer whose tables sit beside an AgensGraph graph.

A thread's checkpoints, channel values and pending writes are rows of plain SQL tables in the
graph's companion schema ``"<graph>_store"`` (see :mod:`._sql` for their layout); the graph itself
need not exist. Values are stored as the serializer gives them, a ``type`` and ``bytea``, and
metadata as ``jsonb``::

    from langchain_agensgraph import AgensEngine
    from langchain_agensgraph.checkpoint import AgensSaver

    engine = AgensEngine("postgresql://app_rw:secret@localhost/db")
    saver = AgensSaver(engine, graph_name="kg", queryable_channels={"messages": "messages"})
    graph = builder.compile(checkpointer=saver)

``queryable_channels`` copies the declared channels of each thread's latest checkpoint into
``checkpoint_items`` as plain JSON, where a Cypher statement reads them next to the graph through
:meth:`AgensSaver.cypher_subquery` (see :mod:`._items` for the modes). A thread's pointer row in
``checkpoint_threads`` names the checkpoint its items reflect; a put moves it and writes the items
in the transaction that saves the checkpoint, and only while the pointer is at that checkpoint, so
a put that loses the pointer to a later checkpoint writes nothing.

One class serves blocking and awaiting graphs: each method and its ``a``-prefixed twin run the same
program (see :mod:`langchain_agensgraph._core.program`), which costs these round trips:

=============================== ===============================================================
``get_tuple``, ``list``         one statement, and one more when a checkpoint older than format
                                4 has a parent, for the sends it reads from it
``put``                         one pipeline: the new blobs, then the checkpoint, in one
                                implicit transaction. With queryable channels the pipeline also
                                moves the pointer and writes the items of the declared channels
                                with a new version; one more when the pointer was not at the
                                parent, as this saver last left it, which then writes every
                                declared channel; and, for a declared channel the checkpoint
                                holds no value of, such as a ``DeltaChannel``, two more to read
                                its history and one to write it
``put_writes``                  one statement, every write an element of its arrays
``delete_thread``               one pipeline
``delete_for_runs``             one statement; with queryable channels, for each thread whose
                                pointer was at a deleted checkpoint, a read and a write of the
                                newest checkpoint left
``copy_thread``                 one statement
``prune``                       ``"delete"``: one pipeline; ``"keep_latest"``: one statement,
                                then a transaction per thread
``get_delta_channel_history``   one statement per 1024 ancestors read and one for the writes and
                                seeds, and one more to find the latest checkpoint when the config
                                names none
=============================== ===============================================================

Nothing is sent when the saver is made. Its first call makes the tables, or brings them up to the
latest migration, on a maintenance connection under the graph's schema lock. Once that is done
through an engine, no saver on that engine checks again; the first call of a saver on another
engine costs one read of the migrations recorded.
"""

from __future__ import annotations

import json
import math
import random
import threading
from collections import OrderedDict, defaultdict
from collections.abc import AsyncIterator, Iterable, Iterator, Mapping, Sequence
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from typing import Any, TypeAlias, cast

from agensgraph import Jsonb, Result
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    DeltaChannelHistory,
    PendingWrite,
    get_checkpoint_id,
    get_serializable_checkpoint_metadata,
    writes_sort_key,
)
from langgraph.checkpoint.serde.base import SerializerProtocol
from langgraph.checkpoint.serde.encrypted import EncryptedSerializer
from langgraph.checkpoint.serde.types import TASKS, _DeltaSnapshot

from .._core.component import Component
from .._core.engine import AgensEngine, bind
from .._core.messages import message_fields
from .._core.names import companion_schema
from .._core.program import InTransaction, Pipeline, Program, Rows, Statement, mappings, scalar
from .._core.setup import Migrations, Setup
from ._items import Mode, checked, folded, items
from ._sql import DELTA_PAGE_SIZE, Tables

PRUNE_STRATEGIES = ("keep_latest", "delete")

SYNCED_THREADS = 4096
"""Threads whose pointer a saver remembers having left at a checkpoint, the least recently used
forgotten first."""

# The shapes of rows. The class defines ``list``, which a ``list[...]`` in its own signatures
# would name instead of the builtin, so those signatures use these.
Row: TypeAlias = dict[str, Any]
RowList: TypeAlias = list[dict[str, Any]]
WriteList: TypeAlias = list[PendingWrite]
BlobRow: TypeAlias = tuple[str, str, str, str, str, bytes | None]
WriteRow: TypeAlias = tuple[str, str, str, str, str, int, str, str, bytes]


@dataclass(frozen=True, slots=True)
class _Target:
    """A checkpoint whose declared channels are written: its config, where it is, its values."""

    config: RunnableConfig
    thread: str
    ns: str
    checkpoint_id: str
    values: Mapping[str, Any]


# What deleting runs leaves to tidy: the threads whose pointer was at a deleted checkpoint, and
# each newest checkpoint left that was written again, with whether it holds the pointer.
_Deleted: TypeAlias = tuple[Sequence[tuple[str, str]], Sequence[tuple[_Target, bool]]]


class AgensSaver(Component, BaseCheckpointSaver[str]):
    """LangGraph checkpoints in tables beside an AgensGraph graph, for blocking and awaiting graphs.

    ``engine`` is an :class:`AgensEngine` or a conninfo string (sharing one engine per string);
    ``graph`` an object with ``engine`` and ``graph_name``, such as an ``AgensGraph``, whose engine
    and graph name are used. ``graph_name`` names the graph otherwise. ``serde`` is the serializer
    for channel values and writes, LangGraph's own by default. With ``setup=False`` the tables are
    taken to exist and are never checked; :meth:`setup` still makes them when called.

    ``queryable_channels`` maps a channel to how its items are read: ``"messages"``, ``"append"``
    or ``"last_value"``. The items are plaintext copies, so with a serializer that encrypts they
    are refused unless ``allow_plaintext=True``.
    """

    _what = "checkpointer"

    def __init__(
        self,
        engine: AgensEngine | str | None = None,
        *,
        graph: Any = None,
        graph_name: str | None = None,
        serde: SerializerProtocol | None = None,
        setup: bool = True,
        queryable_channels: Mapping[str, str] | None = None,
        allow_plaintext: bool = False,
    ) -> None:
        super().__init__(serde=serde)
        self.queryable_channels: dict[str, Mode] = checked(queryable_channels or {})
        if self.queryable_channels and not allow_plaintext and _encrypts(self.serde):
            raise ValueError(
                "queryable_channels keeps plaintext copies of the channels it names, which an "
                "encrypting serializer is meant to prevent; pass allow_plaintext=True to keep them"
            )
        self._engine, self._owns_engine, self.graph_name = bind(engine, graph, graph_name)
        self._tables = Tables(companion_schema(self.graph_name))
        self._setup_wanted = setup
        self._synced: OrderedDict[tuple[str, str], str] = OrderedDict()
        self._synced_lock = threading.Lock()
        migrations = Migrations(
            self._tables.schema, "checkpoint_migrations", dict(enumerate(self._tables.migrations))
        )
        self._setup = Setup(
            self._engine,
            self.graph_name,
            migrations.check,
            migrations.create,
            key=migrations.key,
            forgotten=self._forgotten,
        )

    def _forgotten(self) -> None:
        with self._synced_lock:
            self._synced.clear()

    @classmethod
    @contextmanager
    def from_conn_string(
        cls, conninfo: str, *, graph_name: str, **kwargs: Any
    ) -> Iterator[AgensSaver]:
        """A saver on an engine of its own, closed with the block."""
        with AgensEngine(conninfo) as engine:
            yield cls(engine, graph_name=graph_name, **kwargs)

    @classmethod
    @asynccontextmanager
    async def afrom_conn_string(
        cls, conninfo: str, *, graph_name: str, **kwargs: Any
    ) -> AsyncIterator[AgensSaver]:
        """A saver on an engine of its own, whose pools close with the block."""
        async with AgensEngine(conninfo) as engine:
            yield cls(engine, graph_name=graph_name, **kwargs)

    # -- setup ---------------------------------------------------------------------------------

    def setup(self) -> None:
        """Make the tables, or bring them up to the latest migration, now rather than on the
        first call."""
        self._open()
        self._setup.ensure()

    async def asetup(self) -> None:
        self._open()
        await self._setup.aensure()

    def _ensure(self) -> None:
        self._open()
        if self._setup_wanted:
            self._setup.ensure()

    async def _aensure(self) -> None:
        self._open()
        if self._setup_wanted:
            await self._setup.aensure()

    # -- reading -------------------------------------------------------------------------------

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """The checkpoint ``config`` names, or the latest of its thread and namespace when it
        names none; ``None`` when there is no such checkpoint."""
        self._ensure()
        return self._engine.run(lambda: self._get_tuple(config), replay=True)

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        await self._aensure()
        return await self._engine.arun(lambda: self._get_tuple(config), replay=True)

    def _get_tuple(self, config: RunnableConfig) -> Program[CheckpointTuple | None]:
        thread_id = str(config["configurable"]["thread_id"])
        checkpoint_id = get_checkpoint_id(config)
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        if checkpoint_id:
            args: tuple[Any, ...] = (thread_id, checkpoint_ns, checkpoint_id)
            where = "WHERE thread_id = %s AND checkpoint_ns = %s AND checkpoint_id = %s"
        else:
            args = (thread_id, checkpoint_ns)
            where = (
                "WHERE thread_id = %s AND checkpoint_ns = %s ORDER BY checkpoint_id DESC LIMIT 1"
            )
        values = _checkpoint_rows((yield Statement(self._tables.select + where, args)))
        if not values:
            return None
        yield from self._migrate_sends(values)
        return self._load_checkpoint_tuple(values[0])

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        """Checkpoints newest first: of ``config``'s thread, and of its namespace and checkpoint
        when it names them, or of every thread when ``config`` is ``None``.

        ``filter`` keeps those whose metadata contains it, ``before`` those older than the
        checkpoint it names, and ``limit`` the first that many. With no config every checkpoint of
        every thread is read, and a filter is checked against each.
        """
        self._ensure()
        found = self._engine.run(lambda: self._list(config, filter, before, limit), replay=True)
        for value in found:
            yield self._load_checkpoint_tuple(value)

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        await self._aensure()
        found = await self._engine.arun(
            lambda: self._list(config, filter, before, limit), replay=True
        )
        for value in found:
            yield self._load_checkpoint_tuple(value)

    def _list(
        self,
        config: RunnableConfig | None,
        filter: dict[str, Any] | None,
        before: RunnableConfig | None,
        limit: int | None,
    ) -> Program[RowList]:
        """The checkpoints' rows, each decoded only as the caller reaches it."""
        where, args = self._search_where(config, filter, before)
        query = self._tables.select + where + " ORDER BY checkpoint_id DESC"
        params = [*args]
        if limit is not None:
            query += " LIMIT %s"
            params.append(int(limit))
        values = _checkpoint_rows((yield Statement(query, params)))
        yield from self._migrate_sends(values)
        return values

    def _search_where(
        self,
        config: RunnableConfig | None,
        filter: dict[str, Any] | None,
        before: RunnableConfig | None = None,
    ) -> tuple[str, Sequence[Any]]:
        """The ``WHERE`` clause for ``list``, and its parameters.

        A string in the filter loses its null characters, as one in stored metadata has. A
        ``before`` that names no checkpoint bounds nothing.
        """
        wheres = []
        param_values: list[Any] = []
        if config:
            wheres.append("thread_id = %s ")
            param_values.append(str(config["configurable"]["thread_id"]))
            checkpoint_ns = config["configurable"].get("checkpoint_ns")
            if checkpoint_ns is not None:
                wheres.append("checkpoint_ns = %s")
                param_values.append(checkpoint_ns)
            if checkpoint_id := get_checkpoint_id(config):
                wheres.append("checkpoint_id = %s ")
                param_values.append(checkpoint_id)
        if filter:
            wheres.append("metadata @> %s ")
            param_values.append(Jsonb(_without_nul(filter)))
        if before is not None and (before_id := get_checkpoint_id(before)):
            wheres.append("checkpoint_id < %s ")
            param_values.append(before_id)
        return ("WHERE " + " AND ".join(wheres) if wheres else "", param_values)

    def _migrate_sends(self, values: RowList) -> Program[None]:
        """Give each checkpoint older than format 4 the sends its parent holds, which such a
        checkpoint kept there rather than in its own channel values."""
        to_migrate = [v for v in values if v["checkpoint"]["v"] < 4 and v["parent_checkpoint_id"]]
        if not to_migrate:
            return
        threads = [*dict.fromkeys(v["thread_id"] for v in to_migrate)]
        parents = [*dict.fromkeys(v["parent_checkpoint_id"] for v in to_migrate)]
        found = yield Statement(self._tables.select_pending_sends, (threads, parents))
        grouped_by_parent: defaultdict[tuple[str, str], RowList] = defaultdict(list)
        for value in to_migrate:
            grouped_by_parent[(value["thread_id"], value["parent_checkpoint_id"])].append(value)
        for thread_id, checkpoint_id, sends in found.records:
            for value in grouped_by_parent[(thread_id, checkpoint_id)]:
                if value["channel_values"] is None:
                    value["channel_values"] = []
                self._migrate_pending_sends(sends, value["checkpoint"], value["channel_values"])

    def _migrate_pending_sends(
        self,
        pending_sends: Sequence[Sequence[bytes]],
        checkpoint: dict[str, Any],
        channel_values: Any,
    ) -> None:
        """Add the sends to a checkpoint's blobs as one value of the tasks channel, at the
        checkpoint's latest version."""
        if not pending_sends:
            return
        enc, blob = self.serde.dumps_typed(
            [self.serde.loads_typed((c.decode(), b)) for c, b in pending_sends],
        )
        channel_values.append((TASKS.encode(), enc.encode(), blob))
        checkpoint["channel_versions"][TASKS] = (
            max(checkpoint["channel_versions"].values())
            if checkpoint["channel_versions"]
            else self.get_next_version(None, None)
        )

    def _load_checkpoint_tuple(self, value: Row) -> CheckpointTuple:
        checkpoint = cast(
            Checkpoint,
            {
                **value["checkpoint"],
                "channel_values": {
                    **(value["checkpoint"].get("channel_values") or {}),
                    **self._load_blobs(value["channel_values"]),
                },
            },
        )
        parent: RunnableConfig | None = (
            {
                "configurable": {
                    "thread_id": value["thread_id"],
                    "checkpoint_ns": value["checkpoint_ns"],
                    "checkpoint_id": value["parent_checkpoint_id"],
                }
            }
            if value["parent_checkpoint_id"]
            else None
        )
        return CheckpointTuple(
            {
                "configurable": {
                    "thread_id": value["thread_id"],
                    "checkpoint_ns": value["checkpoint_ns"],
                    "checkpoint_id": value["checkpoint_id"],
                }
            },
            checkpoint,
            value["metadata"],
            parent,
            self._load_writes(value["pending_writes"]),
        )

    def _load_blobs(self, blob_values: Sequence[Sequence[bytes]] | None) -> Row:
        if not blob_values:
            return {}
        return {
            k.decode(): self.serde.loads_typed((t.decode(), v))
            for k, t, v in blob_values
            if t.decode() != "empty"
        }

    def _load_writes(self, writes: Sequence[Sequence[bytes]] | None) -> WriteList:
        return [
            (tid.decode(), channel.decode(), self.serde.loads_typed((t.decode(), v)))
            for tid, channel, t, v, _, _ in sorted(
                writes or [],
                key=lambda w: writes_sort_key(w[4].decode(), w[0].decode(), int(w[5])),
            )
        ]

    # -- writing -------------------------------------------------------------------------------

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Save a checkpoint under the thread and namespace of ``config``, whose checkpoint id,
        if it names one, is the new checkpoint's parent.

        A channel value that is ``None``, a boolean, an integer, a finite float, or a string
        ``jsonb`` can hold is kept inline in the checkpoint. Any other is serialized into a blob
        when ``new_versions`` gives its channel a new version, and read back by that version
        later.
        """
        self._ensure()
        statements, target, parent = self._put(config, checkpoint, metadata, new_versions)
        synced = self._engine.run(
            lambda: self._saved(statements, target, parent, new_versions), replay=True
        )
        self._remember(target, synced)
        return target.config

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        await self._aensure()
        statements, target, parent = self._put(config, checkpoint, metadata, new_versions)
        synced = await self._engine.arun(
            lambda: self._saved(statements, target, parent, new_versions), replay=True
        )
        self._remember(target, synced)
        return target.config

    def _put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> tuple[tuple[Statement, ...], _Target, str | None]:
        """A put's statements, the new blobs and then the checkpoint; the checkpoint as a target
        for the declared channels; and its parent's id."""
        configurable = config["configurable"].copy()
        thread_id = configurable.pop("thread_id")
        checkpoint_ns = configurable.pop("checkpoint_ns")
        checkpoint_id = configurable.pop("checkpoint_id", None)
        thread = str(thread_id)
        copy = cast(dict[str, Any], checkpoint.copy())
        copy["channel_values"] = copy["channel_values"].copy()
        next_config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
            }
        }

        # A value `jsonb` holds stays inline in the checkpoint; the rest are blobs. A
        # `_DeltaSnapshot` is a blob with a `True` marker inline.
        blob_values = {}
        for k, v in checkpoint["channel_values"].items():
            if isinstance(v, _DeltaSnapshot):
                blob_values[k] = copy["channel_values"].pop(k)
                copy["channel_values"][k] = True
            elif not _inline(v):
                blob_values[k] = copy["channel_values"].pop(k)

        statements = []
        if blob_versions := {k: v for k, v in new_versions.items() if k in blob_values}:
            rows = self._dump_blobs(thread, checkpoint_ns, blob_values, blob_versions)
            statements.append(
                Statement(
                    self._tables.upsert_blobs,
                    (thread, checkpoint_ns, *_columns(rows, 2, 3, 4, 5)),
                )
            )
        statements.append(
            Statement(
                self._tables.upsert_checkpoint,
                (
                    thread,
                    checkpoint_ns,
                    checkpoint["id"],
                    checkpoint_id,
                    Jsonb(copy),
                    Jsonb(get_serializable_checkpoint_metadata(config, metadata)),
                ),
            )
        )
        target = _Target(
            next_config, thread, checkpoint_ns, checkpoint["id"], checkpoint["channel_values"]
        )
        return tuple(statements), target, checkpoint_id

    def _dump_blobs(
        self,
        thread_id: str,
        checkpoint_ns: str,
        values: Row,
        versions: ChannelVersions,
    ) -> Sequence[BlobRow]:
        return [
            (
                thread_id,
                checkpoint_ns,
                k,
                str(ver),
                *(self.serde.dumps_typed(values[k]) if k in values else ("empty", None)),
            )
            for k, ver in versions.items()
        ]

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Save the writes a task made against the checkpoint ``config`` names.

        A write to a channel of ``WRITES_IDX_MAP`` is stored under that channel's fixed index,
        any other under its position in ``writes``. When every write is to such a channel, one
        already stored under an index is replaced; otherwise it is kept.
        """
        if not writes:
            return
        self._ensure()
        statement = self._put_writes(config, writes, task_id, task_path)
        self._engine.run(lambda: _send((statement,)), replay=True)

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        if not writes:
            return
        await self._aensure()
        statement = self._put_writes(config, writes, task_id, task_path)
        await self._engine.arun(lambda: _send((statement,)), replay=True)

    def _put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str,
    ) -> Statement:
        """One statement for every write, each index once: the last given where writes replace,
        the first where they are kept, as one statement per write would leave them."""
        upsert = all(w[0] in WRITES_IDX_MAP for w in writes)
        rows = self._dump_writes(
            str(config["configurable"]["thread_id"]),
            config["configurable"]["checkpoint_ns"],
            config["configurable"]["checkpoint_id"],
            task_id,
            task_path,
            writes,
        )
        by_index: dict[int, WriteRow] = {}
        for row in rows:
            if upsert or row[5] not in by_index:
                by_index[row[5]] = row
        return Statement(
            self._tables.upsert_writes if upsert else self._tables.insert_writes,
            (*rows[0][:5], *_columns([*by_index.values()], 5, 6, 7, 8)),
        )

    def _dump_writes(
        self,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
        task_id: str,
        task_path: str,
        writes: Sequence[tuple[str, Any]],
    ) -> Sequence[WriteRow]:
        return [
            (
                thread_id,
                checkpoint_ns,
                checkpoint_id,
                task_id,
                task_path,
                WRITES_IDX_MAP.get(channel, idx),
                channel,
                *self.serde.dumps_typed(value),
            )
            for idx, (channel, value) in enumerate(writes)
        ]

    def get_next_version(self, current: str | None, channel: None) -> str:
        """A version greater than ``current``: its counter plus one, then a random fraction."""
        if current is None:
            current_v = 0
        elif isinstance(current, int):
            current_v = current
        else:
            current_v = int(current.split(".")[0])
        next_v = current_v + 1
        next_h = random.random()
        return f"{next_v:032}.{next_h:016}"

    # -- queryable channels --------------------------------------------------------------------

    def cypher_subquery(self, channel: str, thread: str = "%s", namespace: str = "''") -> str:
        """A SQL subquery a Cypher statement can hold, giving a declared channel's items, as of
        the thread's latest checkpoint, as a list in position order.

        For a ``"messages"`` channel each element is a map of ``position``, ``id``, ``type``
        (``human``, ``ai``, …), ``text`` (always a string), ``content`` (a string or a list of
        blocks) and ``message``, the whole ``message_to_dict`` form; for the other modes it is the
        item itself, and a ``"last_value"`` channel's list holds the one value. ``thread`` and
        ``namespace`` are the SQL for the thread id and the namespace: a placeholder and the root
        namespace, by default, or expressions. For example::

            graph.query(
                f"UNWIND [1] AS a WITH {saver.cypher_subquery('messages')} AS messages "
                'UNWIND messages AS m MATCH (p:"Person") WHERE m.text CONTAINS p.name '
                "RETURN p.name AS who, m.position AS position",
                (thread_id,),
            )
        """
        mode = self.queryable_channels.get(channel)
        if mode is None:
            raise ValueError(f"{channel!r} is not one of the queryable channels")
        element = (
            "jsonb_build_object('position', i.position, 'id', i.item_key, "
            f"{message_fields('i.value')})"
            if mode == "messages"
            else "i.value"
        )
        return self._tables.subquery(channel, element, thread, namespace)

    def _saved(
        self,
        statements: tuple[Statement, ...],
        target: _Target,
        parent: str | None,
        new_versions: ChannelVersions,
    ) -> Program[bool]:
        """A put's statements, and with queryable channels the items of the declared channels;
        whether the pointer is at the checkpoint afterwards.

        When this saver last left the pointer at the parent, the pipeline moves it on only from
        there and writes only the channels with a new version. Otherwise, or when the pointer had
        moved, it is claimed, and every declared channel is written.
        """
        if not self.queryable_channels:
            yield from _send(statements)
            return False
        if parent is not None and self._synced_at(target) == parent:
            channels = [ch for ch in self.queryable_channels if ch in new_versions]
            moved = Statement(
                self._tables.advance_pointer,
                (target.checkpoint_id, target.thread, target.ns, parent),
            )
            done = yield Pipeline((*statements, moved, *self._syncs(target, channels)))
            if done[len(statements)].records:
                yield from self._sync_rebuilt(target, channels)
                return True
            statements = ()
        return (yield from self._claim(target, statements))

    def _claim(self, target: _Target, before: tuple[Statement, ...] = ()) -> Program[bool]:
        """Point the thread at ``target`` unless it points at a later checkpoint, and write every
        declared channel; whether it points at ``target``."""
        channels = [*self.queryable_channels]
        claim = Statement(
            self._tables.claim_pointer, (target.thread, target.ns, target.checkpoint_id)
        )
        done = yield Pipeline((*before, claim, *self._syncs(target, channels)))
        if not done[len(before)].records:
            return False
        yield from self._sync_rebuilt(target, channels)
        return True

    def _syncs(self, target: _Target, channels: Iterable[str]) -> tuple[Statement, ...]:
        """The statements writing each channel the checkpoint holds a value of."""
        return tuple(
            self._sync(target, ch, target.values[ch]) for ch in channels if ch in target.values
        )

    def _sync_rebuilt(self, target: _Target, channels: Sequence[str]) -> Program[None]:
        """Write each channel the checkpoint holds no value of, from its history: a
        ``DeltaChannel`` between snapshots, or a channel never written, which has no items."""
        absent = [ch for ch in channels if ch not in target.values]
        if not absent:
            return
        history = yield from self._delta_history(target.config, absent)
        yield from _send(
            tuple(
                self._sync(target, ch, folded(self.queryable_channels[ch], history[ch]))
                for ch in absent
            )
        )

    def _sync(self, target: _Target, channel: str, value: Any) -> Statement:
        found = items(channel, self.queryable_channels[channel], value)
        return Statement(
            self._tables.sync_items,
            {
                "thread": target.thread,
                "ns": target.ns,
                "checkpoint": target.checkpoint_id,
                "channel": channel,
                "keys": [key for key, _, _ in found],
                "positions": [position for _, position, _ in found],
                "values": [Jsonb(value) for _, _, value in found],
            },
        )

    def _synced_at(self, target: _Target) -> str | None:
        """The checkpoint this saver last left the thread's pointer at, if it remembers one."""
        with self._synced_lock:
            return self._synced.get((target.thread, target.ns))

    def _remember(self, target: _Target, synced: bool) -> None:
        """Remember where the thread's pointer is after a write: at ``target`` when ``synced``,
        and otherwise not known."""
        if not self.queryable_channels:
            return
        key = (target.thread, target.ns)
        with self._synced_lock:
            if not synced:
                self._synced.pop(key, None)
                return
            self._synced[key] = target.checkpoint_id
            self._synced.move_to_end(key)
            while len(self._synced) > SYNCED_THREADS:
                self._synced.popitem(last=False)

    # -- deleting and copying ------------------------------------------------------------------

    def delete_thread(self, thread_id: str) -> None:
        """Delete every checkpoint, blob, write and item of the thread, in every namespace."""
        self._ensure()
        self._engine.run(lambda: self._delete_thread(str(thread_id)), replay=True)

    async def adelete_thread(self, thread_id: str) -> None:
        await self._aensure()
        await self._engine.arun(lambda: self._delete_thread(str(thread_id)), replay=True)

    def _delete_thread(self, thread_id: str) -> Program[None]:
        yield Pipeline(
            tuple(Statement(text, ([thread_id],)) for text in self._tables.delete_threads)
        )

    def delete_for_runs(self, run_ids: Sequence[str]) -> None:
        """Delete every checkpoint whose metadata names one of these runs, with its writes and
        the blobs no other checkpoint references.

        A child of a deleted checkpoint is left with no parent. A thread whose items reflected a
        deleted checkpoint has them written again from its newest checkpoint left, or deleted
        when none is left. A run still saving checkpoints should not be deleted: a checkpoint it
        saves meanwhile may reference a blob deleted here. Deleting checkpoints a thread's
        ``DeltaChannel`` is rebuilt from leaves that channel short; see :meth:`prune`.
        """
        keys = _ids(run_ids, "run_ids")
        if not keys:
            return
        self._ensure()
        stale, synced = self._engine.run(lambda: self._delete_for_runs(keys), replay=True)
        self._forget(stale, synced)

    async def adelete_for_runs(self, run_ids: Sequence[str]) -> None:
        keys = _ids(run_ids, "run_ids")
        if not keys:
            return
        await self._aensure()
        stale, synced = await self._engine.arun(lambda: self._delete_for_runs(keys), replay=True)
        self._forget(stale, synced)

    def _delete_for_runs(self, run_ids: Sequence[str]) -> Program[_Deleted]:
        """Delete the runs, and write the items of the threads they leave stale again.

        With declared channels, one transaction: only the delete says which threads to write
        again, so a replay after a lost answer must find the delete and the writing both done, or
        neither. Without them it is the one statement.
        """
        if not self.queryable_channels:
            found = yield Statement(self._tables.delete_for_runs, (run_ids,))
            return [(str(thread), str(ns)) for thread, ns in found.records], []
        done: _Deleted = yield InTransaction(self._deleting_for_runs(run_ids))
        return done

    def _deleting_for_runs(self, run_ids: Sequence[str]) -> Program[_Deleted]:
        found = yield Statement(self._tables.delete_for_runs, (run_ids,))
        stale = [(str(thread), str(ns)) for thread, ns in found.records]
        synced: list[tuple[_Target, bool]] = []
        for thread, ns in stale:
            newest = yield from self._get_tuple(
                {"configurable": {"thread_id": thread, "checkpoint_ns": ns}}
            )
            if newest is None:
                continue
            target = _Target(
                newest.config,
                thread,
                ns,
                newest.config["configurable"]["checkpoint_id"],
                newest.checkpoint["channel_values"],
            )
            synced.append((target, (yield from self._claim(target))))
        return stale, synced

    def _forget(
        self, stale: Sequence[tuple[str, str]], synced: Sequence[tuple[_Target, bool]]
    ) -> None:
        with self._synced_lock:
            for key in stale:
                self._synced.pop(key, None)
        for target, owned in synced:
            self._remember(target, owned)

    def copy_thread(self, source_thread_id: str, target_thread_id: str) -> None:
        """Copy every checkpoint, blob, write and item of one thread to another, in one
        statement.

        The whole parent chain is copied, which a ``DeltaChannel`` needs to be rebuilt on the
        target. A target that already holds a checkpoint is refused with :class:`ValueError`
        rather than mixed into.
        """
        self._ensure()
        self._engine.run(lambda: self._copy_thread(str(source_thread_id), str(target_thread_id)))

    async def acopy_thread(self, source_thread_id: str, target_thread_id: str) -> None:
        await self._aensure()
        await self._engine.arun(
            lambda: self._copy_thread(str(source_thread_id), str(target_thread_id))
        )

    def _copy_thread(self, source: str, target: str) -> Program[None]:
        found = yield Statement(self._tables.copy_thread, {"source": source, "target": target})
        if scalar(found):
            raise ValueError(
                f"thread {target!r} already holds checkpoints, which a copy would be mixed into; "
                f"delete it first with delete_thread({target!r}), or copy to an unused thread id"
            )

    def prune(self, thread_ids: Sequence[str], *, strategy: str = "keep_latest") -> None:
        """Prune checkpoints for the given threads.

        ``"delete"`` removes every checkpoint, write, blob and item of the threads, in one
        transaction.

        ``"keep_latest"`` keeps the latest checkpoint of each namespace (the one ``get_tuple``
        loads, plus its parent when it predates format 4 and has pending sends), with its pending
        writes and the blobs it references, and deletes the rest. A kept checkpoint whose parent
        it deletes is left with no parent. A subgraph call's namespace is deleted with the root
        checkpoint the call started from, items and all, except a ``checkpointer=True``
        subgraph's, which keeps its latest checkpoint. The items of a kept namespace stay. Each
        thread is pruned in its own transaction, on one snapshot of it, so a run appending to the
        thread meanwhile is safe; forking or replaying from a checkpoint being deleted is not. A
        thread that fails because another transaction changed it leaves the threads before it
        pruned, and retrying is safe.

        ``"keep_latest"`` refuses threads that use a ``DeltaChannel``, whose latest value is
        rebuilt from older checkpoints: if any requested thread does, nothing is pruned. A thread
        that starts using one before its turn in the batch is refused then, after the threads
        before it.

        Pruning runs on a maintenance connection. Inside a caller's transaction it joins that
        transaction instead, where each thread's transaction is a savepoint reading the
        caller's snapshot.

        Raises:
            TypeError: If ``thread_ids`` is a single string.
            ValueError: If ``strategy`` is unknown.
            NotImplementedError: If ``strategy`` is ``"keep_latest"`` and a thread uses a
                ``DeltaChannel``.
            psycopg.errors.SerializationFailure: If another transaction, such as a second
                prune, changes a thread's rows while ``"keep_latest"`` prunes it.
            psycopg.errors.DeadlockDetected: If a second prune of a thread runs at the same time.
        """
        keys = _prune_threads(thread_ids, strategy)
        if not keys:
            return
        self._ensure()
        self._engine.run(lambda: self._prune(keys, strategy), kind="maintenance")

    async def aprune(self, thread_ids: Sequence[str], *, strategy: str = "keep_latest") -> None:
        keys = _prune_threads(thread_ids, strategy)
        if not keys:
            return
        await self._aensure()
        await self._engine.arun(lambda: self._prune(keys, strategy), kind="maintenance")

    def _prune(self, thread_ids: Sequence[str], strategy: str) -> Program[None]:
        tables = self._tables
        if strategy == "delete":
            yield Pipeline(tuple(Statement(text, (thread_ids,)) for text in tables.delete_threads))
            return
        _refuse_delta((yield Statement(tables.prune_delta_threads, (thread_ids, thread_ids))))
        for thread_id in thread_ids:
            # Every statement of a thread's transaction reads one snapshot, so what a run saves
            # meanwhile is neither seen nor deleted; a savepoint in a caller's transaction reads
            # the caller's.
            yield InTransaction(self._keep_latest(thread_id), isolation="REPEATABLE READ")

    def _keep_latest(self, thread_id: str) -> Program[None]:
        """One thread's ``keep_latest`` transaction."""
        tables = self._tables
        found = yield Pipeline(
            (
                Statement(tables.prune_heads, (thread_id,) * 4),
                # Catches a thread that started using a DeltaChannel after the batch check.
                Statement(tables.prune_delta_threads, ([thread_id], [thread_id])),
            )
        )
        heads = mappings(found[0])
        _refuse_delta(found[1])
        kept = {h["checkpoint_ns"]: [i for i in (h["head"], h["sends_parent"]) if i] for h in heads}
        # A subgraph call's namespace has its task id after a ":", unlike a `checkpointer=True`
        # subgraph's, whose state outlives each call. A call goes with the root checkpoint it
        # started from, but only once that one is saved: runs save in the background, so a call
        # can commit first.
        calls = sorted(
            h["checkpoint_ns"]
            for h in heads
            if ":" in h["checkpoint_ns"] and h["root_saved"] and h["root"] not in kept.get("", ())
        )
        if calls:
            yield Pipeline(
                tuple(Statement(text, (thread_id, calls)) for text in tables.prune_subgraph_calls)
            )
        for ns, ids in kept.items():
            if ns in calls:
                continue
            unkept = yield Statement(tables.prune_unkept, (thread_id, ns, ids))
            deleted = [row[0] for row in unkept.records]
            # In order, so the blobs still referenced are read after the checkpoints went.
            done = yield Pipeline(
                (
                    Statement(tables.prune_writes, (thread_id, ns, deleted)),
                    Statement(tables.prune_checkpoints, (thread_id, ns, ids)),
                    Statement(tables.prune_parents, (thread_id, ns, ids, ids)),
                    Statement(tables.prune_referenced_blobs, (thread_id, ns)),
                )
            )
            garbage = _garbage_blobs(mappings(done[1]), mappings(done[3]))
            if garbage:
                yield Pipeline(
                    tuple(
                        Statement(tables.prune_blobs, (thread_id, ns, channel, versions))
                        for channel, versions in garbage.items()
                    )
                )

    # -- delta channel history -----------------------------------------------------------------

    def get_delta_channel_history(
        self, *, config: RunnableConfig, channels: Sequence[str]
    ) -> Mapping[str, DeltaChannelHistory]:
        """Each channel's writes along the parent chain of the checkpoint ``config`` names, or of
        the latest when it names none, back to the nearest ancestor storing a value for the
        channel, which is the channel's ``seed``.

        Two stages. The first pages through the thread's checkpoints newest first from the
        target, reading for every channel its version, whether a blob is stored for it and its
        inline value, and walks each channel's chain as far as the pages read allow, until
        every walk has found its seed or ended. The second reads, in one statement, each
        channel's writes along its own chain and each seed's blob.
        """
        if not channels:
            return {}
        self._ensure()
        return self._engine.run(lambda: self._delta_history(config, [*channels]), replay=True)

    async def aget_delta_channel_history(
        self, *, config: RunnableConfig, channels: Sequence[str]
    ) -> Mapping[str, DeltaChannelHistory]:
        if not channels:
            return {}
        await self._aensure()
        return await self._engine.arun(
            lambda: self._delta_history(config, [*channels]), replay=True
        )

    def _delta_history(
        self, config: RunnableConfig, channels: Sequence[str]
    ) -> Program[dict[str, DeltaChannelHistory]]:
        tables = self._tables
        thread_id = str(config["configurable"]["thread_id"])
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = get_checkpoint_id(config)
        if not checkpoint_id:
            checkpoint_id = scalar((yield Statement(tables.latest_id, (thread_id, checkpoint_ns))))
            if checkpoint_id is None:
                return {ch: {"writes": []} for ch in channels}

        # Stage 1: pages of the thread's checkpoints, walking each channel's parent chain after
        # each page; it stops as soon as every walk has found its seed or ended.
        stage1_sql = tables.delta_stage1(channels, include_cursor=True)
        parent_of: dict[str, str | None] = {}
        ver_by_i_by_cid: list[dict[str, str | None]] = [{} for _ in channels]
        hb_by_i_by_cid: list[dict[str, bool]] = [{} for _ in channels]
        inline_by_i_by_cid: list[dict[str, str | None]] = [{} for _ in channels]
        chain_by_ch: dict[str, list[str]] = {ch: [] for ch in channels}
        seed_ver_by_ch: dict[str, str | None] = {ch: None for ch in channels}
        seed_inline_by_ch: dict[str, Any] = {}
        walk_cursor_by_ch: dict[str, str | None] = {}
        seeded: set[str] = set()
        cursor: str | None = checkpoint_id
        while True:
            stage1_params: list[Any] = []
            for ch in channels:
                # ver_i, blob channel, blob version, inline_i
                stage1_params.extend([ch, ch, ch, ch])
            stage1_params.extend([thread_id, checkpoint_ns, cursor, DELTA_PAGE_SIZE])
            page = mappings((yield Statement(stage1_sql, stage1_params)))
            if not page:
                break
            oldest = _ingest_stage1_page(
                page, channels, parent_of, ver_by_i_by_cid, hb_by_i_by_cid, inline_by_i_by_cid
            )
            _try_advance_walks(
                checkpoint_id,
                oldest,
                channels,
                parent_of,
                ver_by_i_by_cid,
                hb_by_i_by_cid,
                inline_by_i_by_cid,
                chain_by_ch,
                seed_ver_by_ch,
                seed_inline_by_ch,
                walk_cursor_by_ch,
                seeded,
            )
            # The target was not found, every walk is over, or the page was short: the chain is
            # exhausted.
            ended = all(
                ch in seeded or (ch in walk_cursor_by_ch and walk_cursor_by_ch[ch] is None)
                for ch in channels
            )
            if checkpoint_id not in parent_of or ended or len(page) < DELTA_PAGE_SIZE:
                break
            cursor = oldest
            stage1_sql = tables.delta_stage1(channels)

        # Stage 2: a writes branch per channel with a chain, a blob branch per seeded channel.
        channels_with_chain = [ch for ch in channels if chain_by_ch[ch]]
        channels_with_seed = [ch for ch in channels if seed_ver_by_ch[ch] is not None]
        stage2_rows: list[dict[str, Any]] = []
        if channels_with_chain or channels_with_seed:
            stage2_params: list[Any] = []
            for ch in channels_with_chain:
                stage2_params.extend([thread_id, checkpoint_ns, ch, chain_by_ch[ch]])
            for ch in channels_with_seed:
                stage2_params.extend([thread_id, checkpoint_ns, ch, seed_ver_by_ch[ch]])
            stage2_sql = tables.delta_stage2(
                with_chain=len(channels_with_chain), with_seed=len(channels_with_seed)
            )
            stage2_rows = mappings((yield Statement(stage2_sql, stage2_params)))
        return self._build_delta_channels_writes_history(
            channels=channels,
            chain_by_ch=chain_by_ch,
            seed_ver_by_ch=seed_ver_by_ch,
            seed_inline_by_ch=seed_inline_by_ch,
            stage2_rows=stage2_rows,
        )

    def _build_delta_channels_writes_history(
        self,
        *,
        channels: Sequence[str],
        chain_by_ch: Mapping[str, Sequence[str]],
        seed_ver_by_ch: Mapping[str, str | None],
        seed_inline_by_ch: Mapping[str, Any],
        stage2_rows: Sequence[Row],
    ) -> dict[str, DeltaChannelHistory]:
        """Each channel's history from the second stage's rows.

        A seed comes from the blobs when the walk found one there, and otherwise from
        ``seed_inline_by_ch``: an inline value has no blob, so the second stage returns no row for
        it. ``seed`` is left out when the walk ended without finding a stored value, or when the
        seed blob is ``empty``; either way the consumer starts the channel empty.
        """
        # writes_by_ch_by_cid[channel][cid]: (type, blob, task_id, idx, task_path) of each write
        writes_by_ch_by_cid: dict[str, dict[str, list[tuple[str, bytes, str, int, str]]]] = {
            ch: {} for ch in channels
        }
        # seed_blob_by_ver[(channel, version)]: (type, blob)
        seed_blob_by_ver: dict[tuple[str, str], tuple[str, bytes]] = {}
        for r in stage2_rows:
            ch = r["channel"]
            if r["_kind"] == "w":
                writes_by_ch_by_cid.setdefault(ch, {}).setdefault(r["checkpoint_id"], []).append(
                    (r["type"], r["blob"], r["task_id"], r["idx"], r["task_path"])
                )
            else:
                seed_blob_by_ver[(ch, r["version"])] = (r["type"], r["blob"])

        # Each checkpoint's writes newest first, the order the walk collects them in.
        for cid_map in writes_by_ch_by_cid.values():
            for ws in cid_map.values():
                ws.sort(key=lambda w: writes_sort_key(w[4], w[2], w[3]), reverse=True)

        result: dict[str, DeltaChannelHistory] = {}
        for ch in channels:
            seed_version = seed_ver_by_ch.get(ch)
            collected: list[PendingWrite] = []
            cid_writes = writes_by_ch_by_cid.get(ch, {})
            for cid in chain_by_ch.get(ch, []):
                for type_tag, write_blob, task_id, _idx, _path in cid_writes.get(cid, []):
                    collected.append((task_id, ch, self.serde.loads_typed((type_tag, write_blob))))
            collected.reverse()
            entry: DeltaChannelHistory = {"writes": collected}
            if seed_version is not None or ch in seed_inline_by_ch:
                blob = seed_blob_by_ver.get((ch, seed_version)) if seed_version else None
                if blob is not None and blob[0] != "empty":
                    entry["seed"] = self.serde.loads_typed(blob)
                elif ch in seed_inline_by_ch:
                    # An inline value: stored in the checkpoint, so the second stage has no row.
                    entry["seed"] = seed_inline_by_ch[ch]
            result[ch] = entry
        return result


def _send(statements: tuple[Statement, ...]) -> Program[None]:
    """The statements in one round trip, as a pipeline when there are several."""
    yield statements[0] if len(statements) == 1 else Pipeline(statements)


def _refuse_delta(found: Result | Rows) -> None:
    """Refuse ``keep_latest`` for the threads ``prune_delta_threads`` found."""
    if delta := sorted(row[0] for row in found.records):
        raise NotImplementedError(
            f"keep_latest can't prune threads that use a DeltaChannel: {', '.join(delta)}"
        )


def _checkpoint_rows(result: Result) -> RowList:
    """Checkpoint rows with their checkpoint and metadata decoded from the JSON text read.

    Decoded here rather than by the driver, whose reading of numbers can be set process-wide to
    give decimals, so a float kept inline comes back a float.
    """
    rows = mappings(result)
    for row in rows:
        row["checkpoint"] = json.loads(row["checkpoint"])
        row["metadata"] = json.loads(row["metadata"])
    return rows


def _columns(rows: Sequence[tuple[Any, ...]], *positions: int) -> tuple[list[Any], ...]:
    """The values at each position across the rows, one list per position."""
    return tuple([row[position] for row in rows] for position in positions)


def _inline(value: Any) -> bool:
    """Whether a channel value is kept in the checkpoint's ``jsonb``: ``None``, a boolean, an
    integer, a finite float, or a string without null characters that is valid Unicode."""
    if value is None or isinstance(value, int):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, str):
        return "\x00" not in value and (value.isascii() or _valid_unicode(value))
    return False


def _valid_unicode(text: str) -> bool:
    try:
        text.encode()
    except UnicodeEncodeError:
        return False
    return True


def _without_nul(metadata: Mapping[str, Any]) -> Row:
    """The mapping with null characters taken out of its string values, as
    ``get_checkpoint_metadata`` takes them out of stored metadata."""
    return {k: v.replace("\u0000", "") if isinstance(v, str) else v for k, v in metadata.items()}


def _encrypts(serde: Any) -> bool:
    """Whether a serializer, or one it wraps, encrypts."""
    for _ in range(8):
        if serde is None:
            return False
        if isinstance(serde, EncryptedSerializer):
            return True
        serde = getattr(serde, "serde", None)
    return False


def _ids(ids: Sequence[str], what: str) -> list[str]:
    if isinstance(ids, str):
        raise TypeError(f"{what} must be a sequence of ids, not a str")
    return [str(i) for i in ids]


def _prune_threads(thread_ids: Sequence[str], strategy: str) -> list[str]:
    if isinstance(thread_ids, str):
        raise TypeError("thread_ids must be a sequence of thread IDs, not a str")
    if strategy not in PRUNE_STRATEGIES:
        raise ValueError(f"Unknown prune strategy: {strategy!r}")
    return [str(t) for t in thread_ids]


def _garbage_blobs(freed: Sequence[Row], referenced: Sequence[Row]) -> dict[str, list[str]]:
    """Blobs a deleted checkpoint used and no kept one does, by channel."""
    kept = {(r["channel"], r["version"]) for r in referenced}
    garbage: dict[str, list[str]] = {}
    for r in freed:
        if (r["channel"], r["version"]) not in kept:
            garbage.setdefault(r["channel"], []).append(r["version"])
    return garbage


def _ingest_stage1_page(
    stage1_rows: Sequence[Row],
    channels: Sequence[str],
    parent_of: dict[str, str | None],
    ver_by_i_by_cid: Sequence[dict[str, str | None]],
    hb_by_i_by_cid: Sequence[dict[str, bool]],
    inline_by_i_by_cid: Sequence[dict[str, str | None]],
) -> str | None:
    """Fold one page of the first stage into the walk's mappings; the oldest checkpoint id on the
    page, below which the next page continues."""
    oldest: str | None = None
    for r in stage1_rows:
        cid = r["checkpoint_id"]
        parent_of[cid] = r["parent_checkpoint_id"]
        for i in range(len(channels)):
            ver_by_i_by_cid[i][cid] = r.get(f"ver_{i}")
            hb_by_i_by_cid[i][cid] = bool(r.get(f"hb_{i}"))
            inline_by_i_by_cid[i][cid] = r.get(f"inline_{i}")
        # Pages come newest first, so the last row has the smallest id.
        oldest = cid
    return oldest


def _try_advance_walks(
    target_id: str,
    oldest: str | None,
    channels: Sequence[str],
    parent_of: Mapping[str, str | None],
    ver_by_i_by_cid: Sequence[Mapping[str, str | None]],
    hb_by_i_by_cid: Sequence[Mapping[str, bool]],
    inline_by_i_by_cid: Sequence[Mapping[str, str | None]],
    chain_by_ch: dict[str, list[str]],
    seed_ver_by_ch: dict[str, str | None],
    seed_inline_by_ch: dict[str, Any],
    walk_cursor_by_ch: dict[str, str | None],
    seeded: set[str],
) -> None:
    """Advance each channel's walk as far as the pages read so far allow.

    A walk stops when it finds a stored value for its channel, a blob or an inline value, ``None``
    among them, and the channel is seeded; when it reaches a root, whose parent is ``None``; when
    its next ancestor is missing, which a parent at or above ``oldest``, the oldest checkpoint id
    read, and not read is; when that ancestor is not read yet, to go on after a later page; or
    before it starts, when the target was not found. Ids are compared as the server orders them,
    which for the time-ordered ids LangGraph makes is as Python does. Updates the last five
    arguments in place.
    """
    for i, ch in enumerate(channels):
        if ch in seeded:
            continue
        # The first page opens with the target's row, so it is missing only when the target does
        # not exist; a `None` cursor would read as "the target is a root".
        if ch not in walk_cursor_by_ch:
            if target_id not in parent_of:
                continue
            walk_cursor_by_ch[ch] = parent_of[target_id]
        cur_cid = walk_cursor_by_ch[ch]
        ch_chain = chain_by_ch[ch]
        hb_i = hb_by_i_by_cid[i]
        inline_i = inline_by_i_by_cid[i]
        ver_i = ver_by_i_by_cid[i]
        while cur_cid is not None:
            if cur_cid not in parent_of:
                if oldest is not None and cur_cid >= oldest:
                    # Every checkpoint from the target down to `oldest` is read, so the parent
                    # does not exist and the walk ends here.
                    cur_cid = None
                break
            ch_chain.append(cur_cid)
            has_blob = hb_i.get(cur_cid, False)
            inline = inline_i.get(cur_cid)
            if has_blob or inline is not None:
                # A blob wins: for a `_DeltaSnapshot` the inline reading is the `true` marker,
                # not the value.
                seed_ver_by_ch[ch] = ver_i.get(cur_cid)
                if not has_blob and inline is not None:
                    seed_inline_by_ch[ch] = json.loads(inline)
                seeded.add(ch)
                cur_cid = None
                break
            cur_cid = parent_of[cur_cid]
        walk_cursor_by_ch[ch] = cur_cid


__all__ = ["AgensSaver"]
