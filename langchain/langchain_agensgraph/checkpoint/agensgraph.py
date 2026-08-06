"""LangGraph checkpoint saver backed by AgensGraph.

Persists LangGraph checkpoints as graph vertices so an agent's conversation
state survives process restarts and can be resumed by ``thread_id``:

* ``(:Checkpoint {thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id,
   checkpoint_type, checkpoint, metadata_type, metadata})``
* ``(:CheckpointBlob {thread_id, checkpoint_ns, channel, version, type, blob})``
  — channel values, shared across checkpoints by ``version``.
* ``(:CheckpointWrite {thread_id, checkpoint_ns, checkpoint_id, task_id, idx,
   channel, type, value, task_path})`` — pending writes.

Serialized payloads (which are raw bytes from the serializer) are base64-encoded
because AgensGraph stores all properties as ``jsonb``, which cannot hold bytes.

The implementation mirrors the storage contract of LangGraph's reference
``InMemorySaver``. ``AgensSaver`` exposes both sync and async methods;
``AsyncAgensSaver`` is an alias provided for naming convenience.
"""

from __future__ import annotations

import base64
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Sequence, Tuple

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
    get_checkpoint_metadata,
)
from psycopg import sql
from psycopg.types.json import Jsonb

from langchain_agensgraph.graphs.agensgraph import AgensGraph

_CHECKPOINT_LABEL = "Checkpoint"
_BLOB_LABEL = "CheckpointBlob"
_WRITE_LABEL = "CheckpointWrite"


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _unb64(data: str) -> bytes:
    return base64.b64decode(data.encode("ascii"))


class AgensSaver(BaseCheckpointSaver):
    """Store and retrieve LangGraph checkpoints in an AgensGraph graph."""

    def __init__(
        self,
        graph: Optional[AgensGraph] = None,
        *,
        conf: Optional[Dict[str, Any]] = None,
        graph_name: str = "checkpoints",
        serde: Any = None,
    ) -> None:
        super().__init__(serde=serde)
        if graph is None:
            if conf is None:
                raise ValueError("AgensSaver requires either `graph` or `conf`.")
            graph = AgensGraph(graph_name, conf, create=True)
        self._graph = graph
        for label in (_CHECKPOINT_LABEL, _BLOB_LABEL, _WRITE_LABEL):
            self._graph.query(
                sql.SQL("CREATE VLABEL IF NOT EXISTS {l}").format(
                    l=sql.Identifier(label)
                )
            )
        # All reads filter by thread (and namespace); without these composite
        # indexes every get/list/put is a seq scan over the whole label.
        for name, label, props in (
            (
                f"{_CHECKPOINT_LABEL}_thread_idx",
                _CHECKPOINT_LABEL,
                ("thread_id", "checkpoint_ns", "checkpoint_id"),
            ),
            (f"{_BLOB_LABEL}_thread_idx", _BLOB_LABEL, ("thread_id", "checkpoint_ns")),
            (
                f"{_WRITE_LABEL}_thread_idx",
                _WRITE_LABEL,
                ("thread_id", "checkpoint_ns", "checkpoint_id"),
            ),
            # delete_for_runs selects by run, which no thread index covers.
            (f"{_CHECKPOINT_LABEL}_run_idx", _CHECKPOINT_LABEL, ("run_id",)),
        ):
            cols = sql.SQL(", ").join(sql.Identifier(p) for p in props)
            self._graph.query(
                sql.SQL(
                    "CREATE PROPERTY INDEX IF NOT EXISTS {name} ON {l} ({cols})"
                ).format(
                    name=sql.Identifier(name),
                    l=sql.Identifier(label),
                    cols=cols,
                )
            )

    # ---- key extraction ----

    @staticmethod
    def _keys(config: RunnableConfig) -> Tuple[str, str, Optional[str]]:
        cfg = config["configurable"]
        return (
            cfg["thread_id"],
            cfg.get("checkpoint_ns", ""),
            get_checkpoint_id(config),
        )

    # ---- serialization helpers ----

    def _dump(self, obj: Any) -> Tuple[str, str]:
        type_, payload = self.serde.dumps_typed(obj)
        return type_, _b64(payload)

    def _load(self, type_: str, b64: str) -> Any:
        return self.serde.loads_typed((type_, _unb64(b64)))

    def _checkpoint_props(
        self,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        parent_checkpoint_id: Optional[str],
        config: RunnableConfig,
    ) -> Tuple[Dict[str, Any], Dict[str, Tuple[str, str]]]:
        """Return (checkpoint node properties, {channel: (type, b64)} blobs)."""
        c = dict(checkpoint)
        channel_values: Dict[str, Any] = c.pop("channel_values", {})  # type: ignore
        ck_type, ck_b64 = self._dump(c)
        resolved_metadata = get_checkpoint_metadata(config, metadata)
        md_type, md_b64 = self._dump(resolved_metadata)
        props = {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            "checkpoint_id": checkpoint["id"],
            "parent_checkpoint_id": parent_checkpoint_id,
            "checkpoint_type": ck_type,
            "checkpoint": ck_b64,
            "metadata_type": md_type,
            "metadata": md_b64,
            # The serialized metadata is opaque to a query, so the run this checkpoint
            # belongs to is kept as a property of its own for delete_for_runs.
            "run_id": resolved_metadata.get("run_id"),
        }
        blobs = {k: self._dump(v) for k, v in channel_values.items()}
        return props, blobs

    # ---- Cypher builders ----

    def _put_checkpoint_cypher(self):
        return sql.SQL(
            "MERGE (c:{cl} {{thread_id: %(tid)s, checkpoint_ns: %(ns)s, "
            "checkpoint_id: %(cid)s}}) SET c = %(props)s"
        ).format(cl=sql.Identifier(_CHECKPOINT_LABEL))

    def _put_blobs_cypher(self):
        return sql.SQL(
            "UNWIND %(blobs)s AS b "
            "MERGE (x:{bl} {{thread_id: b.thread_id, checkpoint_ns: b.checkpoint_ns, "
            "channel: b.channel, version: b.version}}) SET x = b"
        ).format(bl=sql.Identifier(_BLOB_LABEL))

    def _put_writes_cypher(self):
        return sql.SQL(
            "UNWIND %(writes)s AS w "
            "MERGE (x:{wl} {{thread_id: w.thread_id, checkpoint_ns: w.checkpoint_ns, "
            "checkpoint_id: w.checkpoint_id, task_id: w.task_id, idx: w.idx}}) "
            "SET x = w"
        ).format(wl=sql.Identifier(_WRITE_LABEL))

    def _select_checkpoint_cypher(self, by_id: bool, before: bool, limit: Optional[int]):
        where = "WHERE c.thread_id = %(tid)s AND c.checkpoint_ns = %(ns)s"
        if by_id:
            where += " AND c.checkpoint_id = %(cid)s"
        if before:
            where += " AND c.checkpoint_id < %(before)s"
        tail = " ORDER BY c.checkpoint_id DESC"
        if limit is not None:
            tail += f" LIMIT {int(limit)}"
        return sql.SQL(
            "MATCH (c:{cl}) " + where + " "
            "RETURN c.thread_id AS thread_id, c.checkpoint_ns AS checkpoint_ns, "
            "c.checkpoint_id AS checkpoint_id, c.parent_checkpoint_id AS parent_checkpoint_id, "
            "c.checkpoint_type AS checkpoint_type, c.checkpoint AS checkpoint, "
            "c.metadata_type AS metadata_type, c.metadata AS metadata" + tail
        ).format(cl=sql.Identifier(_CHECKPOINT_LABEL))

    def _select_blobs_cypher(self):
        return sql.SQL(
            "MATCH (x:{bl}) WHERE x.thread_id = %(tid)s AND x.checkpoint_ns = %(ns)s "
            "RETURN x.channel AS channel, x.version AS version, x.type AS type, "
            "x.blob AS blob"
        ).format(bl=sql.Identifier(_BLOB_LABEL))

    def _select_writes_cypher(self):
        return sql.SQL(
            "MATCH (x:{wl}) WHERE x.thread_id = %(tid)s AND x.checkpoint_ns = %(ns)s "
            "AND x.checkpoint_id = %(cid)s "
            "RETURN x.task_id AS task_id, x.idx AS idx, x.channel AS channel, "
            "x.type AS type, x.value AS value ORDER BY x.idx"
        ).format(wl=sql.Identifier(_WRITE_LABEL))

    def _select_all_writes_cypher(self):
        # All writes for a thread/ns in one shot (avoids an N+1 per checkpoint
        # in ``list``); grouped by checkpoint_id in Python.
        return sql.SQL(
            "MATCH (x:{wl}) WHERE x.thread_id = %(tid)s AND x.checkpoint_ns = %(ns)s "
            "RETURN x.checkpoint_id AS checkpoint_id, x.task_id AS task_id, "
            "x.idx AS idx, x.channel AS channel, x.type AS type, x.value AS value "
            "ORDER BY x.idx"
        ).format(wl=sql.Identifier(_WRITE_LABEL))

    @staticmethod
    def _group_writes(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for w in rows:
            grouped.setdefault(w["checkpoint_id"], []).append(w)
        return grouped

    def _delete_label_cypher(self, label: str):
        return sql.SQL(
            "MATCH (n:{l}) WHERE n.thread_id = %(tid)s DETACH DELETE n"
        ).format(l=sql.Identifier(label))

    @staticmethod
    def _value_predicate(prop: str, values: Sequence[str], params: Dict[str, Any]):
        """An OR of equalities over one property.

        Each term is an equality the property's index can answer; a bound list would
        instead be tested for containment and read every row of the label.
        """
        terms = []
        for i, value in enumerate(values):
            params[f"v{i}"] = Jsonb(value)
            terms.append(
                sql.SQL("n.{p} = %({v})s").format(
                    p=sql.Identifier(prop), v=sql.SQL(f"v{i}")
                )
            )
        return sql.SQL(" OR ").join(terms)

    def _select_by_run_cypher(self, predicate):
        return sql.SQL(
            "MATCH (n:{l}) WHERE {pred} "
            "RETURN n.thread_id AS thread_id, n.checkpoint_ns AS checkpoint_ns, "
            "       n.checkpoint_id AS checkpoint_id"
        ).format(l=sql.Identifier(_CHECKPOINT_LABEL), pred=predicate)

    def _select_by_thread_cypher(self, predicate):
        """Checkpoints with the parent link and payload a delta walk needs."""
        return sql.SQL(
            "MATCH (n:{l}) WHERE {pred} "
            "RETURN n.thread_id AS thread_id, n.checkpoint_ns AS checkpoint_ns, "
            "       n.checkpoint_id AS checkpoint_id, "
            "       n.parent_checkpoint_id AS parent_checkpoint_id, "
            "       n.checkpoint_type AS checkpoint_type, n.checkpoint AS checkpoint"
        ).format(l=sql.Identifier(_CHECKPOINT_LABEL), pred=predicate)

    def _select_writes_window_cypher(self):
        """Pending writes for a contiguous span of checkpoints."""
        return sql.SQL(
            "MATCH (x:{wl}) WHERE x.thread_id = %(tid)s AND x.checkpoint_ns = %(ns)s "
            "AND x.checkpoint_id >= %(lo)s AND x.checkpoint_id <= %(hi)s "
            "RETURN x.checkpoint_id AS checkpoint_id, x.task_id AS task_id, "
            "x.idx AS idx, x.channel AS channel, x.type AS type, x.value AS value "
            "ORDER BY x.idx"
        ).format(wl=sql.Identifier(_WRITE_LABEL))

    def _select_ancestors_cypher(self, limit: int):
        """A span of a thread's checkpoints at or below an id, newest first."""
        return sql.SQL(
            "MATCH (c:{cl}) WHERE c.thread_id = %(tid)s AND c.checkpoint_ns = %(ns)s "
            "AND c.checkpoint_id <= %(hi)s "
            "RETURN c.checkpoint_id AS checkpoint_id, "
            "       c.parent_checkpoint_id AS parent_checkpoint_id, "
            "       c.checkpoint_type AS checkpoint_type, c.checkpoint AS checkpoint "
            "ORDER BY c.checkpoint_id DESC LIMIT {n}"
        ).format(cl=sql.Identifier(_CHECKPOINT_LABEL), n=sql.SQL(str(int(limit))))

    def _select_thread_blobs_cypher(self, predicate):
        """Which channel values are stored, across every namespace of a thread."""
        return sql.SQL(
            "MATCH (n:{l}) WHERE {pred} "
            "RETURN n.thread_id AS thread_id, n.checkpoint_ns AS checkpoint_ns, "
            "       n.channel AS channel, n.version AS version, n.type AS type"
        ).format(l=sql.Identifier(_BLOB_LABEL), pred=predicate)

    def _copy_thread_cypher(self, label: str):
        """Copy every row of a label from one thread to another.

        Copying the whole thread carries the complete parent chain, which is what a
        resumed thread needs to rebuild its state.
        """
        return sql.SQL(
            "MATCH (c:{l}) WHERE c.thread_id = %(src)s "
            "CREATE (n:{l}) SET n = properties(c), n.thread_id = %(tgt)s"
        ).format(l=sql.Identifier(label))

    def _delete_checkpoints_cypher(self, label: str, predicate):
        return sql.SQL("MATCH (n:{l}) WHERE {pred} DETACH DELETE n").format(
            l=sql.Identifier(label), pred=predicate
        )

    @staticmethod
    def _triple_predicate(rows: List[Dict[str, Any]], params: Dict[str, Any]):
        """Match specific (thread, namespace, checkpoint) rows."""
        terms = []
        for i, row in enumerate(rows):
            params[f"t{i}"] = Jsonb(row["thread_id"])
            params[f"n{i}"] = Jsonb(row["checkpoint_ns"])
            params[f"c{i}"] = Jsonb(row["checkpoint_id"])
            terms.append(
                sql.SQL(
                    "(n.thread_id = %({t})s AND n.checkpoint_ns = %({n})s "
                    "AND n.checkpoint_id = %({c})s)"
                ).format(
                    t=sql.SQL(f"t{i}"), n=sql.SQL(f"n{i}"), c=sql.SQL(f"c{i}")
                )
            )
        return sql.SQL(" OR ").join(terms)

    def _stored_channels(self, row: Dict[str, Any], stored: set) -> Tuple[set, set]:
        """Split a checkpoint's channels into those it stores and those it does not.

        A channel is stored at a checkpoint when a non-empty blob exists for the version
        that checkpoint records. A channel that is versioned but not stored is carried
        by the writes of this checkpoint and its ancestors instead — which is how a
        delta channel is held between snapshots.
        """
        checkpoint = self._load(row["checkpoint_type"], row["checkpoint"])
        versions = checkpoint.get("channel_versions", {}) or {}
        group = (row["thread_id"], row["checkpoint_ns"])
        held, carried = set(), set()
        for channel, version in versions.items():
            if (group, channel, str(version)) in stored:
                held.add(channel)
            else:
                carried.add(channel)
        return held, carried

    def _superseded(
        self, rows: List[Dict[str, Any]], stored: set
    ) -> List[Dict[str, Any]]:
        """Checkpoints that may be dropped, keeping each group's newest and its chain.

        Checkpoint ids sort in creation order, so the greatest id of a group is its
        current state. Dropping everything older is wrong when a channel is not stored
        at that checkpoint: rebuilding it walks back through ancestors' writes until it
        reaches one that does store it, so those ancestors are part of the current state
        rather than history. They are kept, along with their writes.

        When every channel is stored at the newest checkpoint — the ordinary case,
        with no delta channels — only that checkpoint is kept and this costs nothing.
        """
        by_group: Dict[Tuple[str, str], Dict[str, Dict[str, Any]]] = {}
        for row in rows:
            group = (row["thread_id"], row["checkpoint_ns"])
            by_group.setdefault(group, {})[row["checkpoint_id"]] = row

        keep: set = set()
        for group, by_id in by_group.items():
            latest = max(by_id)
            keep.add((group, latest))
            _, carried = self._stored_channels(by_id[latest], stored)
            cursor = by_id[latest].get("parent_checkpoint_id")
            while carried and cursor and cursor in by_id:
                keep.add((group, cursor))
                held, _ = self._stored_channels(by_id[cursor], stored)
                carried -= held
                cursor = by_id[cursor].get("parent_checkpoint_id")

        return [
            row
            for row in rows
            if ((row["thread_id"], row["checkpoint_ns"]), row["checkpoint_id"])
            not in keep
        ]

    @staticmethod
    def _stored_index(blob_rows: List[Dict[str, Any]]) -> set:
        """The (group, channel, version) triples that hold a value."""
        return {
            (
                (row["thread_id"], row["checkpoint_ns"]),
                row["channel"],
                str(row["version"]),
            )
            for row in blob_rows
            if row["type"] != "empty"
        }

    # ---- assembly helpers (pure) ----

    def _blob_rows(
        self, thread_id: str, checkpoint_ns: str, blobs: Dict[str, Tuple[str, str]],
        versions: ChannelVersions,
    ) -> List[Dict[str, Any]]:
        rows = []
        for channel, (type_, b64) in blobs.items():
            rows.append(
                {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "channel": channel,
                    "version": str(versions.get(channel, "")),
                    "type": type_,
                    "blob": b64,
                }
            )
        return rows

    def _write_rows(
        self,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
        task_path: str,
    ) -> List[Dict[str, Any]]:
        rows = []
        for idx, (channel, value) in enumerate(writes):
            type_, b64 = self._dump(value)
            rows.append(
                {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint_id,
                    "task_id": task_id,
                    "idx": WRITES_IDX_MAP.get(channel, idx),
                    "channel": channel,
                    "type": type_,
                    "value": b64,
                    "task_path": task_path,
                }
            )
        return rows

    def _row_to_tuple(
        self, row: Dict[str, Any], blob_rows: List[Dict[str, Any]],
        write_rows: List[Dict[str, Any]],
    ) -> CheckpointTuple:
        checkpoint: Checkpoint = self._load(row["checkpoint_type"], row["checkpoint"])
        # Reassemble channel_values for this checkpoint's versions.
        versions = checkpoint.get("channel_versions", {})
        by_cv = {
            (b["channel"], str(b["version"])): b for b in blob_rows
        }
        channel_values: Dict[str, Any] = {}
        for channel, version in versions.items():
            b = by_cv.get((channel, str(version)))
            if b is not None and b["type"] != "empty":
                channel_values[channel] = self._load(b["type"], b["blob"])
        checkpoint = {**checkpoint, "channel_values": channel_values}
        metadata = self._load(row["metadata_type"], row["metadata"])
        pending_writes = [
            (w["task_id"], w["channel"], self._load(w["type"], w["value"]))
            for w in write_rows
        ]
        cfg = {
            "configurable": {
                "thread_id": row["thread_id"],
                "checkpoint_ns": row["checkpoint_ns"],
                "checkpoint_id": row["checkpoint_id"],
            }
        }
        parent_config = None
        if row.get("parent_checkpoint_id"):
            parent_config = {
                "configurable": {
                    "thread_id": row["thread_id"],
                    "checkpoint_ns": row["checkpoint_ns"],
                    "checkpoint_id": row["parent_checkpoint_id"],
                }
            }
        return CheckpointTuple(
            config=cfg,
            checkpoint=checkpoint,
            metadata=metadata,
            parent_config=parent_config,
            pending_writes=pending_writes,
        )

    @staticmethod
    def _matches_filter(metadata: CheckpointMetadata, flt: Optional[Dict[str, Any]]) -> bool:
        if not flt:
            return True
        return all(metadata.get(k) == v for k, v in flt.items())

    # ---- sync API ----

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        parent = config["configurable"].get("checkpoint_id")
        props, blobs = self._checkpoint_props(
            thread_id, checkpoint_ns, checkpoint, metadata, parent, config
        )
        self._graph.query(
            self._put_checkpoint_cypher(),
            {
                "tid": Jsonb(thread_id),
                "ns": Jsonb(checkpoint_ns),
                "cid": Jsonb(checkpoint["id"]),
                "props": Jsonb(props),
            },
        )
        blob_rows = self._blob_rows(thread_id, checkpoint_ns, blobs, new_versions)
        if blob_rows:
            self._graph.query(self._put_blobs_cypher(), {"blobs": Jsonb(blob_rows)})
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
            }
        }

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"]["checkpoint_id"]
        rows = self._write_rows(
            thread_id, checkpoint_ns, checkpoint_id, writes, task_id, task_path
        )
        if rows:
            self._graph.query(self._put_writes_cypher(), {"writes": Jsonb(rows)})

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        thread_id, checkpoint_ns, checkpoint_id = self._keys(config)
        rows = self._graph.query(
            self._select_checkpoint_cypher(
                by_id=checkpoint_id is not None, before=False, limit=1
            ),
            self._params(thread_id, checkpoint_ns, cid=checkpoint_id),
        )
        if not rows:
            return None
        row = rows[0]
        blob_rows = self._graph.query(
            self._select_blobs_cypher(), self._params(thread_id, checkpoint_ns)
        )
        write_rows = self._graph.query(
            self._select_writes_cypher(),
            self._params(thread_id, checkpoint_ns, cid=row["checkpoint_id"]),
        )
        return self._row_to_tuple(row, blob_rows, write_rows)

    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        thread_id = config["configurable"]["thread_id"] if config else None
        checkpoint_ns = (
            config["configurable"].get("checkpoint_ns", "") if config else ""
        )
        before_id = get_checkpoint_id(before) if before else None
        rows = self._graph.query(
            self._select_checkpoint_cypher(
                by_id=False, before=before_id is not None, limit=limit
            ),
            self._params(thread_id, checkpoint_ns, before=before_id),
        )
        blob_rows = self._graph.query(
            self._select_blobs_cypher(), self._params(thread_id, checkpoint_ns)
        )
        # One query for all writes in the thread, grouped in Python (no N+1).
        writes_by_ckpt = self._group_writes(
            self._graph.query(
                self._select_all_writes_cypher(),
                self._params(thread_id, checkpoint_ns),
            )
        )
        for row in rows:
            write_rows = writes_by_ckpt.get(row["checkpoint_id"], [])
            tup = self._row_to_tuple(row, blob_rows, write_rows)
            if self._matches_filter(tup.metadata, filter):
                yield tup

    def delete_thread(self, thread_id: str) -> None:
        for label in (_CHECKPOINT_LABEL, _BLOB_LABEL, _WRITE_LABEL):
            self._graph.query(
                self._delete_label_cypher(label), {"tid": Jsonb(thread_id)}
            )

    def _delete_checkpoints(self, rows: List[Dict[str, Any]]) -> None:
        """Remove the named checkpoints and the writes belonging to them.

        Channel blobs are shared between the checkpoints of a thread by version, so they
        are left for ``delete_thread`` rather than removed with one of their readers.
        """
        if not rows:
            return
        params: Dict[str, Any] = {}
        predicate = self._triple_predicate(rows, params)
        for label in (_WRITE_LABEL, _CHECKPOINT_LABEL):
            self._graph.query(
                self._delete_checkpoints_cypher(label, predicate), params
            )

    async def _adelete_checkpoints(self, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return
        params: Dict[str, Any] = {}
        predicate = self._triple_predicate(rows, params)
        for label in (_WRITE_LABEL, _CHECKPOINT_LABEL):
            await self._graph.aquery(
                self._delete_checkpoints_cypher(label, predicate), params
            )

    def delete_for_runs(self, run_ids: Sequence[str]) -> None:
        if not run_ids:
            return
        params: Dict[str, Any] = {}
        rows = self._graph.query(
            self._select_by_run_cypher(
                self._value_predicate("run_id", run_ids, params)
            ),
            params,
        )
        self._delete_checkpoints(rows)

    def copy_thread(self, source_thread_id: str, target_thread_id: str) -> None:
        params = {"src": Jsonb(source_thread_id), "tgt": Jsonb(target_thread_id)}
        for label in (_CHECKPOINT_LABEL, _BLOB_LABEL, _WRITE_LABEL):
            self._graph.query(self._copy_thread_cypher(label), params)

    def prune(
        self, thread_ids: Sequence[str], *, strategy: str = "keep_latest"
    ) -> None:
        if not thread_ids:
            return
        if strategy == "delete":
            for thread_id in thread_ids:
                self.delete_thread(thread_id)
            return
        if strategy != "keep_latest":
            raise ValueError(f"Unsupported prune strategy: {strategy}")
        params: Dict[str, Any] = {}
        predicate = self._value_predicate("thread_id", thread_ids, params)
        rows = self._graph.query(self._select_by_thread_cypher(predicate), params)
        blobs = self._graph.query(self._select_thread_blobs_cypher(predicate), params)
        self._delete_checkpoints(self._superseded(rows, self._stored_index(blobs)))

    # ---- delta channel history ----

    DELTA_WINDOW = 32
    """Ancestors fetched per round when rebuilding a delta channel.

    The inherited walk asks for one ancestor at a time, a round trip each; reading the
    whole thread instead is far better when the walk is long and worse when it is short,
    because most of what it reads is then discarded. A window is a compromise that holds
    at both ends: a walk that stops after a few ancestors touches one window, and a long
    one costs a round trip per window rather than per ancestor.
    """

    def _delta_seed(self, channel: str, row: Dict[str, Any], values: Dict[Any, Any]):
        """The stored value of a channel at a checkpoint, if it holds one."""
        checkpoint = self._load(row["checkpoint_type"], row["checkpoint"])
        versions = checkpoint.get("channel_versions", {}) or {}
        if channel not in versions:
            return None
        return values.get((channel, str(versions[channel])))

    def _delta_walk(
        self,
        channels: Sequence[str],
        cursor: Optional[str],
        values: Dict[Any, Any],
        fetch_window,
    ) -> Dict[str, Any]:
        """Collect each channel's writes back to the ancestor that stores it.

        ``fetch_window`` returns the next span of ancestors and their writes, so one
        walk serves both the sync and the async path.
        """
        collected: Dict[str, List[Any]] = {c: [] for c in channels}
        seeds: Dict[str, Any] = {}
        remaining = set(channels)

        while cursor is not None and remaining:
            rows, writes_by_ckpt = fetch_window(cursor)
            if not rows:
                break
            by_id = {row["checkpoint_id"]: row for row in rows}
            advanced = False
            while cursor is not None and remaining and cursor in by_id:
                row = by_id[cursor]
                advanced = True
                pending = [
                    (w["task_id"], w["channel"], self._load(w["type"], w["value"]))
                    for w in writes_by_ckpt.get(cursor, [])
                ]
                for write in reversed(pending):
                    if write[1] in remaining:
                        collected[write[1]].append(write)
                for channel in list(remaining):
                    blob = self._delta_seed(channel, row, values)
                    if blob is not None:
                        seeds[channel] = self._load(blob["type"], blob["blob"])
                        remaining.discard(channel)
                cursor = row.get("parent_checkpoint_id")
            if not advanced:
                break

        history: Dict[str, Any] = {}
        for channel in channels:
            entry: Dict[str, Any] = {"writes": list(reversed(collected[channel]))}
            if channel in seeds:
                entry["seed"] = seeds[channel]
            history[channel] = entry
        return history

    @staticmethod
    def _values_index(blobs: List[Dict[str, Any]]) -> Dict[Any, Any]:
        return {
            (b["channel"], str(b["version"])): b for b in blobs if b["type"] != "empty"
        }

    def _parent_of(self, config: RunnableConfig) -> Optional[str]:
        tuple_ = self.get_tuple(config)
        if tuple_ is None or tuple_.parent_config is None:
            return None
        return get_checkpoint_id(tuple_.parent_config)

    def get_delta_channel_history(
        self, *, config: RunnableConfig, channels: Sequence[str]
    ) -> Dict[str, Any]:
        """Per-channel writes and seed, walked a window of ancestors at a time.

        Starts at the target's parent, so the target's own writes are excluded, and
        omits ``seed`` for a channel no ancestor stores — read as "start empty".
        """
        if not channels:
            return {}
        thread_id, checkpoint_ns, _ = self._keys(config)
        base = {"tid": Jsonb(thread_id), "ns": Jsonb(checkpoint_ns)}
        values = self._values_index(
            self._graph.query(self._select_blobs_cypher(), base)
        )

        def fetch_window(cursor: str):
            rows = self._graph.query(
                self._select_ancestors_cypher(self.DELTA_WINDOW),
                {**base, "hi": Jsonb(cursor)},
            )
            if not rows:
                return [], {}
            writes = self._graph.query(
                self._select_writes_window_cypher(),
                {**base, "lo": Jsonb(rows[-1]["checkpoint_id"]), "hi": Jsonb(cursor)},
            )
            return rows, self._group_writes(writes)

        return self._delta_walk(
            channels, self._parent_of(config), values, fetch_window
        )

    # ---- async API ----

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        parent = config["configurable"].get("checkpoint_id")
        props, blobs = self._checkpoint_props(
            thread_id, checkpoint_ns, checkpoint, metadata, parent, config
        )
        await self._graph.aquery(
            self._put_checkpoint_cypher(),
            {
                "tid": Jsonb(thread_id),
                "ns": Jsonb(checkpoint_ns),
                "cid": Jsonb(checkpoint["id"]),
                "props": Jsonb(props),
            },
        )
        blob_rows = self._blob_rows(thread_id, checkpoint_ns, blobs, new_versions)
        if blob_rows:
            await self._graph.aquery(
                self._put_blobs_cypher(), {"blobs": Jsonb(blob_rows)}
            )
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
            }
        }

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"]["checkpoint_id"]
        rows = self._write_rows(
            thread_id, checkpoint_ns, checkpoint_id, writes, task_id, task_path
        )
        if rows:
            await self._graph.aquery(self._put_writes_cypher(), {"writes": Jsonb(rows)})

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        thread_id, checkpoint_ns, checkpoint_id = self._keys(config)
        rows = await self._graph.aquery(
            self._select_checkpoint_cypher(
                by_id=checkpoint_id is not None, before=False, limit=1
            ),
            self._params(thread_id, checkpoint_ns, cid=checkpoint_id),
        )
        if not rows:
            return None
        row = rows[0]
        blob_rows = await self._graph.aquery(
            self._select_blobs_cypher(), self._params(thread_id, checkpoint_ns)
        )
        write_rows = await self._graph.aquery(
            self._select_writes_cypher(),
            self._params(thread_id, checkpoint_ns, cid=row["checkpoint_id"]),
        )
        return self._row_to_tuple(row, blob_rows, write_rows)

    async def alist(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        thread_id = config["configurable"]["thread_id"] if config else None
        checkpoint_ns = (
            config["configurable"].get("checkpoint_ns", "") if config else ""
        )
        before_id = get_checkpoint_id(before) if before else None
        rows = await self._graph.aquery(
            self._select_checkpoint_cypher(
                by_id=False, before=before_id is not None, limit=limit
            ),
            self._params(thread_id, checkpoint_ns, before=before_id),
        )
        blob_rows = await self._graph.aquery(
            self._select_blobs_cypher(), self._params(thread_id, checkpoint_ns)
        )
        writes_by_ckpt = self._group_writes(
            await self._graph.aquery(
                self._select_all_writes_cypher(),
                self._params(thread_id, checkpoint_ns),
            )
        )
        for row in rows:
            write_rows = writes_by_ckpt.get(row["checkpoint_id"], [])
            tup = self._row_to_tuple(row, blob_rows, write_rows)
            if self._matches_filter(tup.metadata, filter):
                yield tup

    async def adelete_thread(self, thread_id: str) -> None:
        for label in (_CHECKPOINT_LABEL, _BLOB_LABEL, _WRITE_LABEL):
            await self._graph.aquery(
                self._delete_label_cypher(label), {"tid": Jsonb(thread_id)}
            )

    async def adelete_for_runs(self, run_ids: Sequence[str]) -> None:
        if not run_ids:
            return
        params: Dict[str, Any] = {}
        rows = await self._graph.aquery(
            self._select_by_run_cypher(
                self._value_predicate("run_id", run_ids, params)
            ),
            params,
        )
        await self._adelete_checkpoints(rows)

    async def acopy_thread(self, source_thread_id: str, target_thread_id: str) -> None:
        params = {"src": Jsonb(source_thread_id), "tgt": Jsonb(target_thread_id)}
        for label in (_CHECKPOINT_LABEL, _BLOB_LABEL, _WRITE_LABEL):
            await self._graph.aquery(self._copy_thread_cypher(label), params)

    async def aprune(
        self, thread_ids: Sequence[str], *, strategy: str = "keep_latest"
    ) -> None:
        if not thread_ids:
            return
        if strategy == "delete":
            for thread_id in thread_ids:
                await self.adelete_thread(thread_id)
            return
        if strategy != "keep_latest":
            raise ValueError(f"Unsupported prune strategy: {strategy}")
        params: Dict[str, Any] = {}
        predicate = self._value_predicate("thread_id", thread_ids, params)
        rows = await self._graph.aquery(
            self._select_by_thread_cypher(predicate), params
        )
        blobs = await self._graph.aquery(
            self._select_thread_blobs_cypher(predicate), params
        )
        await self._adelete_checkpoints(
            self._superseded(rows, self._stored_index(blobs))
        )

    async def aget_delta_channel_history(
        self, *, config: RunnableConfig, channels: Sequence[str]
    ) -> Dict[str, Any]:
        if not channels:
            return {}
        thread_id, checkpoint_ns, _ = self._keys(config)
        base = {"tid": Jsonb(thread_id), "ns": Jsonb(checkpoint_ns)}
        values = self._values_index(
            await self._graph.aquery(self._select_blobs_cypher(), base)
        )
        target = await self.aget_tuple(config)
        cursor = (
            get_checkpoint_id(target.parent_config)
            if target is not None and target.parent_config is not None
            else None
        )

        collected: Dict[str, List[Any]] = {c: [] for c in channels}
        seeds: Dict[str, Any] = {}
        remaining = set(channels)
        while cursor is not None and remaining:
            rows = await self._graph.aquery(
                self._select_ancestors_cypher(self.DELTA_WINDOW),
                {**base, "hi": Jsonb(cursor)},
            )
            if not rows:
                break
            writes = await self._graph.aquery(
                self._select_writes_window_cypher(),
                {**base, "lo": Jsonb(rows[-1]["checkpoint_id"]), "hi": Jsonb(cursor)},
            )
            writes_by_ckpt = self._group_writes(writes)
            by_id = {row["checkpoint_id"]: row for row in rows}
            advanced = False
            while cursor is not None and remaining and cursor in by_id:
                row = by_id[cursor]
                advanced = True
                pending = [
                    (w["task_id"], w["channel"], self._load(w["type"], w["value"]))
                    for w in writes_by_ckpt.get(cursor, [])
                ]
                for write in reversed(pending):
                    if write[1] in remaining:
                        collected[write[1]].append(write)
                for channel in list(remaining):
                    blob = self._delta_seed(channel, row, values)
                    if blob is not None:
                        seeds[channel] = self._load(blob["type"], blob["blob"])
                        remaining.discard(channel)
                cursor = row.get("parent_checkpoint_id")
            if not advanced:
                break

        history: Dict[str, Any] = {}
        for channel in channels:
            entry: Dict[str, Any] = {"writes": list(reversed(collected[channel]))}
            if channel in seeds:
                entry["seed"] = seeds[channel]
            history[channel] = entry
        return history

    # ---- param helper ----

    @staticmethod
    def _params(
        thread_id: Optional[str],
        checkpoint_ns: str,
        *,
        cid: Optional[str] = None,
        before: Optional[str] = None,
    ) -> Dict[str, Any]:
        p: Dict[str, Any] = {
            "tid": Jsonb(thread_id),
            "ns": Jsonb(checkpoint_ns),
        }
        if cid is not None:
            p["cid"] = Jsonb(cid)
        if before is not None:
            p["before"] = Jsonb(before)
        return p


# Alias provided for naming convenience. The class implements both sync and
# async methods, so the alias is the same class.
AsyncAgensSaver = AgensSaver

__all__ = ["AgensSaver", "AsyncAgensSaver"]
