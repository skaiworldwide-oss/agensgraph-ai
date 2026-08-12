"""LangGraph checkpoint saver backed by AgensGraph.

Persists LangGraph checkpoints as graph vertices so an agent's conversation
state survives process restarts and can be resumed by ``thread_id``:

* ``(:Checkpoint {thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id,
   checkpoint_type, checkpoint, metadata_type, metadata})``
* ``(:CheckpointBlob {thread_id, checkpoint_ns, channel, version, type, blob})``
  — channel values, shared across checkpoints by ``version``.
* ``(:CheckpointWrite {thread_id, checkpoint_ns, checkpoint_id, task_id, idx,
   channel, type, value, task_path})`` — pending writes.

Serialized payloads (which are raw bytes from the serializer) are base64-encoded because
AgensGraph stores all properties as ``jsonb``, which cannot hold bytes.

The implementation mirrors the storage contract of LangGraph's reference
``InMemorySaver``. ``AgensSaver`` exposes both sync and async methods;
``AsyncAgensSaver`` is an alias provided for naming convenience.
"""

from __future__ import annotations

import base64
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
)

from agensgraph.introspect import DesiredIndex
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


DELETE_CHUNK = 5000
"""How many checkpoints one delete statement names.

Each one binds three parameters -- thread, namespace and checkpoint id -- and PostgreSQL
takes at most 65,535 in a statement, so an unchunked prune of a long-lived thread failed
at around 21,800 checkpoints with a protocol error rather than anything a caller could
read. Five thousand leaves room for the statement's own parameters.
"""


def _chunked(rows: List[Dict[str, Any]], size: int) -> Iterator[List[Dict[str, Any]]]:
    """The rows in batches, and nothing at all for no rows."""
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


class _ListScope(NamedTuple):
    """Which checkpoints a ``list`` call is asking about."""

    thread_id: Optional[str]
    checkpoint_ns: Optional[str]
    checkpoint_id: Optional[str]
    before_id: Optional[str]

    @property
    def any_thread(self) -> bool:
        return self.thread_id is None

    @property
    def any_ns(self) -> bool:
        return self.checkpoint_ns is None

    @property
    def params(self) -> Dict[str, Any]:
        p: Dict[str, Any] = {"tid": self.thread_id, "ns": self.checkpoint_ns}
        if self.checkpoint_id is not None:
            p["cid"] = self.checkpoint_id
        if self.before_id is not None:
            p["before"] = self.before_id
        return p

    def checkpoint_args(
        self, limit: Optional[int], filter: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """How to build the statement that reads the checkpoints themselves.

        ``limit`` is asked of the server only when there is no metadata filter for a row
        to fail afterwards; with one, the rows are counted as they pass it instead.

        Which means a filtered list reads the thread rather than a page of it -- 200
        checkpoints read to return 10 -- and with no config at all, every thread in the
        database. That is the price of the answer being right: cutting at the server cuts
        before the filter has been applied, so a page of ten comes back holding however
        many of its ten happened to match, which is what this looked like before and is
        not what a limit means. A filter narrow enough to be worth writing over a thread
        long enough for this to hurt wants a property to narrow on instead.
        """
        return {
            "by_id": self.checkpoint_id is not None,
            "before": self.before_id is not None,
            "limit": None if filter else limit,
            "any_thread": self.any_thread,
            "any_ns": self.any_ns,
        }


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
            # Nothing here reads the schema, and describing a graph counts every vertex
            # and every edge in it, for a component that never looks at the answer.
            graph = AgensGraph(
                graph_name, conf, create=True, refresh_schema=False
            )
            self._owns_graph = True
        else:
            self._owns_graph = False
        self._graph = graph
        self._graph.create_labels(
            vertices=(_CHECKPOINT_LABEL, _BLOB_LABEL, _WRITE_LABEL)
        )
        # Every read filters by thread and namespace, so without these composite indexes
        # a get, a list or a put reads the whole label.
        #
        # Each one is unique over exactly the key its `MERGE` matches on, and that is
        # what keeps two writers from creating the same element twice: `MERGE` looks,
        # finds nothing, and creates, and two of them can look before either creates. Only
        # the index can refuse the second. A read narrowing by thread and namespace is
        # still served by the leading columns of the same index.
        self._graph.ensure_indexes(
            [
                DesiredIndex(
                    label=_CHECKPOINT_LABEL,
                    properties=("thread_id", "checkpoint_ns", "checkpoint_id"),
                    unique=True,
                    name=f"{_CHECKPOINT_LABEL}_thread_idx",
                ),
                DesiredIndex(
                    label=_BLOB_LABEL,
                    properties=("thread_id", "checkpoint_ns", "channel", "version"),
                    unique=True,
                    name=f"{_BLOB_LABEL}_thread_idx",
                ),
                DesiredIndex(
                    label=_WRITE_LABEL,
                    properties=(
                        "thread_id",
                        "checkpoint_ns",
                        "checkpoint_id",
                        "task_id",
                        "idx",
                    ),
                    unique=True,
                    name=f"{_WRITE_LABEL}_thread_idx",
                ),
                # delete_for_runs selects by run, which no thread index covers. A run
                # holds many checkpoints, so this one is not unique.
                DesiredIndex(
                    label=_CHECKPOINT_LABEL,
                    properties=("run_id",),
                    name=f"{_CHECKPOINT_LABEL}_run_idx",
                ),
            ]
        )

    # ---- key extraction ----

    @staticmethod
    def _keys(config: Optional[RunnableConfig]) -> Tuple[Optional[str], str, Optional[str]]:
        """The thread, namespace and checkpoint a config names.

        ``config`` is optional in the interface and means "across every thread". It used
        to be indexed straight into, so ``list(None)`` produced a predicate comparing
        the thread to null, which matches nothing -- silently returning no checkpoints
        where every checkpoint was asked for.
        """
        if config is None:
            return (None, "", None)
        cfg = config["configurable"]
        return (
            cfg["thread_id"],
            cfg.get("checkpoint_ns", ""),
            get_checkpoint_id(config),
        )

    def close(self) -> None:
        """Close the graph this saver opened for itself.

        A saver built from ``conf=`` opens an ``AgensGraph`` the caller never sees and
        so cannot close; one built from ``graph=`` was given a graph that belongs to
        somebody else and leaves it alone.
        """
        if self._owns_graph:
            self._graph.close()

    async def aclose(self) -> None:
        """Async sibling of :meth:`close`."""
        if self._owns_graph:
            await self._graph.aclose()
            self._graph.close()

    def __enter__(self) -> "AgensSaver":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    async def __aenter__(self) -> "AgensSaver":
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.aclose()

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
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Return (checkpoint node properties, the channel values held out of them).

        The values are kept out of the checkpoint's own properties and written as rows of
        their own, so reading a checkpoint's metadata does not carry its state with it.
        They are returned undumped: which of them is written is decided by the versions
        the superstep produced, so serializing here would serialize values that are not
        going to be written.
        """
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
        return props, channel_values

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

    @staticmethod
    def _scope(alias: str, any_thread: bool, any_ns: bool) -> List[str]:
        """The terms that narrow a label to one thread and namespace.

        A term is left out rather than compared to null when the caller named nothing to
        compare against: ``list`` takes an optional config and an absent namespace means
        every namespace, so comparing to null would match nothing and answer that a
        thread holds no checkpoints.
        """
        terms = []
        if not any_thread:
            terms.append(f"{alias}.thread_id = %(tid)s")
        if not any_ns:
            terms.append(f"{alias}.checkpoint_ns = %(ns)s")
        return terms

    @staticmethod
    def _where(terms: Sequence[str]) -> str:
        return ("WHERE " + " AND ".join(terms) + " ") if terms else ""

    def _select_checkpoint_cypher(
        self,
        by_id: bool,
        before: bool,
        limit: Optional[int],
        any_thread: bool = False,
        any_ns: bool = False,
    ):
        terms = self._scope("c", any_thread, any_ns)
        if by_id:
            terms.append("c.checkpoint_id = %(cid)s")
        if before:
            terms.append("c.checkpoint_id < %(before)s")
        tail = " ORDER BY c.checkpoint_id DESC"
        if limit is not None:
            tail += f" LIMIT {int(limit)}"
        return sql.SQL(
            "MATCH (c:{cl}) " + self._where(terms) +
            "RETURN c.thread_id AS thread_id, c.checkpoint_ns AS checkpoint_ns, "
            "c.checkpoint_id AS checkpoint_id, c.parent_checkpoint_id AS parent_checkpoint_id, "
            "c.checkpoint_type AS checkpoint_type, c.checkpoint AS checkpoint, "
            "c.metadata_type AS metadata_type, c.metadata AS metadata" + tail
        ).format(cl=sql.Identifier(_CHECKPOINT_LABEL))

    def _select_blobs_cypher(self, any_thread: bool = False, any_ns: bool = False):
        return sql.SQL(
            "MATCH (x:{bl}) " + self._where(self._scope("x", any_thread, any_ns)) +
            "RETURN x.thread_id AS thread_id, x.checkpoint_ns AS checkpoint_ns, "
            "x.channel AS channel, x.version AS version, x.type AS type, x.blob AS blob"
        ).format(bl=sql.Identifier(_BLOB_LABEL))

    def _select_named_blobs(
        self, thread_id: str, checkpoint_ns: str, versions: ChannelVersions
    ) -> Tuple[sql.Composed, Dict[str, Any]]:
        """The channel values one checkpoint records, and no others.

        A thread accumulates a value per channel per version for as long as it runs, and
        the checkpoint being read names the handful it is made of, so resuming costs the
        state rather than the age of the conversation.

        Each pair is an equality the composite index answers, unwound from a list so that
        the statement is the same one however many channels the checkpoint has.

        A checkpoint that names no channels asks for nothing rather than for the thread.
        """
        keys = [
            {"c": channel, "v": str(version)} for channel, version in versions.items()
        ]
        params: Dict[str, Any] = {
            "tid": thread_id,
            "ns": checkpoint_ns,
            "bkeys": Jsonb(keys),
        }
        statement = sql.SQL(
            "UNWIND %(bkeys)s AS k MATCH (x:{bl}) "
            "WHERE x.thread_id = %(tid)s AND x.checkpoint_ns = %(ns)s "
            "AND x.channel = k.c AND x.version = k.v "
            "RETURN x.thread_id AS thread_id, x.checkpoint_ns AS checkpoint_ns, "
            "x.channel AS channel, x.version AS version, x.type AS type, x.blob AS blob"
        ).format(bl=sql.Identifier(_BLOB_LABEL))
        return statement, params

    def _blobs_for_rows(
        self, rows: List[Dict[str, Any]]
    ) -> Tuple[Optional[sql.Composed], Dict[str, Any]]:
        """The channel values the checkpoints in hand are made of, and no others.

        A thread accumulates a value per channel per version for as long as it runs, so a
        page names the versions its own checkpoints hold rather than reading everything
        the thread has ever held.

        Each checkpoint names its versions, and the union of what the page names is what
        is asked for. ``None`` when the page names nothing, since then there is nothing to
        ask.
        """
        wanted: set = set()
        for row in rows:
            group = (row["thread_id"], row["checkpoint_ns"])
            for channel, version in self._versions_of(row).items():
                wanted.add((group, channel, str(version)))
        if not wanted:
            return None, {}
        keys = [
            {"t": thread_id, "n": checkpoint_ns, "c": channel, "v": version}
            for ((thread_id, checkpoint_ns), channel, version) in sorted(wanted)
        ]
        statement = sql.SQL(
            "UNWIND %(bkeys)s AS k MATCH (x:{bl}) "
            "WHERE x.thread_id = k.t AND x.checkpoint_ns = k.n "
            "AND x.channel = k.c AND x.version = k.v "
            "RETURN x.thread_id AS thread_id, x.checkpoint_ns AS checkpoint_ns, "
            "x.channel AS channel, x.version AS version, x.type AS type, x.blob AS blob"
        ).format(bl=sql.Identifier(_BLOB_LABEL))
        return statement, {"bkeys": Jsonb(keys)}

    def _writes_for_rows(
        self, rows: List[Dict[str, Any]]
    ) -> Tuple[Optional[sql.Composed], Dict[str, Any]]:
        """The pending writes of the checkpoints in hand.

        The keys are unwound rather than written into the statement as a term each. One
        term per key makes the statement grow with the page and the planner pays for every
        term; one probe repeated over an unwound list is planned once, whatever the page.
        """
        if not rows:
            return None, {}
        keys = [
            {
                "t": row["thread_id"],
                "n": row["checkpoint_ns"],
                "c": row["checkpoint_id"],
            }
            for row in rows
        ]
        statement = sql.SQL(
            "UNWIND %(wkeys)s AS k MATCH (x:{wl}) "
            "WHERE x.thread_id = k.t AND x.checkpoint_ns = k.n "
            "AND x.checkpoint_id = k.c "
            "RETURN x.thread_id AS thread_id, x.checkpoint_ns AS checkpoint_ns, "
            "x.checkpoint_id AS checkpoint_id, x.task_id AS task_id, "
            "x.idx AS idx, x.channel AS channel, x.type AS type, x.value AS value "
            "ORDER BY x.idx"
        ).format(wl=sql.Identifier(_WRITE_LABEL))
        return statement, {"wkeys": Jsonb(keys)}

    def _versions_of(self, row: Dict[str, Any]) -> ChannelVersions:
        """The channel versions a checkpoint row records."""
        checkpoint = self._load(row["checkpoint_type"], row["checkpoint"])
        return checkpoint.get("channel_versions", {}) or {}

    def _select_writes_cypher(self):
        return sql.SQL(
            "MATCH (x:{wl}) WHERE x.thread_id = %(tid)s AND x.checkpoint_ns = %(ns)s "
            "AND x.checkpoint_id = %(cid)s "
            "RETURN x.task_id AS task_id, x.idx AS idx, x.channel AS channel, "
            "x.type AS type, x.value AS value ORDER BY x.idx"
        ).format(wl=sql.Identifier(_WRITE_LABEL))

    def _select_all_writes_cypher(self, any_thread: bool = False, any_ns: bool = False):
        # All writes for a thread/ns in one shot (avoids an N+1 per checkpoint in
        # ``list``); grouped by checkpoint in Python. The thread and namespace come back
        # so that grouping can key on the checkpoint a write belongs to rather than on
        # its id alone, which is not unique once more than one thread is being read.
        return sql.SQL(
            "MATCH (x:{wl}) " + self._where(self._scope("x", any_thread, any_ns)) +
            "RETURN x.thread_id AS thread_id, x.checkpoint_ns AS checkpoint_ns, "
            "x.checkpoint_id AS checkpoint_id, x.task_id AS task_id, "
            "x.idx AS idx, x.channel AS channel, x.type AS type, x.value AS value "
            "ORDER BY x.idx"
        ).format(wl=sql.Identifier(_WRITE_LABEL))

    @staticmethod
    def _group_writes(
        rows: List[Dict[str, Any]],
    ) -> Dict[Tuple[str, str, str], List[Dict[str, Any]]]:
        """Writes by the checkpoint they belong to, thread and namespace included.

        A checkpoint id identifies a checkpoint within its thread, so reading more than
        one thread at a time and grouping on the id alone would hand one checkpoint's
        pending writes to another's.
        """
        grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
        for w in rows:
            key = (w["thread_id"], w["checkpoint_ns"], w["checkpoint_id"])
            grouped.setdefault(key, []).append(w)
        return grouped

    @staticmethod
    def _group_writes_by_id(
        rows: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Writes by checkpoint id, for a read already narrowed to one thread.

        The walk back through a thread's ancestors names the thread and the namespace in
        its predicate, so within its results an id identifies a checkpoint on its own.
        """
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
            params[f"v{i}"] = value
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
        """Checkpoints with the parent link and payload a delta walk needs.

        The payload of every checkpoint of the threads named, which is what a prune costs
        and is not narrowable from here: what the walk needs is each checkpoint's channel
        versions, and those live inside the payload. Reading fewer would mean keeping the
        versions somewhere a query can reach without the payload around them.
        """
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

    def _thread_is_occupied_cypher(self):
        """Whether a thread holds anything at all, asked of the cheapest label."""
        return sql.SQL(
            "MATCH (n:{l}) WHERE n.thread_id = %(tid)s RETURN count(n) AS c"
        ).format(l=sql.Identifier(_CHECKPOINT_LABEL))

    def _delete_checkpoints_cypher(self, label: str, predicate):
        return sql.SQL("MATCH (n:{l}) WHERE {pred} DETACH DELETE n").format(
            l=sql.Identifier(label), pred=predicate
        )

    @staticmethod
    def _triple_predicate(rows: List[Dict[str, Any]], params: Dict[str, Any]):
        """Match specific (thread, namespace, checkpoint) rows."""
        terms = []
        for i, row in enumerate(rows):
            params[f"t{i}"] = row["thread_id"]
            params[f"n{i}"] = row["checkpoint_ns"]
            params[f"c{i}"] = row["checkpoint_id"]
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

        When every channel is stored at the newest checkpoint — the ordinary case, with
        no delta channels — only that checkpoint is kept and this costs nothing.
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
        self, thread_id: str, checkpoint_ns: str, channel_values: Dict[str, Any],
        new_versions: ChannelVersions,
    ) -> List[Dict[str, Any]]:
        """One row per channel that took a new version at this checkpoint.

        The new versions are what a superstep changed, and they are what is written. A
        channel whose value did not change keeps the row written when it last did, so a
        long-lived channel is serialized once rather than at every superstep, and the row
        is keyed by the version the checkpoint records so a read can find it.

        A channel that took a new version while holding no value gets a row saying it is
        empty, which is a different answer from having no row: no row means the value is
        carried by the writes of this checkpoint and its ancestors, and a reader walks
        back through them looking for it.
        """
        rows = []
        for channel, version in new_versions.items():
            if channel in channel_values:
                type_, blob = self._dump(channel_values[channel])
            else:
                type_, blob = "empty", ""
            rows.append(
                {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "channel": channel,
                    "version": str(version),
                    "type": type_,
                    "blob": blob,
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

    @staticmethod
    def _blob_index(
        blob_rows: List[Dict[str, Any]],
    ) -> Dict[Tuple[Any, str, str], Dict[str, Any]]:
        """Where each stored value is, by the checkpoint and version that names it.

        Built once for a page and handed to every row of it. Built per row it was the
        whole set of blobs walked again for each checkpoint returned, which for a list is
        the page multiplied by the values the page is made of.
        """
        return {
            (
                (b["thread_id"], b["checkpoint_ns"]),
                b["channel"],
                str(b["version"]),
            ): b
            for b in blob_rows
        }

    def _row_to_tuple(
        self, row: Dict[str, Any], blob_rows: List[Dict[str, Any]],
        write_rows: List[Dict[str, Any]],
        blob_index: Optional[Dict[Tuple[Any, str, str], Dict[str, Any]]] = None,
    ) -> CheckpointTuple:
        checkpoint: Checkpoint = self._load(row["checkpoint_type"], row["checkpoint"])
        # Reassemble channel_values for this checkpoint's versions. A blob is found by
        # the thread and namespace it belongs to as well as its channel and version,
        # because `list` may be reading across every thread and namespace at once and a
        # version numbers a channel within its own thread, not across them.
        versions = checkpoint.get("channel_versions", {})
        group = (row["thread_id"], row["checkpoint_ns"])
        by_cv = blob_index if blob_index is not None else self._blob_index(blob_rows)
        channel_values: Dict[str, Any] = {}
        for channel, version in versions.items():
            b = by_cv.get((group, channel, str(version)))
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

    def _list_scope(
        self, config: Optional[RunnableConfig], before: Optional[RunnableConfig]
    ) -> "_ListScope":
        """What a ``list`` config narrows the search to.

        Each part the caller left out widens it. A namespace that was not named is not
        compared, rather than compared against the default one -- naming a thread and no
        namespace asks about the thread, and answering only for its default namespace
        leaves out every checkpoint a subgraph wrote.
        """
        return _ListScope(
            thread_id=config["configurable"]["thread_id"] if config else None,
            checkpoint_ns=(
                config["configurable"].get("checkpoint_ns") if config else None
            ),
            checkpoint_id=get_checkpoint_id(config) if config else None,
            before_id=get_checkpoint_id(before) if before else None,
        )

    def _assemble(
        self,
        rows: List[Dict[str, Any]],
        blob_rows: List[Dict[str, Any]],
        writes_by_ckpt: Dict[Tuple[str, str, str], List[Dict[str, Any]]],
        filter: Optional[Dict[str, Any]],
        limit: Optional[int],
    ) -> Iterator[CheckpointTuple]:
        """Build a tuple per row, counting toward ``limit`` only the ones returned.

        A metadata filter is applied to the assembled tuple, since the metadata is stored
        serialized and no query can read into it. So the count has to happen here: a row
        that goes on to fail the filter is not one of the ones the caller asked for, and
        counting it would answer a request for ten matches with fewer than ten.
        """
        returned = 0
        index = self._blob_index(blob_rows)
        for row in rows:
            key = (row["thread_id"], row["checkpoint_ns"], row["checkpoint_id"])
            tup = self._row_to_tuple(
                row, blob_rows, writes_by_ckpt.get(key, []), index
            )
            if not self._matches_filter(tup.metadata, filter):
                continue
            yield tup
            returned += 1
            if limit is not None and returned >= limit:
                return

    def _put_statements(
        self,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
        props: Dict[str, Any],
        blob_rows: List[Dict[str, Any]],
    ) -> List[Tuple[sql.Composed, Dict[str, Any]]]:
        """What a superstep writes: the checkpoint, and the values it is made of."""
        statements: List[Tuple[sql.Composed, Dict[str, Any]]] = [
            (
                self._put_checkpoint_cypher(),
                {
                    "tid": thread_id,
                    "ns": checkpoint_ns,
                    "cid": checkpoint_id,
                    "props": Jsonb(props),
                },
            )
        ]
        if blob_rows:
            statements.append(
                (self._put_blobs_cypher(), {"blobs": Jsonb(blob_rows)})
            )
        return statements

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
        blob_rows = self._blob_rows(thread_id, checkpoint_ns, blobs, new_versions)

        statements = self._put_statements(
            thread_id, checkpoint_ns, checkpoint["id"], props, blob_rows
        )

        def write() -> None:
            # The checkpoint and the values it is made of in one transaction: a
            # checkpoint whose blobs did not land cannot be resumed, and would sit there
            # looking as though it can. The two go in one round trip since neither answers
            # anything.
            with self._graph.transaction() as conn:
                conn.pipeline_batch(
                    [(one.as_string(conn), params) for one, params in statements]
                )

        self._graph.merging(write)
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
            self._graph.merging(
                lambda: self._graph.query(
                    self._put_writes_cypher(), {"writes": Jsonb(rows)}
                )
            )

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
        statement, params = self._select_named_blobs(
            row["thread_id"], row["checkpoint_ns"], self._versions_of(row)
        )
        blob_rows = self._graph.query(statement, params)
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
        """Checkpoints newest first, narrowed by whatever the caller named.

        Every part of the config is optional and each absent part widens the search: no
        config at all asks across every thread, and a config naming no namespace asks
        across every namespace of the thread it names.
        """
        scope = self._list_scope(config, before)
        rows = self._graph.query(
            self._select_checkpoint_cypher(**scope.checkpoint_args(limit, filter)),
            scope.params,
        )
        blob_statement, blob_params = self._blobs_for_rows(rows)
        blob_rows = (
            self._graph.query(blob_statement, blob_params) if blob_statement else []
        )
        # One query for the writes of every checkpoint on the page, grouped in Python
        # rather than one query each.
        write_statement, write_params = self._writes_for_rows(rows)
        writes_by_ckpt = self._group_writes(
            self._graph.query(write_statement, write_params) if write_statement else []
        )
        yield from self._assemble(rows, blob_rows, writes_by_ckpt, filter, limit)

    def delete_thread(self, thread_id: str) -> None:
        """Remove a thread's checkpoints, the values they are made of, and their writes.

        All three together. A thread whose checkpoints went but whose blobs stayed is not
        a partly-deleted thread, it is a thread that still answers a history read with
        entries nothing can load.
        """
        with self._graph.transaction():
            for label in (_CHECKPOINT_LABEL, _BLOB_LABEL, _WRITE_LABEL):
                self._graph.query(
                    self._delete_label_cypher(label), {"tid": thread_id}
                )

    def _delete_checkpoints(self, rows: List[Dict[str, Any]]) -> None:
        """Remove the named checkpoints and the writes belonging to them.

        Channel blobs are shared between the checkpoints of a thread by version, so they
        are left for ``delete_thread`` rather than removed with one of their readers.
        """
        for batch in _chunked(rows, DELETE_CHUNK):
            params: Dict[str, Any] = {}
            predicate = self._triple_predicate(batch, params)
            for label in (_WRITE_LABEL, _CHECKPOINT_LABEL):
                self._graph.query(
                    self._delete_checkpoints_cypher(label, predicate), params
                )

    async def _adelete_checkpoints(self, rows: List[Dict[str, Any]]) -> None:
        """Async sibling of :meth:`_delete_checkpoints`, chunked for the same reason."""
        for batch in _chunked(rows, DELETE_CHUNK):
            params: Dict[str, Any] = {}
            predicate = self._triple_predicate(batch, params)
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
        """Copy a thread's checkpoints, blobs and writes onto a thread that has none.

        The copy creates, so a target that already holds checkpoints would end up with
        two of everything: the same checkpoint id twice over, a history that lists each
        entry twice, and a prune that keeps both copies of the newest. It is refused
        rather than merged, because a thread the caller did not expect to be occupied is
        as likely to be the wrong name as it is to be a repeat of this call.

        The three copies go together. A target holding checkpoints whose values were not
        copied is worse than a target holding nothing: it looks resumable and is not, and
        the refusal above will not let a second attempt fix it.
        """
        self._refuse_occupied(target_thread_id)
        params = {"src": source_thread_id, "tgt": target_thread_id}
        with self._graph.transaction():
            for label in (_CHECKPOINT_LABEL, _BLOB_LABEL, _WRITE_LABEL):
                self._graph.query(self._copy_thread_cypher(label), params)

    @staticmethod
    def _occupied_message(target_thread_id: str, held: int) -> str:
        return (
            f"thread {target_thread_id!r} already holds {held} checkpoint(s), and copying "
            f"onto it would duplicate every one of them rather than replace it. Delete it "
            f"first with delete_thread({target_thread_id!r}), or copy to a thread id that "
            f"is not in use"
        )

    def _refuse_occupied(self, target_thread_id: str) -> None:
        held = self._graph.query(
            self._thread_is_occupied_cypher(), {"tid": target_thread_id}
        )[0]["c"]
        if held:
            raise ValueError(self._occupied_message(target_thread_id, held))

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
        base = {"tid": thread_id, "ns": checkpoint_ns}
        values = self._values_index(
            self._graph.query(self._select_blobs_cypher(), base)
        )

        def fetch_window(cursor: str):
            rows = self._graph.query(
                self._select_ancestors_cypher(self.DELTA_WINDOW),
                {**base, "hi": cursor},
            )
            if not rows:
                return [], {}
            writes = self._graph.query(
                self._select_writes_window_cypher(),
                {**base, "lo": rows[-1]["checkpoint_id"], "hi": cursor},
            )
            return rows, self._group_writes_by_id(writes)

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
        blob_rows = self._blob_rows(thread_id, checkpoint_ns, blobs, new_versions)

        statements = self._put_statements(
            thread_id, checkpoint_ns, checkpoint["id"], props, blob_rows
        )

        async def write() -> None:
            async with self._graph.atransaction() as conn:
                await conn.pipeline_batch(
                    [(one.as_string(conn), params) for one, params in statements]
                )

        await self._graph.amerging(write)
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
            await self._graph.amerging(
                lambda: self._graph.aquery(
                    self._put_writes_cypher(), {"writes": Jsonb(rows)}
                )
            )

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
        statement, params = self._select_named_blobs(
            row["thread_id"], row["checkpoint_ns"], self._versions_of(row)
        )
        blob_rows = await self._graph.aquery(statement, params)
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
        """Async sibling of :meth:`list`."""
        scope = self._list_scope(config, before)
        rows = await self._graph.aquery(
            self._select_checkpoint_cypher(**scope.checkpoint_args(limit, filter)),
            scope.params,
        )
        blob_statement, blob_params = self._blobs_for_rows(rows)
        blob_rows = (
            await self._graph.aquery(blob_statement, blob_params)
            if blob_statement
            else []
        )
        write_statement, write_params = self._writes_for_rows(rows)
        writes_by_ckpt = self._group_writes(
            await self._graph.aquery(write_statement, write_params)
            if write_statement
            else []
        )
        for tup in self._assemble(rows, blob_rows, writes_by_ckpt, filter, limit):
            yield tup

    async def adelete_thread(self, thread_id: str) -> None:
        async with self._graph.atransaction():
            for label in (_CHECKPOINT_LABEL, _BLOB_LABEL, _WRITE_LABEL):
                await self._graph.aquery(
                    self._delete_label_cypher(label), {"tid": thread_id}
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
        """Async sibling of :meth:`copy_thread`, refusing an occupied target for the
        same reason."""
        held = (
            await self._graph.aquery(
                self._thread_is_occupied_cypher(), {"tid": target_thread_id}
            )
        )[0]["c"]
        if held:
            raise ValueError(self._occupied_message(target_thread_id, held))
        params = {"src": source_thread_id, "tgt": target_thread_id}
        async with self._graph.atransaction():
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
        base = {"tid": thread_id, "ns": checkpoint_ns}
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
                {**base, "hi": cursor},
            )
            if not rows:
                break
            writes = await self._graph.aquery(
                self._select_writes_window_cypher(),
                {**base, "lo": rows[-1]["checkpoint_id"], "hi": cursor},
            )
            writes_by_ckpt = self._group_writes_by_id(writes)
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
            "tid": thread_id,
            "ns": checkpoint_ns,
        }
        if cid is not None:
            p["cid"] = cid
        if before is not None:
            p["before"] = before
        return p


# Alias provided for naming convenience. The class implements both sync and async
# methods, so the alias is the same class.
AsyncAgensSaver = AgensSaver

__all__ = ["AgensSaver", "AsyncAgensSaver"]
