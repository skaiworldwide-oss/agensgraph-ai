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

import hashlib
import importlib.util
import re
from functools import wraps
from typing import Any, Dict, List, Union

import psycopg
from psycopg import sql

from agensgraph.cypher import without_literals
from agensgraph.errors import safe_message
from agensgraph.introspect import MAX_IDENTIFIER
from agensgraph.vector import search_option_statements


class AgensQueryException(Exception):
    """Something the server refused, with what it said about it.

    ``detail`` and ``details`` are both read. The four places that raise this wrote the
    first and this read the second, so every failure arrived saying ``"unknown"`` and
    the server's own words were dropped on the floor -- which is why a run of a hundred
    failing statements could not be told apart from a run of one.

    The cause is kept as ``__cause__`` so a caller can ask the driver whether it is
    worth trying again.
    """

    def __init__(
        self, exception: Union[str, Dict], cause: Union[BaseException, None] = None
    ) -> None:
        if isinstance(exception, dict):
            self.message = exception.get("message", "unknown")
            self.details = exception.get("details", exception.get("detail", "unknown"))
        else:
            self.message = exception
            self.details = "unknown"
        self.cause = cause
        if cause is not None:
            self.__cause__ = cause

    def get_message(self) -> str:
        return self.message

    def get_details(self) -> Any:
        return self.details


def query_failed(query: Any, exc: BaseException) -> AgensQueryException:
    """The exception to raise for a statement the server refused.

    ``safe_message`` gives the SQLSTATE and the server's primary line and leaves off
    ``DETAIL``, which carries the row that failed -- and these results are handed to a
    language model.
    """
    return AgensQueryException(
        {
            "message": "Error executing graph query",
            "details": (
                safe_message(exc) if isinstance(exc, psycopg.Error) else str(exc)
            ),
            "query": str(query),
        },
        cause=exc,
    )
    

# What a vector search needs in order to answer the question it was asked.
#
# An HNSW scan visits about ``hnsw.ef_search`` candidates and a metadata filter is
# applied to those, so a filtered search for the ten nearest returned however many
# of the forty visited happened to pass -- measured on 20,000 elements with a
# filter keeping one in ten, between two and eight rows, every time, with no error
# and no warning. ``iterative_scan`` keeps scanning until the limit is satisfied.
#
# ``strict_order`` rather than ``relaxed_order``: both return the full ten, but
# relaxed hands them back out of distance order, and these results carry a
# similarity score a caller ranks on. Measured: 3.9 ms lossy, 6.7 ms relaxed and
# out of order, 9.5 ms strict and ordered. On an unfiltered search it costs
# nothing, because nothing makes the scan come up short.
COMPLETE_FILTERED_SEARCH = {"hnsw.iterative_scan": "strict_order"}

_APPLIED = "_agens_llama_index_search_options"


def known_search_options(conn, options: Dict[str, Any]) -> Dict[str, Any]:
    """Whichever of ``options`` this server's pgvector understands.

    ``hnsw.iterative_scan`` arrived in pgvector 0.8, and a default this package
    chose must not be what stops a store from opening against an older one.

    The version is read from the catalog rather than from ``pg_settings``, because
    pgvector registers its settings when its library is first loaded into the
    session -- on a connection that has not yet touched a vector, ``pg_settings``
    holds no ``hnsw.*`` row at all and every option would look unsupported. (The
    ``SET`` itself is accepted either way: the server keeps a prefixed name it does
    not recognise as a placeholder and applies it once the library arrives. What
    would surface later, at the first vector query, is a value that version
    rejects.)
    """
    if not options:
        return {}
    row = conn.execute(
        "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
    ).fetchone()
    if row is None:
        return {}
    try:
        major, minor = (int(part) for part in str(row[0]).split(".")[:2])
    except ValueError:
        return dict(options)
    if (major, minor) >= (0, 8):
        return dict(options)
    return {
        name: value for name, value in options.items()
        if not name.endswith(".iterative_scan")
    }


def apply_search_options(conn, options: Dict[str, Any]) -> None:
    """Put the vector-search settings on this connection.

    ``SET LOCAL`` is the natural spelling and the wrong one here: a pooled
    connection is in autocommit, there is no transaction for it to last for, and
    the settings silently did nothing on every pooled borrow. A plain ``SET`` is
    session-level, which suits a connection this store borrows again and again --
    and it is re-issued only when the connection is not already carrying exactly
    these settings, so a pool shared with something that wants different ones
    cannot be left holding the wrong ones.
    """
    if not options or getattr(conn, _APPLIED, None) == options:
        return
    for statement in search_option_statements(options, local=False):
        conn.execute(statement)
    setattr(conn, _APPLIED, dict(options))


async def aapply_search_options(conn, options: Dict[str, Any]) -> None:
    """Async sibling of :func:`apply_search_options`."""
    if not options or getattr(conn, _APPLIED, None) == options:
        return
    for statement in search_option_statements(options, local=False):
        await conn.execute(statement)
    setattr(conn, _APPLIED, dict(options))


def bounded_name(*parts: str) -> str:
    """A name for a constraint or an index that stays distinct after truncation.

    An identifier is 63 bytes and a label can be longer -- these are whatever an LLM
    called an entity type. Two labels agreeing for the first 63 bytes would be given
    one name, and the second would be read as already there and skipped, leaving that
    label without the uniqueness that stops two writers making two of one node.
    """
    name = "_".join(parts)
    encoded = name.encode()
    if len(encoded) <= MAX_IDENTIFIER:
        return name
    digest = hashlib.blake2b(encoded, digest_size=8).hexdigest()
    head = encoded[: MAX_IDENTIFIER - len(digest) - 1].decode("utf-8", "ignore")
    return f"{head}_{digest}"


COPY_LEADS = re.compile(r"(?:\A|;)\s*COPY\b", re.IGNORECASE)


def check_no_copy(statement: str) -> None:
    """Refuse ``COPY``, which cannot be run here without breaking the connection.

    ``COPY`` needs the copy protocol. psycopg refuses it in ``execute()``, but only
    after sending it, so the connection is left mid-copy with no way out: after
    ``COPY (SELECT 1) TO STDOUT`` every later statement failed with "another
    command is already in progress", and rollback, cancel and rollback again all
    failed too. With a pool, that connection is handed to the next caller.

    Literals and comments are blanked first, so a property or string containing the
    word is not mistaken for the statement.
    """
    if COPY_LEADS.search(without_literals(str(statement))):
        raise ValueError(
            "COPY cannot be run through this method: it needs the copy protocol, "
            "and sending it here leaves the connection unusable. Use the "
            "connection's own copy() -- reachable as the store's `client`."
        )


def lost_the_creation_race(exc: BaseException) -> bool:
    """Whether the error means another process created the object first.

    ``IF NOT EXISTS`` checks and then creates, which is two steps, so two stores
    opening on the same graph both see it missing and the slower one fails. Of
    eight stores opened at once, two survived.

    Trying again would not help, and there is nothing to fix: the object exists,
    which is what was asked for. The cause is checked as well as the error itself,
    since statements here raise wrapped.
    """
    for candidate in (exc, getattr(exc, "__cause__", None)):
        if isinstance(
            candidate,
            (
                psycopg.errors.UniqueViolation,
                psycopg.errors.DuplicateSchema,
                psycopg.errors.DuplicateTable,
                psycopg.errors.DuplicateObject,
                psycopg.errors.DuplicateColumn,
                psycopg.errors.DuplicateFunction,
            ),
        ):
            return True
    return False


def get_graph_id(curs, graph_name: str):
    graph_id_query = (
        """SELECT oid as graphid FROM ag_graph WHERE graphname = %(graph_name)s;"""
    )
    execute_query(
        curs, graph_id_query, {"graph_name": graph_name}, "Error getting graph id"
    )
    return curs.fetchone().graphid if curs.rowcount > 0 else None

def create_graph(curs, graph_name: str) -> None:
    """Make the graph, and be content if somebody else just did.

    ``IF NOT EXISTS`` is not the whole answer: it reads the catalog and then
    creates the schema, which is two steps, so several processes starting
    together each found it missing and each created it. One won and the rest were
    told "duplicate key value violates unique constraint
    pg_namespace_nspname_index" -- measured, ten of fifteen runs of the
    concurrency test.

    Losing that race means the graph is there, which is all the caller wanted, so
    it is not an error. Anything else is.
    """
    create_statement = sql.SQL("""
                    CREATE GRAPH IF NOT EXISTS {};
                """).format(sql.Identifier(graph_name))
    try:
        execute_query(curs, create_statement, error_message="Error creating graph")
    except AgensQueryException as failure:
        if not lost_the_creation_race(failure):
            raise

def set_graph_path(curs, graph_name: str):
    graph_path = sql.SQL("SET graph_path = {};").format(
        sql.Identifier(graph_name)
    )
    execute_query(curs, graph_path)

def execute_query(curs, query, params={}, error_message = "Error executing query"):
    try:
        curs.execute(query, params)
    except psycopg.Error as e:
        # A refused statement leaves its transaction able to run nothing else, so
        # without this every later statement on the connection reports the abort
        # instead of what actually went wrong -- one racing label refusal turned into
        # 72 failures in a run of 200.
        try:
            curs.connection.rollback()
        except psycopg.Error:
            pass
        raise AgensQueryException(
            {"message": error_message, "details": safe_message(e)}, cause=e
        ) from e

def require_psycopg(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if importlib.util.find_spec("psycopg") is None:
            raise ImportError(
                "Could not import psycopg python package. "
                "Please install it with `pip install psycopg`."
            )
        return func(*args, **kwargs)
    return wrapper

def format_triples(triples: List[Dict[str, str]]) -> List[str]:
    """
    Convert a list of relationships from dictionaries to formatted strings
    to be better readable by an llm

    Args:
        triples (List[Dict[str,str]]): a list relationships in the form
            {'start':<from_label>, 'type':<edge_label>, 'end':<from_label>}

    Returns:
        List[str]: a list of relationships in the form
            "(:"<from_label>")-[:"<edge_label>"]->(:"<to_label>")"
    """
    triple_template = '(:"{start}")-[:"{type}"]->(:"{end}")'
    triple_schema = [triple_template.format(**triple) for triple in triples]

    return triple_schema
