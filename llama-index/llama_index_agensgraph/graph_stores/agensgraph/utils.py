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

from typing import Any, Dict, Union, List
import psycopg
from psycopg import sql
from functools import wraps

from agensgraph.errors import safe_message


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
    
def get_graph_id(curs, graph_name: str):
    graph_id_query = (
        """SELECT oid as graphid FROM ag_graph WHERE graphname = %(graph_name)s;"""
    )
    execute_query(
        curs, graph_id_query, {"graph_name": graph_name}, "Error getting graph id"
    )
    return curs.fetchone().graphid if curs.rowcount > 0 else None

def create_graph(curs, graph_name: str):
    create_statement = sql.SQL("""
                    CREATE GRAPH {};
                """).format(sql.Identifier(graph_name))
    execute_query(curs, create_statement, error_message="Error creating graph")

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
        try:
            import psycopg
        except ImportError:
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
