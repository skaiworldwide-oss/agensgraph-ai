"""Shared core for the AgensGraph MCP servers.

Re-exports the helpers the three servers build on: configuration parsing,
connection/pool lifecycle + query execution, vertex/edge result parsing, query-safety helpers,
and the transport bootstrap.
"""

from .results import (
    EDGE_REGEX,
    VERTEX_REGEX,
    record_to_dict,
    truncate_to_tokens,
    value_sanitize,
)
from .config import (
    connection_config,
    format_namespace,
    parse_boolean_safely,
    read_controls,
    transport_config,
)
from .safety import (
    is_write_query,
    quote_identifiers,
    quote_label,
    strip_comments_and_strings,
)
from .transport import build_middleware, run_server

__all__ = [
    # config
    "connection_config",
    "transport_config",
    "read_controls",
    "format_namespace",
    "parse_boolean_safely",
    # safety
    "quote_identifiers",
    "quote_label",
    "is_write_query",
    "strip_comments_and_strings",
    # results (vertex/edge parsing + shaping)
    "record_to_dict",
    "value_sanitize",
    "truncate_to_tokens",
    "VERTEX_REGEX",
    "EDGE_REGEX",
    # transport
    "run_server",
    "build_middleware",
]

# Connection helpers are imported lazily (require the `db` extra / psycopg) so the
# DB-less data-modeling server can use config/transport without psycopg installed.
def __getattr__(name: str):  # pragma: no cover - thin lazy shim
    if name in {
        "build_dsn",
        "create_pool",
        "get_pool_connection",
        "ensure_graph",
        "run_query",
    }:
        from . import connection

        return getattr(connection, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
