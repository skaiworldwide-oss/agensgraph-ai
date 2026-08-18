"""Shared configuration parsing for the AgensGraph MCP servers.

Merges CLI arguments (highest priority) with environment variables, falling back
to documented defaults. Composable so each server pulls only the sections it needs:

- ``connection_config``  — db_url / username / password / database / graphname
- ``transport_config``   — namespace / transport / host / port / path / origins / hosts
- ``read_controls``      — read_timeout / token_limit / read_only  (cypher only)
- ``pool_config``        — pool_min_size / pool_max_size

Canonical environment variables (standardized across all servers):

    AGENSGRAPH_URL            AGENSGRAPH_USERNAME   AGENSGRAPH_PASSWORD
    AGENSGRAPH_DATABASE       AGENSGRAPH_GRAPHNAME  AGENSGRAPH_NAMESPACE
    AGENSGRAPH_TRANSPORT      AGENSGRAPH_MCP_SERVER_HOST / _PORT / _PATH
    AGENSGRAPH_MCP_SERVER_ALLOW_ORIGINS / _ALLOWED_HOSTS
    AGENSGRAPH_READ_TIMEOUT   AGENSGRAPH_RESPONSE_TOKEN_LIMIT   AGENSGRAPH_READ_ONLY
    AGENSGRAPH_POOL_MIN_SIZE  AGENSGRAPH_POOL_MAX_SIZE
    AGENSGRAPH_ALLOW_SERVER_PROGRAMS
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Any, Optional, Union

logger = logging.getLogger("mcp_agensgraph_common")


def parse_boolean_safely(value: Union[str, bool]) -> bool:
    """Parse a string/bool to bool with strict ``true``/``false`` validation."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    raise ValueError(f"Invalid boolean value: {value!r}. Must be 'true' or 'false'")


def format_namespace(namespace: str) -> str:
    """Format a tool namespace with a trailing dash if non-empty."""
    if not namespace:
        return ""
    return namespace if namespace.endswith("-") else f"{namespace}-"


def _pick(arg: Any, *env_vars: str, default: Any = None) -> Any:
    """CLI arg (if not None) → first set env var → default."""
    if arg is not None:
        return arg
    for name in env_vars:
        val = os.getenv(name)
        if val is not None:
            return val
    return default


def _split_csv(value: Optional[str]) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def connection_config(
    args: argparse.Namespace, *, default_graphname: str = "agens"
) -> dict[str, Any]:
    """Connection settings for DB-backed servers (cypher, memory).

    The password has no default. A default one is a credential in the source: it was ``agens``,
    which is the packaged instance's, so a server pointed at a machine running one connected as
    it without anybody choosing to. Left unset, nothing is sent, and libpq resolves it the way
    it resolves everything else -- ``PGPASSWORD``, ``.pgpass``, ``PGSERVICE``, or an
    authentication method that needs no password at all, none of which was reachable while a
    guess was always sent ahead of them.
    """
    cfg: dict[str, Any] = {}
    cfg["db_url"] = _pick(
        getattr(args, "db_url", None),
        "AGENSGRAPH_URL",
        "AGENSGRAPH_URI",
        default="postgresql://localhost:5432",
    )
    cfg["username"] = _pick(
        getattr(args, "username", None), "AGENSGRAPH_USERNAME", default="agens"
    )
    cfg["password"] = _pick(
        getattr(args, "password", None), "AGENSGRAPH_PASSWORD", default=""
    )
    cfg["database"] = _pick(
        getattr(args, "database", None), "AGENSGRAPH_DATABASE", "AGENSGRAPH_DB",
        default="agens",
    )
    cfg["graphname"] = _pick(
        getattr(args, "graphname", None),
        "AGENSGRAPH_GRAPHNAME",
        "AGENSGRAPH_GRAPH_NAME",
        default=default_graphname,
    )
    return cfg


def transport_config(args: argparse.Namespace) -> dict[str, Any]:
    """Transport + HTTP server settings, shared by all three servers."""
    cfg: dict[str, Any] = {}
    cfg["namespace"] = _pick(
        getattr(args, "namespace", None), "AGENSGRAPH_NAMESPACE", default=""
    )
    transport = _pick(
        getattr(args, "transport", None), "AGENSGRAPH_TRANSPORT", default="stdio"
    )
    cfg["transport"] = transport
    is_stdio = transport == "stdio"

    host = _pick(getattr(args, "server_host", None), "AGENSGRAPH_MCP_SERVER_HOST")
    cfg["host"] = host if host is not None else (None if is_stdio else "127.0.0.1")

    port = _pick(getattr(args, "server_port", None), "AGENSGRAPH_MCP_SERVER_PORT")
    if port is not None:
        cfg["port"] = int(port)
    else:
        cfg["port"] = None if is_stdio else 8000

    path = _pick(getattr(args, "server_path", None), "AGENSGRAPH_MCP_SERVER_PATH")
    cfg["path"] = path if path is not None else (None if is_stdio else "/mcp/")

    origins = getattr(args, "allow_origins", None)
    if origins is not None:
        cfg["allow_origins"] = _split_csv(origins)
    else:
        cfg["allow_origins"] = _split_csv(
            os.getenv("AGENSGRAPH_MCP_SERVER_ALLOW_ORIGINS")
        )

    hosts = getattr(args, "allowed_hosts", None)
    if hosts is not None:
        cfg["allowed_hosts"] = _split_csv(hosts)
    elif os.getenv("AGENSGRAPH_MCP_SERVER_ALLOWED_HOSTS") is not None:
        cfg["allowed_hosts"] = _split_csv(
            os.getenv("AGENSGRAPH_MCP_SERVER_ALLOWED_HOSTS")
        )
    else:
        # Secure default: only localhost when not explicitly configured.
        cfg["allowed_hosts"] = ["localhost", "127.0.0.1"]

    if not is_stdio and host is None and getattr(args, "server_host", None) is None:
        logger.info("No server host provided for %s transport; using 127.0.0.1", transport)
    return cfg


# What one tool result may cost the model reading it. On by default: an unbounded read of a
# graph holding text returns whatever the rows hold, and `RETURN n.title, n.abstract` over a
# thousand papers came to 874,043 bytes -- around 217,000 tokens -- in a single result.
# ``0`` turns the bound off for a caller who means to take everything.
DEFAULT_TOKEN_LIMIT = 10_000

# How long a read may take. It becomes the connection's own `statement_timeout`, so every read
# on it is bounded before the first one is sent.
DEFAULT_READ_TIMEOUT = 30

# How many connections the pool holds, and how many it may ever open.
DEFAULT_POOL_MIN_SIZE = 4
DEFAULT_POOL_MAX_SIZE = 16


def read_controls(args: argparse.Namespace) -> dict[str, Any]:
    """Read-query controls (cypher server): timeout, token limit, read-only."""
    cfg: dict[str, Any] = {}

    read_timeout = _pick(getattr(args, "read_timeout", None), "AGENSGRAPH_READ_TIMEOUT")
    if read_timeout is None:
        cfg["read_timeout"] = DEFAULT_READ_TIMEOUT
    else:
        try:
            seconds = int(read_timeout)
        except (TypeError, ValueError):
            logger.warning(
                "Read timeout %r is not a number of seconds; using %d",
                read_timeout,
                DEFAULT_READ_TIMEOUT,
            )
            seconds = DEFAULT_READ_TIMEOUT
        if seconds <= 0:
            # PostgreSQL reads a `statement_timeout` of nought as no limit at all, so this is
            # the one wrong value that looks like the strictest one. Measured: a read timeout of
            # 0 let a two-second `pg_sleep` run to completion, and one of -5 failed every call
            # with 22023. A server is not told to bound a read by being given no bound.
            raise ValueError(
                f"a read timeout is how many seconds a read may take, so it has to be more "
                f"than nothing; got {seconds}. PostgreSQL reads nought as no limit at all, "
                f"which is the opposite of what asking for one means."
            )
        cfg["read_timeout"] = seconds

    token_limit = _pick(
        getattr(args, "token_limit", None),
        "AGENSGRAPH_RESPONSE_TOKEN_LIMIT",
        default=DEFAULT_TOKEN_LIMIT,
    )
    try:
        limit = int(token_limit)
    except (TypeError, ValueError):
        logger.warning(
            "Invalid response token limit %r; using default %d",
            token_limit,
            DEFAULT_TOKEN_LIMIT,
        )
        limit = DEFAULT_TOKEN_LIMIT
    cfg["token_limit"] = limit if limit > 0 else None

    if getattr(args, "read_only", False):
        cfg["read_only"] = True
    else:
        env_ro = os.getenv("AGENSGRAPH_READ_ONLY")
        cfg["read_only"] = parse_boolean_safely(env_ro) if env_ro is not None else False

    cfg.update(server_program_control(args))
    cfg.update(graph_ddl_control(args))

    return cfg


def pool_config(args: argparse.Namespace) -> dict[str, Any]:
    """How many connections the server holds, and how many it may ever open.

    Both ends, because a pool given only the lower one is that wide at the top as well: four
    connections however much arrives at once, and a trivial read measured waiting 7.77 s behind
    eighteen slow ones. What the right numbers are depends on the database server, so they are
    settings rather than a guess written into the code.
    """
    cfg: dict[str, Any] = {}
    for key, env, default in (
        ("pool_min_size", "AGENSGRAPH_POOL_MIN_SIZE", DEFAULT_POOL_MIN_SIZE),
        ("pool_max_size", "AGENSGRAPH_POOL_MAX_SIZE", DEFAULT_POOL_MAX_SIZE),
    ):
        given = _pick(getattr(args, key, None), env, default=default)
        try:
            size = int(given)
        except (TypeError, ValueError):
            logger.warning("%s is not a number of connections; using %d", env, default)
            size = default
        cfg[key] = max(1, size)
    if cfg["pool_max_size"] < cfg["pool_min_size"]:
        raise ValueError(
            f"a pool cannot hold {cfg['pool_min_size']} connections and open at most "
            f"{cfg['pool_max_size']}"
        )
    return cfg


def graph_ddl_control(args: argparse.Namespace) -> dict[str, Any]:
    """Whether the tools may change a graph's shape as well as its contents.

    Off unless it is asked for. A tool that ingests data needs to write elements, not to drop a
    graph, and the two arrive through the same free-text parameter.
    """
    if getattr(args, "allow_graph_ddl", False):
        return {"allow_graph_ddl": True}
    env = os.getenv("AGENSGRAPH_ALLOW_GRAPH_DDL")
    return {"allow_graph_ddl": parse_boolean_safely(env) if env is not None else False}


def graph_adoption_control(args: argparse.Namespace) -> dict[str, Any]:
    """Whether the operator has said a graph someone else filled holds this server's data.

    Off unless it is asked for. Starting up folds elements sharing a name into one and moves
    relationships onto another label, which is justified by owning the data; a graph already
    holding those labels is not distinguishable from this server's by reading it, so the
    operator settles it.
    """
    asked = getattr(args, "adopt_existing_graph", False)
    if asked:
        return {"adopt_existing_graph": True}
    env = os.getenv("AGENSGRAPH_ADOPT_EXISTING_GRAPH")
    return {"adopt_existing_graph": parse_boolean_safely(env) if env is not None else False}


def server_program_control(args: argparse.Namespace) -> dict[str, Any]:
    """Whether the operator has accepted a role that can run a command on the server's host.

    Off unless it is asked for. `COPY ... TO PROGRAM` is not stopped by a read-only transaction,
    so a server advertising a read-only tool while connected as such a role is making a claim it
    cannot keep -- it refuses to start instead, and this is how that is overridden deliberately.
    """
    asked = getattr(args, "allow_server_programs", False)
    if asked:
        return {"allow_server_programs": True}
    env = os.getenv("AGENSGRAPH_ALLOW_SERVER_PROGRAMS")
    return {"allow_server_programs": parse_boolean_safely(env) if env is not None else False}
