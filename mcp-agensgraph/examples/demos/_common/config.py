"""Connection config for the MCP demos.

These demos talk to a live AgensGraph (the MCP servers are pure tools — there is no
LLM and no API key). Connection comes from ``AGENS_*`` env vars (a ``.env`` in the
demos root is loaded if present), defaulting to the local dev instance.
"""

from __future__ import annotations

import getpass
import os
import pathlib

try:
    from dotenv import load_dotenv

    load_dotenv(pathlib.Path(__file__).resolve().parent.parent / ".env")
except Exception:  # pragma: no cover - dotenv optional
    pass

HOST = os.getenv("AGENS_HOST", os.getenv("AGENSGRAPH_HOST", "127.0.0.1"))
PORT = os.getenv("AGENS_PORT", os.getenv("AGENSGRAPH_PORT", "55432"))
USER = os.getenv("AGENS_USER", os.getenv("AGENSGRAPH_USERNAME", getpass.getuser()))
PASSWORD = os.getenv("AGENS_PASSWORD", os.getenv("AGENSGRAPH_PASSWORD", ""))

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / ".data"


def dsn(database: str) -> str:
    """Build a DSN for ``database`` via the shared core's builder."""
    from mcp_agensgraph_common.connection import build_dsn

    return build_dsn(f"postgresql://{HOST}:{PORT}", USER, PASSWORD, database)


def role_can_run_programs(database: str) -> bool:
    """Whether the demo's role could run a command on the server's host through ``COPY``.

    A server started as such a role refuses to serve, because a read-only tool would not be a
    boundary for it -- ``COPY ... TO PROGRAM`` takes rows out rather than putting any in, so a
    read-only transaction has no write to refuse. The demos connect to a local dev instance as
    its owner, which is a superuser, so the refusal is the normal case here rather than the
    exception, and a demo that did not say so simply died on the first real server process.
    """
    import psycopg

    with psycopg.connect(dsn(database), autocommit=True) as conn:
        row = conn.execute(
            "SELECT rolsuper OR pg_has_role(current_user, 'pg_execute_server_program', "
            "'member') FROM pg_roles WHERE rolname = current_user"
        ).fetchone()
    return bool(row and row[0])


def server_env(database: str, graphname: str) -> dict[str, str]:
    """Env vars to launch an MCP server (stdio/HTTP) against ``database``/``graphname``.

    Accepting the privileged role out loud when the demo holds one, which is what the server
    asks for. Nothing is loosened for a role that does not hold it.
    """
    env = {
        "AGENSGRAPH_URL": f"postgresql://{HOST}:{PORT}",
        "AGENSGRAPH_USERNAME": USER,
        "AGENSGRAPH_PASSWORD": PASSWORD,
        "AGENSGRAPH_DATABASE": database,
        "AGENSGRAPH_GRAPHNAME": graphname,
    }
    accepted = {"AGENSGRAPH_ALLOW_SERVER_PROGRAMS": "true"} if role_can_run_programs(database) else {}
    return {k: v for k, v in env.items() if v != ""} | {"AGENSGRAPH_PASSWORD": PASSWORD} | accepted


def ensure_db(database: str) -> None:
    """Create ``database`` if it does not already exist (the server creates the graph)."""
    import psycopg

    with psycopg.connect(dsn("postgres"), autocommit=True) as conn:
        exists = conn.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s", (database,)
        ).fetchone()
        if not exists:
            conn.execute(f'CREATE DATABASE "{database}"')


def graph_exists(database: str, graphname: str) -> bool:
    """True if ``graphname`` exists in ``database`` (used to gate the read-only scale demo)."""
    import psycopg

    try:
        with psycopg.connect(dsn(database), autocommit=True) as conn:
            row = conn.execute(
                "SELECT 1 FROM ag_graph WHERE graphname = %s", (graphname,)
            ).fetchone()
            return row is not None
    except psycopg.Error:
        return False
