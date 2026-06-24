"""Connection config for the MCP demos.

These demos talk to a live AgensGraph (the MCP servers are pure tools — there is no
LLM and no API key). Connection comes from ``AGENS_*`` env vars (a ``.env`` in the
demos root is loaded if present), defaulting to the local dev instance.
"""

from __future__ import annotations

import os
import pathlib

try:
    from dotenv import load_dotenv

    load_dotenv(pathlib.Path(__file__).resolve().parent.parent / ".env")
except Exception:  # pragma: no cover - dotenv optional
    pass

HOST = os.getenv("AGENS_HOST", os.getenv("AGENSGRAPH_HOST", "127.0.0.1"))
PORT = os.getenv("AGENS_PORT", os.getenv("AGENSGRAPH_PORT", "55432"))
USER = os.getenv("AGENS_USER", os.getenv("AGENSGRAPH_USERNAME", "taha-linux"))
PASSWORD = os.getenv("AGENS_PASSWORD", os.getenv("AGENSGRAPH_PASSWORD", ""))

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / ".data"


def dsn(database: str) -> str:
    """Build a DSN for ``database`` via the shared core's builder."""
    from mcp_agensgraph_common.connection import build_dsn

    return build_dsn(f"postgresql://{HOST}:{PORT}", USER, PASSWORD, database)


def server_env(database: str, graphname: str) -> dict[str, str]:
    """Env vars to launch an MCP server (stdio/HTTP) against ``database``/``graphname``."""
    env = {
        "AGENSGRAPH_URL": f"postgresql://{HOST}:{PORT}",
        "AGENSGRAPH_USERNAME": USER,
        "AGENSGRAPH_PASSWORD": PASSWORD,
        "AGENSGRAPH_DATABASE": database,
        "AGENSGRAPH_GRAPHNAME": graphname,
    }
    return {k: v for k, v in env.items() if v != ""} | {"AGENSGRAPH_PASSWORD": PASSWORD}


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
