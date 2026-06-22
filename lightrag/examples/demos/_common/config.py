"""Connection + credential config for the demos.

The lightrag-agensgraph storages read their connection straight from the
environment (``AGENSGRAPH_DB/USER/PASSWORD/HOST/PORT/WORKSPACE``) at construction
time, so this module's job is to load a ``.env`` and then **set those variables**
before any LightRAG/storage is built.

A ``.env`` living in the demos root (``examples/demos/.env``) is loaded
automatically; only ``OPENAI_API_KEY`` is required there. The AgensGraph
connection has sensible defaults for the local dev instance (trust auth on
``localhost:55432``). Settings are read from the integration's own
``AGENSGRAPH_*`` names, with the ``AGENS_*`` names accepted as a fallback so a
single ``.env`` can serve every integration's demos.

Each demo owns its **own database** (``lightrag_wiki``, ``lightrag_news``, …) so
their knowledge graphs never collide — the graph store isolates by graph name,
and LightRAG always uses the same graph name (``chunk_entity_relation``) within a
database. ``ensure_db`` creates a demo's database + the ``vector`` extension on
first run.
"""

from __future__ import annotations

import getpass
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load examples/demos/.env (this file is examples/demos/_common/config.py).
DEMOS_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(DEMOS_ROOT / ".env")

DEFAULT_HOST = "localhost"
DEFAULT_PORT = "55432"


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    """Read ``AGENSGRAPH_<name>`` (preferred) or ``AGENS_<name>`` (fallback).

    A set-but-empty value counts as unset: the .env template ships blank lines
    for the user to fill (or leave for the local defaults), so "" means "use the
    default".
    """
    for env_name in (f"AGENSGRAPH_{name}", f"AGENS_{name}"):
        val = os.getenv(env_name)
        if val is not None and val.strip() != "":
            return val.strip()
    return default


def user() -> str:
    return _env("USER", getpass.getuser())


def password() -> Optional[str]:
    return _env("PASSWORD")  # None under trust auth


def host() -> str:
    return _env("HOST", DEFAULT_HOST)


def port() -> str:
    return _env("PORT", DEFAULT_PORT)


def apply_env(db: str, *, workspace: str = "") -> None:
    """Export the ``AGENSGRAPH_*`` variables the storages read at construction.

    Call this once, before building a LightRAG instance. ``db`` selects the
    database (each demo uses its own); ``workspace`` partitions the relational
    stores (vector / KV / doc-status) within that database.
    """
    os.environ["AGENSGRAPH_DB"] = db
    os.environ["AGENSGRAPH_USER"] = user()
    os.environ["AGENSGRAPH_HOST"] = host()
    os.environ["AGENSGRAPH_PORT"] = port()
    os.environ["AGENSGRAPH_WORKSPACE"] = workspace
    pwd = password()
    if pwd:
        os.environ["AGENSGRAPH_PASSWORD"] = pwd
    else:
        # psycopg/libpq under trust auth is happy with an empty password; set one
        # so the integration's `password='{...}'` conninfo fragment is well-formed.
        os.environ.setdefault("AGENSGRAPH_PASSWORD", "")


def _conninfo(dbname: str) -> str:
    pwd = password() or ""
    return (
        f"dbname='{dbname}' user='{user()}' password='{pwd}' "
        f"host='{host()}' port={port()}"
    )


def ensure_db(db: str) -> None:
    """Create ``db`` and its ``vector`` extension if they don't exist (psycopg).

    ``psql`` isn't on PATH in this environment, so database setup goes through
    psycopg. ``CREATE DATABASE`` can't run inside a transaction, hence autocommit.
    """
    import psycopg

    with psycopg.connect(_conninfo("postgres"), autocommit=True) as conn:
        exists = conn.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s", (db,)
        ).fetchone()
        if not exists:
            conn.execute(f'CREATE DATABASE "{db}"')
    with psycopg.connect(_conninfo(db), autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")


def require_openai_key() -> None:
    """Fail fast with a clear message if the OpenAI key is missing."""
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY is not set. Copy examples/demos/.env.example to "
            "examples/demos/.env and add your key (used for all embeddings + LLM "
            "calls — nothing runs locally)."
        )
