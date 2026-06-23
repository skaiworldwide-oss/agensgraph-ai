"""Connection + model config for the cognee demos.

cognee is configured process-globally. This module:
  1. Loads ``examples/demos/.env`` and exports the OpenAI model settings cognee
     reads (LLM + embeddings) — done at import, BEFORE any ``import cognee``, so
     cognee's settings pick them up. Import ``_common.config`` (or ``_common``)
     before importing ``cognee`` in a demo.
  2. Provides a per-demo AgensGraph DSN + ``ensure_db`` (creates the database and
     the ``vector`` extension via psycopg — ``psql`` isn't on PATH here).
  3. ``configure(db, name)`` points cognee's graph AND vector stores at one
     AgensGraph database and sets per-demo data/system directories under ``.data``.

Models: OpenAI ``gpt-4o-mini`` + ``text-embedding-3-small`` (1536-d). cognee's
default embedding is text-embedding-3-large (3072-d); we override it (cheaper, and
matches the other integrations' demos). Override again with the ``DEMO_*`` vars.
"""

from __future__ import annotations

import getpass
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

DEMOS_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = DEMOS_ROOT / ".data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
load_dotenv(DEMOS_ROOT / ".env")

DEFAULT_HOST = "localhost"
DEFAULT_PORT = "55432"
LLM_MODEL = os.getenv("DEMO_LLM_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("DEMO_EMBED_MODEL", "openai/text-embedding-3-small")
EMBED_DIM = os.getenv("DEMO_EMBED_DIM", "1536")


def _apply_model_env() -> None:
    """Export the LLM/embedding settings cognee reads from the environment.

    Must run before cognee's (pydantic-settings) configs are first instantiated,
    hence at import time. The OpenAI key feeds both the LLM and the embedder.
    """
    key = os.getenv("OPENAI_API_KEY")
    os.environ.setdefault("LLM_PROVIDER", "openai")
    os.environ.setdefault("LLM_MODEL", LLM_MODEL)
    os.environ.setdefault("EMBEDDING_PROVIDER", "openai")
    os.environ.setdefault("EMBEDDING_MODEL", EMBED_MODEL)
    os.environ.setdefault("EMBEDDING_DIMENSIONS", EMBED_DIM)
    if key:
        os.environ.setdefault("LLM_API_KEY", key)
        os.environ.setdefault("EMBEDDING_API_KEY", key)


_apply_model_env()


# ---- AgensGraph connection ----

def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    for env_name in (f"AGENSGRAPH_{name}", f"AGENS_{name}"):
        val = os.getenv(env_name)
        if val is not None and val.strip() != "":
            return val.strip()
    return default


def _user() -> str:
    return _env("USER", getpass.getuser())


def _host() -> str:
    return _env("HOST", DEFAULT_HOST)


def _port() -> str:
    return _env("PORT", DEFAULT_PORT)


def _password() -> Optional[str]:
    return _env("PASSWORD")  # None under trust auth


def dsn(db: str) -> str:
    """libpq DSN for a demo database (what cognee's agensgraph adapters expect)."""
    auth = _user()
    pwd = _password()
    if pwd:
        auth = f"{auth}:{pwd}"
    return f"postgresql://{auth}@{_host()}:{_port()}/{db}"


def ensure_db(db: str) -> None:
    """Create ``db`` and its ``vector`` extension if missing (psycopg, autocommit)."""
    import psycopg

    admin = dsn("postgres")
    with psycopg.connect(admin, autocommit=True) as conn:
        exists = conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db,)).fetchone()
        if not exists:
            conn.execute(f'CREATE DATABASE "{db}"')
    with psycopg.connect(dsn(db), autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")


# ---- cognee wiring ----

def configure(db: str, name: Optional[str] = None) -> None:
    """Point cognee's graph + vector stores at one AgensGraph database.

    ``name`` (defaults to ``db``) namespaces the local data/system directories so
    each demo keeps its own cognee metadata (SQLite) + raw docs under ``.data``.
    """
    import cognee
    import cognee_agensgraph  # noqa: F401  (registers the agensgraph adapters)

    url = dsn(db)
    cognee.config.set_graph_db_config({"graph_database_url": url, "graph_database_provider": "agensgraph"})
    cognee.config.set_vector_db_config({"vector_db_url": url, "vector_db_provider": "agensgraph"})

    key = os.getenv("OPENAI_API_KEY")
    if key:
        cognee.config.set_llm_api_key(key)

    sub = name or db
    cognee.config.data_root_directory(str(DATA_DIR / sub / "data"))
    cognee.config.system_root_directory(str(DATA_DIR / sub / "system"))


async def aprune() -> None:
    """Reset this demo's cognee state (local data + graph/vector/metadata)."""
    import cognee

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)


async def search(query_text: str, query_type, **kwargs):
    """``cognee.search`` that tolerates empty results.

    cognee 0.2.1's core raises ``IndexError`` when a search returns zero results
    (its history-logging does ``search_results[0]`` on an empty list) — in the
    upstream framework, not in this AgensGraph integration. We don't patch cognee
    here; we just guard the demo so a query with no hits prints "(no results)"
    instead of crashing.
    """
    import cognee

    try:
        return await cognee.search(query_text=query_text, query_type=query_type, **kwargs)
    except IndexError:
        return []


def quiet() -> None:
    """Silence cognee's very verbose INFO logging so demo output stays readable."""
    import logging

    logging.disable(logging.INFO)
    for n in ("langfuse", "httpx", "httpcore", "litellm", "LiteLLM"):
        logging.getLogger(n).setLevel(logging.ERROR)


def require_openai_key() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY is not set. Copy examples/demos/.env.example to "
            "examples/demos/.env and add your key (used for all embeddings + LLM "
            "calls — nothing runs locally)."
        )
