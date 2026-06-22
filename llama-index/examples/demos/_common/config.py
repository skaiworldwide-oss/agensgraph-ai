"""Connection + credential config for the demos, read from the environment.

A ``.env`` file living in the demos root (``examples/demos/.env``) is loaded
automatically; only ``OPENAI_API_KEY`` is required there. The AgensGraph
connection has sensible defaults for the local dev instance (trust auth on
``localhost:55432``, database ``llamaindex_demos``), all overridable via:

    AGENS_DB, AGENS_USER, AGENS_PASSWORD,
    AGENS_HOST (default localhost), AGENS_PORT (default 55432)

or a single ``AGENS_URL`` (postgresql://user:pwd@host:port/dbname), which takes
precedence when set.

These are the env-var names the integration's own tests use. For convenience
(so one ``.env`` can serve both the langchain and llama-index demos) the
``AGENSGRAPH_*`` names are accepted as a fallback.
"""

from __future__ import annotations

import getpass
import os
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

# Load examples/demos/.env (this file is examples/demos/_common/config.py).
DEMOS_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(DEMOS_ROOT / ".env")

DEFAULT_DB = "llamaindex_demos"
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 55432


def _env(name: str, default: str | None = None) -> str | None:
    """Read ``AGENS_<name>`` (preferred) or ``AGENSGRAPH_<name>`` (fallback).

    A set-but-empty value counts as unset: the .env template ships blank lines
    for the user to fill (or leave for the local defaults), so "" must mean
    "use the default".
    """
    for env_name in (f"AGENS_{name}", f"AGENSGRAPH_{name}"):
        val = os.getenv(env_name)
        if val is not None and val.strip() != "":
            return val.strip()
    return default


def conf() -> Dict[str, Any]:
    """psycopg-style connection dict — what ``AgensPropertyGraphStore(conf=...)`` expects.

    Under the dev instance's trust auth no password is needed, so the
    ``password`` key is omitted unless one is explicitly provided.
    """
    c: Dict[str, Any] = {
        "dbname": _env("DB", DEFAULT_DB),
        "user": _env("USER", getpass.getuser()),
        "host": _env("HOST", DEFAULT_HOST),
        "port": int(_env("PORT", str(DEFAULT_PORT))),
    }
    password = _env("PASSWORD")
    if password:
        c["password"] = password
    return c


def url() -> str:
    """libpq URL — what ``AgensgraphVectorStore(url=...)`` / ``AgensEngine.from_url`` expect."""
    explicit = _env("URL")
    if explicit:
        return explicit
    c = conf()
    auth = c["user"]
    if c.get("password"):
        auth = f"{auth}:{c['password']}"
    return f"postgresql://{auth}@{c['host']}:{c['port']}/{c['dbname']}"


def require_openai_key() -> None:
    """Fail fast with a clear message if the OpenAI key is missing."""
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY is not set. Copy examples/demos/.env.example to "
            "examples/demos/.env and add your key (used for all embeddings + LLM "
            "calls — nothing runs locally)."
        )
