"""Connection + credential config for the demos, read from the environment.

A ``.env`` file living in the demos root (``examples/demos/.env``) is loaded
automatically; only ``OPENAI_API_KEY`` is required there. The AgensGraph
connection has sensible defaults for the local dev instance (trust auth on
``localhost:55432``, database ``agensgraph_demos``), all overridable via:

    AGENSGRAPH_DB, AGENSGRAPH_USER, AGENSGRAPH_PASSWORD,
    AGENSGRAPH_HOST (default localhost), AGENSGRAPH_PORT (default 55432)

or a single ``AGENSGRAPH_URL`` (postgresql://user:pwd@host:port/dbname), which
takes precedence when set.
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

DEFAULT_DB = "agensgraph_demos"
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 55432


def _env(name: str, default: str | None = None) -> str | None:
    """os.getenv that treats a set-but-empty value as unset.

    The .env template ships blank AGENSGRAPH_* lines for the user to fill (or
    leave for the local defaults), so "" must mean "use the default".
    """
    val = os.getenv(name)
    if val is None or val.strip() == "":
        return default
    return val.strip()


def conf() -> Dict[str, Any]:
    """psycopg-style connection dict — what ``AgensGraph(conf=...)`` expects.

    Under the dev instance's trust auth no password is needed, so the
    ``password`` key is omitted unless one is explicitly provided.
    """
    c: Dict[str, Any] = {
        "dbname": _env("AGENSGRAPH_DB", DEFAULT_DB),
        "user": _env("AGENSGRAPH_USER", getpass.getuser()),
        "host": _env("AGENSGRAPH_HOST", DEFAULT_HOST),
        "port": int(_env("AGENSGRAPH_PORT", str(DEFAULT_PORT))),
    }
    password = _env("AGENSGRAPH_PASSWORD")
    if password:
        c["password"] = password
    return c


def url() -> str:
    """libpq URL — what ``AgensgraphVector(url=...)`` / ``AgensEngine.from_url`` expect."""
    explicit = _env("AGENSGRAPH_URL")
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
