"""Connection and credential config for the demos, read from the environment.

A ``.env`` file in the demos root (``examples/demos/.env``) is loaded automatically. The
AgensGraph connection has defaults for a local dev instance (trust auth on ``localhost:55432``,
database ``agensgraph_demos``), all overridable with:

    AGENSGRAPH_DB, AGENSGRAPH_USER, AGENSGRAPH_PASSWORD,
    AGENSGRAPH_HOST (default localhost), AGENSGRAPH_PORT (default 55432)

or a single ``AGENSGRAPH_URL`` (postgresql://user:pwd@host:port/dbname), which takes precedence
when set. That role owns the demos' graphs. ``AGENSGRAPH_UNTRUSTED_URL`` is the URL of the role
model-written Cypher runs as, such as ``llm_ro`` (see the README's "Roles").

``OPENAI_API_KEY`` is needed unless ``DEMOS_FAKE=1``, which swaps in deterministic fake models.
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

    The .env template ships blank AGENSGRAPH_* lines for the user to fill (or leave for the local
    defaults), so "" means "use the default".
    """
    val = os.getenv(name)
    if val is None or val.strip() == "":
        return default
    return val.strip()


def fake() -> bool:
    """Whether ``DEMOS_FAKE=1`` asks for the fake models and the fixtures."""
    return (_env("DEMOS_FAKE") or "0").lower() in ("1", "true", "yes")


def cache_dir() -> str | None:
    """The folder ``DEMOS_CACHE_DIR`` names for the model caches, or None for no caching."""
    return _env("DEMOS_CACHE_DIR")


def conf() -> Dict[str, Any]:
    """The connection settings as psycopg keyword arguments.

    Under the dev instance's trust auth no password is needed, so the ``password`` key is left out
    unless one is given.
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
    """The libpq URL of the role that owns the demos' graphs: what ``AgensEngine`` takes."""
    explicit = _env("AGENSGRAPH_URL")
    if explicit:
        return explicit
    c = conf()
    auth = c["user"]
    if c.get("password"):
        auth = f"{auth}:{c['password']}"
    return f"postgresql://{auth}@{c['host']}:{c['port']}/{c['dbname']}"


def untrusted_url() -> str | None:
    """The libpq URL of the role model-written Cypher runs as (``AGENSGRAPH_UNTRUSTED_URL``)."""
    return _env("AGENSGRAPH_UNTRUSTED_URL")


def require_untrusted_url() -> str:
    """The untrusted role's URL, or exit saying what to set."""
    found = untrusted_url()
    if not found:
        raise SystemExit(
            "AGENSGRAPH_UNTRUSTED_URL is not set. Model-written Cypher runs as a role made for it, "
            "such as llm_ro from langchain_agensgraph.roles.render_roles: set it to that role's "
            "URL, e.g. postgresql://llm_ro:secret@localhost:55432/agensgraph_demos."
        )
    return found


def require_openai_key() -> None:
    """Exit with a clear message if the OpenAI key is missing, unless the fake models are on."""
    if fake():
        return
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY is not set. Copy examples/demos/.env.example to "
            "examples/demos/.env and add your key, or set DEMOS_FAKE=1 to run on the fake "
            "models and the fixtures."
        )
