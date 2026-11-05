"""Shared infrastructure for the langchain-agensgraph demos.

    config    — .env, the AgensGraph URLs (owner and untrusted role), DEMOS_FAKE
    console   — section/sub/timer/table pretty-printing
    datautil  — Hugging Face streaming, the fixtures, env knobs
    models    — OpenAI embeddings + chat model, or the fakes with DEMOS_FAKE=1
    fake      — the deterministic fake chat model
    agens     — get_engine/make_graph/make_vector/grant on one shared AgensEngine
"""

from . import agens, config, console, datautil, models  # noqa: F401
