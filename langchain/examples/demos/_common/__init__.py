"""Shared infrastructure for the langchain-agensgraph demos.

    config    — .env + AgensGraph connection (conf/url)
    console   — section/sub/timer/table pretty-printing
    datautil  — Hugging Face streaming + .data cache + env knobs
    models    — OpenAI embeddings + chat model (remote only, no local models)
    agens     — make_engine/make_graph/make_vector (one shared pool)
"""

from . import agens, config, console, datautil, models  # noqa: F401
