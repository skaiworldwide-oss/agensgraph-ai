"""Model providers for the demos: OpenAI, or deterministic fakes with ``DEMOS_FAKE=1``.

Embeddings and chat completions go through the OpenAI API, so no local model is loaded. Override
the models with DEMO_EMBED_MODEL / DEMO_LLM_MODEL. With ``DEMOS_FAKE=1`` the embeddings are
langchain_core's ``DeterministicFakeEmbedding`` and the chat model is
:class:`_common.fake.FakeChatModel`, so nothing needs an API key and every run answers the same.
With ``DEMOS_CACHE_DIR`` set, embeddings and answers are kept on disk (:mod:`_common.cache`), so
a second run, on the same server or another, pays only for what the first did not ask.
"""

from __future__ import annotations

import json
import os
from typing import Any

from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings
from langchain_core.language_models import BaseChatModel

from . import config

# text-embedding-3-small: 1536-dim, $0.02 / 1M tokens.
DEFAULT_EMBED_MODEL = "text-embedding-3-small"
# gpt-6-luna: $0.10 / 1M input and $0.50 / 1M output tokens; it takes no temperature.
DEFAULT_LLM_MODEL = "gpt-6-luna"
# The fake embeddings' dimension: small, since they carry no meaning.
FAKE_DIMS = 64


def get_embeddings(model: str | None = None, **kwargs: Any) -> Embeddings:
    if config.fake():
        return DeterministicFakeEmbedding(size=FAKE_DIMS)
    config.require_openai_key()
    from langchain_openai import OpenAIEmbeddings

    name = model or os.getenv("DEMO_EMBED_MODEL", DEFAULT_EMBED_MODEL)
    embeddings = OpenAIEmbeddings(model=name, **kwargs)
    folder = config.cache_dir()
    if folder is None:
        return embeddings
    from .cache import CachedEmbeddings

    # Settings such as dimensions change the vectors, so they are part of what the cache keys by.
    settings = f" {json.dumps(kwargs, sort_keys=True, default=str)}" if kwargs else ""
    return CachedEmbeddings(embeddings, folder, name + settings)


def get_llm(model: str | None = None, temperature: float | None = None, **kwargs: Any) -> BaseChatModel:
    """The chat model. ``temperature`` is sent only when given: the default model takes none."""
    if config.fake():
        from .fake import FakeChatModel

        return FakeChatModel()
    config.require_openai_key()
    from langchain_openai import ChatOpenAI

    if temperature is not None:
        kwargs["temperature"] = temperature
    folder = config.cache_dir()
    if folder is not None:
        from .cache import AnswerCache

        kwargs.setdefault("cache", AnswerCache(folder))
    return ChatOpenAI(model=model or os.getenv("DEMO_LLM_MODEL", DEFAULT_LLM_MODEL), **kwargs)
