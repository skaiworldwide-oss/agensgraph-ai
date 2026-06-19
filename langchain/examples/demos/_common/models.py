"""Model providers for the demos — OpenAI only, nothing runs locally.

Embeddings and chat completions both go through the OpenAI API, so no local
CPU/GPU model is ever loaded (no fastembed / onnxruntime / sentence-transformers).
Override the models with DEMO_EMBED_MODEL / DEMO_LLM_MODEL.
"""

from __future__ import annotations

import os

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from . import config

# text-embedding-3-small: 1536-dim, ~$0.02 / 1M tokens — cheap at scale.
DEFAULT_EMBED_MODEL = "text-embedding-3-small"
# gpt-4o-mini: cheap + fast, supports with_structured_output (needed by the KG transformer).
DEFAULT_LLM_MODEL = "gpt-4o-mini"


def get_embeddings(model: str | None = None, **kwargs) -> OpenAIEmbeddings:
    config.require_openai_key()
    return OpenAIEmbeddings(model=model or os.getenv("DEMO_EMBED_MODEL", DEFAULT_EMBED_MODEL), **kwargs)


def get_llm(model: str | None = None, temperature: float = 0.0, **kwargs) -> ChatOpenAI:
    config.require_openai_key()
    return ChatOpenAI(
        model=model or os.getenv("DEMO_LLM_MODEL", DEFAULT_LLM_MODEL),
        temperature=temperature,
        **kwargs,
    )
