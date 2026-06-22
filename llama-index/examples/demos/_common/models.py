"""Model providers for the demos — OpenAI only, nothing runs locally.

Embeddings and chat completions both go through the OpenAI API, so no local
CPU/GPU model is ever loaded (no fastembed / onnxruntime / sentence-transformers).
Override the models with DEMO_EMBED_MODEL / DEMO_LLM_MODEL.
"""

from __future__ import annotations

import os

from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from . import config

# text-embedding-3-small: 1536-dim, ~$0.02 / 1M tokens — cheap at scale.
DEFAULT_EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536
# gpt-4o-mini: cheap + fast, supports the structured output the KG extractor needs.
DEFAULT_LLM_MODEL = "gpt-4o-mini"


def get_embed_model(model: str | None = None, **kwargs) -> OpenAIEmbedding:
    config.require_openai_key()
    return OpenAIEmbedding(
        model=model or os.getenv("DEMO_EMBED_MODEL", DEFAULT_EMBED_MODEL), **kwargs
    )


def get_llm(model: str | None = None, temperature: float = 0.0, **kwargs) -> OpenAI:
    config.require_openai_key()
    return OpenAI(
        model=model or os.getenv("DEMO_LLM_MODEL", DEFAULT_LLM_MODEL),
        temperature=temperature,
        **kwargs,
    )


def configure_settings(*, llm: bool = True, embed: bool = True) -> None:
    """Wire the global LlamaIndex ``Settings`` once per entry script.

    Retrievers/extractors/query-engines that aren't handed an explicit model
    fall back to ``Settings`` — setting it here keeps behavior unambiguous even
    when a sub-component is constructed without args.
    """
    if llm:
        Settings.llm = get_llm()
    if embed:
        Settings.embed_model = get_embed_model()
