"""OpenAI model wiring for the demos (no local models — everything is OpenAI).

LightRAG ships ready-made OpenAI helpers: ``gpt_4o_mini_complete`` (an
``llm_model_func``) and ``openai_embed`` (already an ``EmbeddingFunc`` at
dimension 1536 for ``text-embedding-3-small``). Both read ``OPENAI_API_KEY`` from
the environment. Override the models with ``DEMO_LLM_MODEL`` / ``DEMO_EMBED_MODEL``
(+ ``DEMO_EMBED_DIM`` when the embedding model's dimension differs from 1536).
"""

from __future__ import annotations

import os
from functools import partial

from lightrag.llm.openai import gpt_4o_mini_complete, openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc, wrap_embedding_func_with_attrs

LLM_MODEL = os.getenv("DEMO_LLM_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("DEMO_EMBED_MODEL", "text-embedding-3-small")
EMBED_DIM = int(os.getenv("DEMO_EMBED_DIM", "1536"))

# Rough OpenAI pricing (USD / 1M tokens) for the pre-flight estimate only.
_PRICE_IN = 0.15      # gpt-4o-mini input
_PRICE_OUT = 0.60     # gpt-4o-mini output
_PRICE_EMBED = 0.02   # text-embedding-3-small


def get_llm_func():
    """Return an async ``llm_model_func`` for LightRAG."""
    if LLM_MODEL == "gpt-4o-mini":
        return gpt_4o_mini_complete

    async def _llm(prompt, system_prompt=None, history_messages=None, **kwargs):
        kwargs.pop("keyword_extraction", None)
        kwargs.pop("entity_extraction", None)
        return await openai_complete_if_cache(
            LLM_MODEL, prompt, system_prompt=system_prompt,
            history_messages=history_messages or [], **kwargs,
        )

    return _llm


def get_embed_func() -> EmbeddingFunc:
    """Return an ``EmbeddingFunc`` for LightRAG (default text-embedding-3-small)."""
    if EMBED_MODEL == "text-embedding-3-small" and EMBED_DIM == 1536:
        return openai_embed  # already wrapped at dim 1536

    @wrap_embedding_func_with_attrs(embedding_dim=EMBED_DIM, max_token_size=8192)
    async def _embed(texts):
        return await openai_embed(texts, model=EMBED_MODEL)

    return _embed


def count_tokens(text: str) -> int:
    """Approximate token count (o200k_base, the gpt-4o family encoding)."""
    import tiktoken

    try:
        enc = tiktoken.get_encoding("o200k_base")
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text or ""))


def print_cost_estimate(total_input_tokens: int, *, chunk_token_size: int, gleaning: int) -> None:
    """Print a rough OpenAI cost/time estimate before an extraction-heavy build.

    LightRAG insert is LLM-extraction-bound: each chunk drives ``1 + gleaning``
    entity/relation extraction calls. These are deliberately conservative
    ballpark numbers, not a billing guarantee.
    """
    calls_per_chunk = 1 + max(gleaning, 0)
    n_chunks = max(1, total_input_tokens // max(chunk_token_size, 1))
    prompt_overhead = 1500  # extraction system/few-shot prompt, tokens per call
    out_per_call = 450      # extracted entities/relations, tokens per call
    in_tokens = n_chunks * calls_per_chunk * (chunk_token_size + prompt_overhead)
    out_tokens = n_chunks * calls_per_chunk * out_per_call
    embed_tokens = int(total_input_tokens * 1.6)  # chunks + entity/relation descriptions
    usd = (in_tokens * _PRICE_IN + out_tokens * _PRICE_OUT + embed_tokens * _PRICE_EMBED) / 1_000_000
    print(
        f"  [estimate] ~{n_chunks:,} chunks x {calls_per_chunk} extraction call(s)\n"
        f"  [estimate] ~{in_tokens/1e6:.2f}M in + {out_tokens/1e6:.2f}M out (LLM) + "
        f"{embed_tokens/1e6:.2f}M embed tokens\n"
        f"  [estimate] ~${usd:.2f} OpenAI (rough, {LLM_MODEL} + {EMBED_MODEL})"
    )
