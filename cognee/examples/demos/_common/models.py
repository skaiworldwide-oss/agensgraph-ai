"""Token counting + a rough pre-flight cost estimate for cognify.

cognify is LLM-bound: per chunk it runs entity/relationship **extraction** and a
**summarization** call (~2 LLM calls/chunk), plus embeddings for chunks + entities.
These are deliberately conservative ballpark numbers, not a billing guarantee.
"""

from __future__ import annotations

# Rough OpenAI pricing (USD / 1M tokens).
_PRICE_IN = 0.15      # gpt-4o-mini input
_PRICE_OUT = 0.60     # gpt-4o-mini output
_PRICE_EMBED = 0.02   # text-embedding-3-small


def count_tokens(text: str) -> int:
    """Approximate token count (o200k_base, the gpt-4o family encoding)."""
    import tiktoken

    try:
        enc = tiktoken.get_encoding("o200k_base")
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text or ""))


def print_cost_estimate(total_input_tokens: int, *, chunk_tokens: int = 1024) -> None:
    """Print a rough OpenAI cost estimate before an extraction-heavy cognify."""
    n_chunks = max(1, total_input_tokens // max(chunk_tokens, 1))
    prompt_overhead = 1800            # extraction/summarization system+few-shot, per call
    extract_out, summary_out = 600, 200
    # 2 calls/chunk: extract (chunk+overhead in, ~600 out) + summarize (chunk in, ~200 out)
    in_tokens = n_chunks * (2 * (chunk_tokens + prompt_overhead))
    out_tokens = n_chunks * (extract_out + summary_out)
    embed_tokens = int(total_input_tokens * 2.0)   # chunks + entity/relation descriptions
    usd = (in_tokens * _PRICE_IN + out_tokens * _PRICE_OUT + embed_tokens * _PRICE_EMBED) / 1_000_000
    print(
        f"  [estimate] ~{n_chunks:,} chunks x 2 LLM calls (extract + summarize)\n"
        f"  [estimate] ~{in_tokens/1e6:.2f}M in + {out_tokens/1e6:.2f}M out (LLM) + "
        f"{embed_tokens/1e6:.2f}M embed tokens\n"
        f"  [estimate] ~${usd:.2f} OpenAI (rough, gpt-4o-mini + text-embedding-3-small)"
    )
