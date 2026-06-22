"""01 · ask — the five LightRAG query modes, side by side.

LightRAG's signature is dual-level retrieval. The same question runs through all
five modes so the difference is visible, not asserted:

  naive   — vector similarity over text chunks only (the baseline, no graph)
  local   — entity-centric: low-level keywords → specific entities + their facts
  global  — relationship-centric: high-level keywords → cross-document themes
  hybrid  — local + global merged
  mix     — hybrid KG retrieval + naive chunks together (LightRAG's default)

Then `aquery_data` shows WHAT each mode pulled from the graph, and
`only_need_context` shows the retrieved context without an LLM answer.

    cd lightrag
    .venv/bin/python examples/demos/01_kg_modes/ask.py
    .venv/bin/python examples/demos/01_kg_modes/ask.py "your question"
"""

from __future__ import annotations

import asyncio
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from lightrag import QueryParam
from lightrag.kg.shared_storage import initialize_pipeline_status

from _common import config, console
from _common.rag import build_rag

DB = "lightrag_wiki"
MODES = ["naive", "local", "global", "hybrid", "mix"]


def qp(mode: str, **kw) -> QueryParam:
    # No rerank model is configured, so disable reranking to avoid a no-op warning.
    return QueryParam(mode=mode, enable_rerank=False, **kw)


async def pick_question(rag) -> str:
    if len(sys.argv) > 1:
        return sys.argv[1]
    popular = await rag.chunk_entity_relation_graph.get_popular_labels(limit=1)
    entity = popular[0] if popular else "this collection"
    # Phrasing that yields concrete low-level keywords (the entity + related
    # nouns) so `local` mode has entity keywords to work with — avoid meta words
    # like "entities"/"connections", which the keyword extractor picks up as-is.
    return f"Tell me about {entity}: its key facts and the people, places, and events associated with it."


def _summarize(result: dict) -> str:
    """Render the dual-level retrieval signal from an aquery_data result.

    Shape: {status, message, data:{entities,relationships,chunks,references},
            metadata:{keywords:{high_level,low_level}, ...}}.
    """
    data = result.get("data", {}) if isinstance(result, dict) else {}
    meta = result.get("metadata", {}) if isinstance(result, dict) else {}
    kw = meta.get("keywords", {})
    counts = ", ".join(
        f"{k}={len(data.get(k, []))}" for k in ("entities", "relationships", "chunks")
    )
    hl = ", ".join(kw.get("high_level", []) or []) or "—"
    ll = ", ".join(kw.get("low_level", []) or []) or "—"
    return f"{counts}  | high-level kw: {hl}  | low-level kw: {ll}"


async def main() -> None:
    config.require_openai_key()
    rag = build_rag(DB)
    await rag.initialize_storages()
    await initialize_pipeline_status()
    try:
        question = await pick_question(rag)
        console.section(f"Q: {question}")

        for mode in MODES:
            console.sub(f"mode = {mode}")
            with console.timer(f"{mode} answer"):
                answer = await rag.aquery(question, param=qp(mode))
            text = str(answer).strip().replace("\n", " ")
            print("  " + (text[:600] + (" …" if len(text) > 600 else "")))

        console.section("What each mode retrieved (aquery_data — the dual-level signal)")
        # local leans on low-level (entity) keywords; global on high-level (theme)
        # keywords; mix pulls entities + relationships + raw chunks together.
        for mode in ("local", "global", "mix"):
            result = await rag.aquery_data(question, param=qp(mode))
            console.kv(mode, _summarize(result if isinstance(result, dict) else {}))

        console.section("only_need_context — retrieved context, no LLM answer (mix)")
        ctx = await rag.aquery(question, param=qp("mix", only_need_context=True))
        snippet = str(ctx).strip()
        print("  " + snippet[:800].replace("\n", "\n  ") + (" …" if len(snippet) > 800 else ""))
    finally:
        await rag.finalize_storages()


if __name__ == "__main__":
    asyncio.run(main())
