"""Factory that wires LightRAG to the four AgensGraph storages + OpenAI.

A single AgensGraph database serves all four LightRAG storage roles (graph,
vector, KV, doc-status) through one shared, pooled connection. ``open_rag`` is an
async context manager that ensures the database exists, builds the instance,
initializes the storages + pipeline, yields it, and finalizes cleanly.

Cost-control defaults matter here because LightRAG insert is LLM-extraction-bound:
``entity_extract_max_gleaning=1`` and a ~1200-token chunk keep the number of
extraction calls sane, while ``llm_model_max_async`` parallelizes them.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator, List, Optional

import lightrag_agensgraph  # noqa: F401  — side effect: registers the Agens* storages
from lightrag import LightRAG
from lightrag.kg.shared_storage import initialize_pipeline_status

from . import config
from .models import EMBED_DIM, get_embed_func, get_llm_func

# A general-purpose ontology that suits encyclopedic + news text. Demos override
# via the ``entity_types`` argument when a domain-specific set fits better.
DEFAULT_ENTITY_TYPES: List[str] = [
    "person", "organization", "location", "event", "product", "work", "concept",
]


def working_dir_for(db: str, workspace: str) -> str:
    d = config.DEMOS_ROOT / ".data" / "working" / (f"{db}__{workspace}" if workspace else db)
    d.mkdir(parents=True, exist_ok=True)
    return str(d)


def build_rag(
    db: str,
    *,
    workspace: str = "",
    working_dir: Optional[str] = None,
    gleaning: int = 1,
    chunk_token_size: int = 1200,
    chunk_overlap_token_size: int = 100,
    max_async: int = 8,
    max_parallel_insert: int = 6,
    entity_types: Optional[List[str]] = None,
    addon_params: Optional[dict] = None,
) -> LightRAG:
    """Construct (but do not initialize) a LightRAG backed by AgensGraph.

    ``max_parallel_insert`` (documents extracted concurrently; LightRAG's default
    is 3) and ``max_async`` (concurrent LLM calls) together govern how fast the
    extraction-bound insert runs.
    """
    config.apply_env(db, workspace=workspace)
    addon = {
        "language": "English",
        "entity_types": entity_types or DEFAULT_ENTITY_TYPES,
        **(addon_params or {}),
    }
    return LightRAG(
        working_dir=working_dir or working_dir_for(db, workspace),
        workspace=workspace,
        llm_model_func=get_llm_func(),
        llm_model_max_async=max_async,
        max_parallel_insert=max_parallel_insert,
        embedding_func=get_embed_func(),
        chunk_token_size=chunk_token_size,
        chunk_overlap_token_size=chunk_overlap_token_size,
        entity_extract_max_gleaning=gleaning,
        addon_params=addon,
        graph_storage="AgensgraphStorage",
        vector_storage="AgensgraphVectorStorage",
        kv_storage="AgensgraphKVStorage",
        doc_status_storage="AgensgraphDocStatusStorage",
    )


async def reset_rag(rag: LightRAG) -> None:
    """Drop everything for this instance (graph + all relational stores).

    Used by ``*_RESET=1`` to rebuild a demo's KG from scratch. Each store knows
    how to drop only its own data (the graph by name, the relational stores by
    workspace), so this is safe within a demo's dedicated database.
    """
    for attr in (
        "chunk_entity_relation_graph", "entities_vdb", "relationships_vdb",
        "chunks_vdb", "full_docs", "text_chunks", "doc_status", "llm_response_cache",
    ):
        store = getattr(rag, attr, None)
        drop = getattr(store, "drop", None)
        if drop is not None:
            await drop()


@asynccontextmanager
async def open_rag(db: str, *, ensure: bool = True, **kwargs) -> AsyncIterator[LightRAG]:
    """Ensure the DB exists, build + initialize a LightRAG, yield it, finalize.

        async with open_rag("lightrag_wiki") as rag:
            await rag.ainsert(text)
            print(await rag.aquery("..."))
    """
    if ensure:
        config.ensure_db(db)
    rag = build_rag(db, **kwargs)
    await rag.initialize_storages()
    await initialize_pipeline_status()
    try:
        yield rag
    finally:
        await rag.finalize_storages()
