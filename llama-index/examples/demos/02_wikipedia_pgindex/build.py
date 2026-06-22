"""Wikipedia knowledge graph — build (LLM-extracted PropertyGraphIndex).

The signature LlamaIndex graph feature: an LLM extracts a typed knowledge graph
from documents and writes it into AgensGraph via the AgensPropertyGraphStore.
Entities are embedded (embed_kg_nodes) so the graph also supports vector context
retrieval, and enhanced_schema is on so Text2Cypher (ask.py) gets a rich schema.

    cd llama-index
    .venv/bin/python examples/demos/02_wikipedia_pgindex/build.py
    WIKI_LIMIT=20 WIKI_RESET=1 .venv/bin/python examples/demos/02_wikipedia_pgindex/build.py  # dry run

Knobs: WIKI_LIMIT (articles, default 500), WIKI_CHARS (lead chars/article, 1800),
WIKI_WORKERS (extraction concurrency, 8), WIKI_RESET=1.
"""

from __future__ import annotations

import os
import pathlib
import sys
from typing import Literal

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import psycopg
from llama_index.core import Document, PropertyGraphIndex
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor

from _common import agens, config, console
from _common.datautil import env_int, stream_hf
from _common.models import EMBED_DIM, configure_settings, get_embed_model, get_llm

GRAPH = "wikipedia_kg"
DATASET = "wikimedia/wikipedia"
DATASET_CONFIG = "20231101.en"

LIMIT = env_int("WIKI_LIMIT", 500)
CHARS = env_int("WIKI_CHARS", 1800)
WORKERS = env_int("WIKI_WORKERS", 8)
RESET = os.getenv("WIKI_RESET", "").strip() not in ("", "0", "false", "False")

# A curated, strict ontology -> a clean, queryable KG (vs. free-text labels).
ENTITIES = Literal["Person", "Organization", "Place", "Event", "Work"]
RELATIONS = Literal[
    "FOUNDED", "LOCATED_IN", "BORN_IN", "PART_OF", "CREATED",
    "MEMBER_OF", "OCCURRED_IN", "PARTICIPATED_IN", "INFLUENCED", "SUCCEEDED",
]
VALIDATION_SCHEMA = {
    "Person": ["FOUNDED", "MEMBER_OF", "CREATED", "BORN_IN", "INFLUENCED", "PARTICIPATED_IN"],
    "Organization": ["LOCATED_IN", "PART_OF", "CREATED", "INFLUENCED", "SUCCEEDED"],
    "Place": ["PART_OF", "LOCATED_IN"],
    "Event": ["OCCURRED_IN", "PART_OF"],
    "Work": ["PART_OF", "CREATED"],
}


def _docs():
    for rec in stream_hf(DATASET, config=DATASET_CONFIG, limit=LIMIT):
        text = (rec.get("text") or "").strip()[:CHARS]
        if len(text) < 200:
            continue
        yield Document(
            text=text,
            metadata={"title": rec.get("title") or "", "url": rec.get("url") or ""},
        )


def _reset_graph() -> None:
    console.sub(f"reset: dropping graph '{GRAPH}'")
    with psycopg.connect(config.url(), autocommit=True) as conn:
        conn.execute(f"DROP GRAPH IF EXISTS {GRAPH} CASCADE")


def main() -> None:
    config.require_openai_key()
    console.section(f"Wikipedia → PropertyGraphIndex '{GRAPH}'  (limit={LIMIT:,} articles)")
    if RESET:
        _reset_graph()

    configure_settings()
    store = agens.make_pg_store(GRAPH, vector_dimension=EMBED_DIM, enhanced_schema=True)
    try:
        docs = list(_docs())
        console.kv("articles", f"{len(docs):,}")
        # NOTE: strict=False. With strict=True the extractor enforces a generated
        # JSON schema on the LLM's structured output and, with gpt-4o-mini + this
        # many entity/relation types, that path returns ZERO triplets. strict=False
        # still steers the model with the ontology below (it emits only these
        # entity/relation types) but doesn't hard-reject — yielding a clean KG.
        extractor = SchemaLLMPathExtractor(
            llm=get_llm(),
            possible_entities=ENTITIES,
            possible_relations=RELATIONS,
            kg_validation_schema=VALIDATION_SCHEMA,
            strict=False,
            max_triplets_per_chunk=10,
            num_workers=WORKERS,
        )
        with console.timer("LLM extraction + ingest + embed") as t:
            PropertyGraphIndex.from_documents(
                docs,
                property_graph_store=store,
                embed_model=get_embed_model(),
                llm=get_llm(),
                kg_extractors=[extractor],
                embed_kg_nodes=True,
                show_progress=True,
                use_async=False,
            )
        print("  " + t.rate(len(docs), "articles"))

        # report what got built (entities live on __Node__ with the type in labels)
        n_entities = store.structured_query(
            "MATCH (n:\"__Node__\") WHERE '__Entity__' IN n.labels RETURN count(n) AS c")[0]["c"]
        n_rels = store.structured_query(
            "MATCH (:\"__Node__\")-[r]->(:\"__Node__\") RETURN count(r) AS c")[0]["c"]
        console.section("done")
        console.kv("entities", f"{n_entities:,}")
        console.kv("relationships", f"{n_rels:,}")
        print("\nNext: .venv/bin/python examples/demos/02_wikipedia_pgindex/ask.py")
    finally:
        agens.close()


if __name__ == "__main__":
    main()
