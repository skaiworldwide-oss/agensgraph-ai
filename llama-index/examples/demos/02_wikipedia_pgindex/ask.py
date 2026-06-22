"""Wikipedia knowledge graph — ask (PropertyGraphIndex retriever stack).

Run after build.py. Queries the LLM-built KG with the full PropertyGraph
retriever stack:

  - LLMSynonymRetriever   — keyword/synonym match into the graph
  - VectorContextRetriever — entity vector_query (HNSW) + get_rel_map expansion
  - TextToCypherRetriever  — NL → AgensGraph Cypher (custom dialect prompt)

The Text2Cypher prompt is the AgensGraph-specific one in _common/cypher.py: the
store keeps every node on "__Node__" with the entity type in a `labels` list, so
the default LlamaIndex prompt (which emits (:Person)-style Cypher) does NOT work.

    cd llama-index
    .venv/bin/python examples/demos/02_wikipedia_pgindex/ask.py
    .venv/bin/python examples/demos/02_wikipedia_pgindex/ask.py "your question"
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from llama_index.core import PropertyGraphIndex
from llama_index.core.indices.property_graph import (
    LLMSynonymRetriever,
    VectorContextRetriever,
)
from llama_index.core.query_engine import RetrieverQueryEngine

from _common import agens, config, console
from _common.cypher import SafeTextToCypherRetriever, read_only_validator
from _common.models import EMBED_DIM, configure_settings, get_embed_model, get_llm

GRAPH = "wikipedia_kg"
DEFAULT_QUESTIONS = [
    "How many entities of each type are in the graph?",
    "Which 5 entities are connected to the most other entities?",
    "Show 8 example relationships as source, relationship, target.",
]


def text2cypher_demo(store, llm) -> None:
    """Run TextToCypher alone and show the AgensGraph Cypher it generated."""
    console.section("Text2Cypher (AgensGraph dialect) — generated query + result")
    t2c = SafeTextToCypherRetriever(
        graph_store=store,
        llm=llm,
        cypher_validator=read_only_validator,
    )
    for q in DEFAULT_QUESTIONS[:2]:
        console.sub(q)
        with console.timer("text2cypher"):
            nodes = t2c.retrieve(q)
        # the retrieved node's text carries "Generated Cypher query: ... Response: ..."
        print("  " + (nodes[0].node.text.strip().replace("\n", "\n  ") if nodes else "(no result)"))


def ask(index, llm, questions) -> None:
    store = index.property_graph_store
    syn = LLMSynonymRetriever(graph_store=store, llm=llm, include_text=True)
    vec = VectorContextRetriever(
        graph_store=store, embed_model=get_embed_model(),
        similarity_top_k=5, path_depth=2, include_text=True,
    )
    t2c = SafeTextToCypherRetriever(
        graph_store=store, llm=llm,
        cypher_validator=read_only_validator,
    )
    retriever = index.as_retriever(sub_retrievers=[syn, vec, t2c])
    qe = RetrieverQueryEngine.from_args(retriever, llm=llm)
    for q in questions:
        console.section(f"Q: {q}")
        with console.timer("answer"):
            resp = qe.query(q)
        print("  " + str(resp).strip().replace("\n", "\n  "))


def main() -> None:
    config.require_openai_key()
    questions = [sys.argv[1]] if len(sys.argv) > 1 else DEFAULT_QUESTIONS
    configure_settings()
    llm = get_llm()
    store = agens.make_pg_store(GRAPH, vector_dimension=EMBED_DIM, enhanced_schema=True, create=False)
    try:
        console.section("graph schema (fed to Text2Cypher)")
        print(store.get_schema_str()[:1200])

        text2cypher_demo(store, llm)

        index = PropertyGraphIndex.from_existing(
            property_graph_store=store, embed_model=get_embed_model(), llm=llm,
            kg_extractors=[], use_async=False,
        )
        ask(index, llm, questions)
    finally:
        agens.close()


if __name__ == "__main__":
    main()
