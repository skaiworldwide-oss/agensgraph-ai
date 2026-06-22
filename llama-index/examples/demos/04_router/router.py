"""Combined routing — one AgensEngine, both stores, one router.

Ties the previous demos together two ways. A LlamaIndex RouterQueryEngine routes
a natural-language question to either the arXiv **property-graph** engine (demo 1)
or the news **vector** engine (demo 3); then a **FunctionAgent** is given the same
two tools as an autonomous alternative. Both stores are built on the SAME shared
AgensEngine connection pool — different graphs in one database, served together.

Run after 01_arxiv_pg/prepare.py and 03_news_vector_rag/ingest.py.

    cd llama-index
    .venv/bin/python examples/demos/04_router/router.py
    .venv/bin/python examples/demos/04_router/router.py "your question"
"""

from __future__ import annotations

import asyncio
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from llama_index.core import PropertyGraphIndex, VectorStoreIndex
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.indices.property_graph import VectorContextRetriever
from llama_index.core.query_engine import RetrieverQueryEngine, RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import QueryEngineTool

from _common import agens, config, console
from _common.models import EMBED_DIM, configure_settings, get_embed_model, get_llm

ARXIV_GRAPH = "arxiv"
NEWS_GRAPH = "news"
DEFAULT_QUESTIONS = [
    "What approaches use neural networks for scientific prediction tasks?",
    "What are companies doing with artificial intelligence?",
]


def pool_activity() -> None:
    """Prove both engines share ONE pool: count this app's pooled connections."""
    rows = agens.make_vector_store(graph_name=NEWS_GRAPH, node_label="Article").database_query(
        "SELECT count(*) AS c, state FROM pg_stat_activity "
        "WHERE application_name = 'llama-index-agensgraph' GROUP BY state"
    )
    console.sub("shared AgensEngine pool (pg_stat_activity)")
    for r in rows:
        print(f"  {r['c']} connection(s)  state={r['state']}  app=llama-index-agensgraph")


def build_tools(llm, embed_model) -> list[QueryEngineTool]:
    # arXiv graph engine — VectorContextRetriever (vector_query + get_rel_map).
    arxiv_store = agens.make_pg_store(ARXIV_GRAPH, vector_dimension=EMBED_DIM, create=False)
    arxiv_index = PropertyGraphIndex.from_existing(
        property_graph_store=arxiv_store, embed_model=embed_model, llm=llm,
        kg_extractors=[], use_async=False,
    )
    graph_retriever = arxiv_index.as_retriever(
        sub_retrievers=[VectorContextRetriever(
            graph_store=arxiv_store, embed_model=embed_model,
            similarity_top_k=5, path_depth=1, include_text=True)]
    )
    graph_qe = RetrieverQueryEngine.from_args(graph_retriever, llm=llm)

    # news vector engine — same shared engine, different graph.
    news_store = agens.make_vector_store(graph_name=NEWS_GRAPH, node_label="Article")
    news_qe = VectorStoreIndex.from_vector_store(
        news_store, embed_model=embed_model
    ).as_query_engine(similarity_top_k=5, llm=llm)

    return [
        QueryEngineTool.from_defaults(
            graph_qe, name="arxiv_papers",
            description=(
                "Scientific and academic questions about arXiv research papers, "
                "their authors, categories, methods and findings. A knowledge "
                "graph with semantic search over paper abstracts."),
        ),
        QueryEngineTool.from_defaults(
            news_qe, name="news_articles",
            description=(
                "Current-events and news questions (business, technology, sports, "
                "markets, world events) answered from a large news-article corpus "
                "via semantic vector search."),
        ),
    ]


def build_router(tools, llm) -> RouterQueryEngine:
    return RouterQueryEngine.from_defaults(
        query_engine_tools=tools,
        selector=LLMSingleSelector.from_defaults(llm=llm),
        llm=llm,
        verbose=True,
    )


async def agent_variant(tools, llm, question: str) -> None:
    console.section("agentic variant — FunctionAgent over the same two tools")
    # Same two QueryEngineTools, but the agent decides which to call (and may call
    # more than one). FunctionAgent.run is async, so it exercises the engine's
    # ASYNC pool — the router above used the sync pool, same shared AgensEngine.
    agent = FunctionAgent(
        tools=tools,
        llm=llm,
        system_prompt=(
            "Answer the question using the tools. Use `arxiv_papers` for "
            "scientific/academic questions and `news_articles` for current "
            "events; call a tool, then answer from its result."),
    )
    print(f"  Q: {question}\n")
    try:
        with console.timer("agent run"):
            resp = await agent.run(question)
        print("  " + str(resp).strip().replace("\n", "\n  "))
    finally:
        # Close BOTH pools from inside this event loop (the async pool's workers
        # are bound to it) so nothing is orphaned at exit.
        await agens.aclose()


def main() -> None:
    config.require_openai_key()
    questions = [sys.argv[1]] if len(sys.argv) > 1 else DEFAULT_QUESTIONS
    configure_settings()
    llm, embed_model = get_llm(), get_embed_model()
    tools = build_tools(llm, embed_model)
    router = build_router(tools, llm)
    try:
        for q in questions:
            console.section(f"Q: {q}")
            with console.timer("route + answer"):
                resp = router.query(q)
            sel = (resp.metadata or {}).get("selector_result")
            if sel is not None:
                print(f"  routed to → {sel.selections[0].index} "
                      f"(reason: {sel.selections[0].reason[:100]})")
            print("  " + str(resp).strip().replace("\n", "\n  "))
        pool_activity()
        # autonomous alternative to the router — same tools, agent picks
        asyncio.run(agent_variant(tools, llm, questions[-1]))
    finally:
        agens.close()


if __name__ == "__main__":
    main()
