"""Prove that each demo's core path runs, on the fake models, against AGENSGRAPH_URL.

    cd langchain
    AGENSGRAPH_URL=postgresql://owner:secret@localhost:55432/agensgraph_demos \\
    AGENSGRAPH_UNTRUSTED_URL=postgresql://llm_ro:secret@localhost:55432/agensgraph_demos \\
        .venv/bin/python examples/demos/smoke.py

``DEMOS_FAKE=1`` is set here, so no model is called and no key is needed. Each demo loads its
fixture into graphs of its own (``smoke_*``) through the demo's own functions, then runs one
query, retrieval or agent turn and checks what came back; its graphs and their companion schemas
are dropped afterwards, whether it passed or not. Model-written Cypher runs as the
AGENSGRAPH_UNTRUSTED_URL role, after ``render_grants`` has been applied to the graph as its owner.

One line per demo; the exit status is 1 if any demo failed, whose output and traceback follow.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import pathlib
import sys
import time
import traceback
import warnings
from types import ModuleType
from typing import Callable, Dict, List

os.environ["DEMOS_FAKE"] = "1"

DEMOS = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(DEMOS))

import psycopg
from psycopg.conninfo import conninfo_to_dict

from _common import agens, config, models
from langchain_agensgraph import AgensChatMessageHistory
from langchain_agensgraph.errors import UnsafeCypherError
from langchain_agensgraph.vectorstores import SearchType

_loaded: Dict[str, ModuleType] = {}


def load(path: str) -> ModuleType:
    """A demo script as a module, loaded by path: its directory name starts with a digit."""
    if path not in _loaded:
        name = "smoke_" + path.replace("/", "_").removesuffix(".py")
        spec = importlib.util.spec_from_file_location(name, DEMOS / path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        _loaded[path] = module
    return _loaded[path]


def check(condition: object, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def drop(*graphs: str) -> None:
    for name in graphs:
        agens.make_graph(name).drop_graph()


# ── the demos' core paths ────────────────────────────────────────────────────


def arxiv_graphrag() -> str:
    graph_name = "smoke_arxiv"
    prepare = load("01_arxiv_graphrag/prepare.py")
    query = load("01_arxiv_graphrag/query.py")
    graph = agens.make_graph(graph_name, create=True)
    store = prepare.paper_store(
        models.get_embeddings(), graph_name, create_graph=True, index_build="manual"
    )
    papers = list(prepare.records(50))
    counts = prepare.ingest(graph, store, papers, batch_size=20)
    check(counts["embedded"] == len(papers) == 50, f"embedded {counts['embedded']} of 50")
    again = prepare.ingest(graph, store, papers, batch_size=20)
    check(again["embedded"] == 0, f"a re-run embedded {again['embedded']} papers again")
    check(store.build_index(), "no HNSW index was built")

    analytics = query.graph_analytics(graph)
    check(analytics["authors"] and analytics["years"], "the analytics queries returned nothing")
    reader = query.get_store(models.get_embeddings(), graph_name)
    hits = query.vector_search(reader, prepare.paper_text(papers[0]), k=3)
    check(hits[0][0].metadata["id"] == papers[0]["id"], "a paper's own text did not find it")
    result = query.graphrag(reader, graph, models.get_llm(), query.DEFAULT_QUESTION)
    check(result["answer"].startswith("[fake answer]"), "no answer")
    edges = graph.query("MATCH ()-[r]->() RETURN count(*) AS n")[0]["n"]
    return (
        f"{counts['papers']} papers, {edges} edges; top hit {hits[0][0].metadata['id']}; "
        f"{len(result['related'])} related papers; answered"
    )


def wikipedia_kg() -> str:
    graph_name = "smoke_wikipedia_kg"
    build = load("02_wikipedia_kg/build_kg.py")
    ask = load("02_wikipedia_kg/ask.py")
    graph = agens.make_graph(graph_name, create=True)
    docs = list(build.documents(50, 1800))
    graph_docs, failed = build.extract(build.transformer(models.get_llm()), docs)
    check(len(graph_docs) == 50 and not failed, f"{failed} articles failed extraction")
    build.load(graph, graph_docs)

    agens.grant(graph_name)
    reader = agens.make_graph(graph_name, enhanced_schema=True)
    chain = ask.make_chain(reader, models.get_llm())
    cypher, rows, answer = ask.ask(chain, ask.DEFAULT_QUESTIONS[1])
    check(len(rows) == 5 and rows[0]["entities"] > 0, f"the statement returned {rows}")
    try:
        reader.query_untrusted('MATCH (n:"Document") DETACH DELETE n')
    except UnsafeCypherError:
        refused = True
    else:
        refused = False
    check(refused, "a model-written write was not refused")
    vertices = graph.query("MATCH (n) RETURN count(*) AS n")[0]["n"]
    return (
        f"{len(graph_docs)} articles, {vertices} vertices; text2cypher as the untrusted role "
        f"returned {len(rows)} rows; a write was refused"
    )


def news_vector_rag() -> str:
    graph_name = "smoke_news"
    ingest = load("03_news_vector_rag/ingest.py")
    rag = load("03_news_vector_rag/rag.py")
    store = ingest.news_store(
        models.get_embeddings(), graph_name, create_graph=True, index_build="manual"
    )
    items = list(ingest.chunks(50, 900))
    total = ingest.ingest(store, items, batch=20)
    check(total == 50, f"{total} chunks stored")

    dates, domains = rag.corpus(agens.make_graph(graph_name))
    vector = rag.open_store(SearchType.VECTOR, graph_name)
    hits = vector.similarity_search_with_relevance_scores(items[0][0], k=3)
    check(hits[0][0].metadata["title"] == items[0][1]["title"], "a chunk's own text missed it")
    where = {"$and": [{"domain": {"$in": domains}}, {"date": {"$gte": dates["lo"]}}]}
    filtered = vector.similarity_search("business and markets", k=5, filter=where)
    check(filtered and all(d.metadata["domain"] in domains for d in filtered), "filter leaked")
    hybrid = rag.open_store(SearchType.HYBRID, graph_name, fusion_weights=(1.0, 2.0))
    fused = hybrid.hybrid_search_with_details("stock market", k=5)
    check(any(hit.keyword_rank for hit in fused), "the keyword half found nothing")
    answer = rag.rag_chain(vector, models.get_llm()).invoke(rag.DEFAULT_QUESTION)
    check(answer.startswith("[fake answer]"), "no answer")
    return (
        f"{total} chunks; {len(filtered)} filtered hits; {len(fused)} hybrid hits "
        f"({sum(1 for h in fused if h.keyword_rank)} by keyword); answered"
    )


def chat_memory_agent() -> str:
    news_graph, memory_graph, facts_graph = (
        "smoke_agent_news",
        "smoke_agent_memory",
        "smoke_agent_facts",
    )
    ingest = load("03_news_vector_rag/ingest.py")
    agent = load("04_chat_memory_agent/agent.py")
    loaded = ingest.news_store(models.get_embeddings(), news_graph, create_graph=True)
    ingest.ingest(loaded, ingest.chunks(50, 900), batch=50)

    facts = agent.facts_store(facts_graph)
    tools = agent.make_tools(agent.news_store(news_graph), facts)
    bot, saver = agent.build_agent(tools, facts, memory_graph)
    thread, question = "smoke-thread", "Search the news for stories about artificial intelligence."
    first = agent.ask(bot, thread, question)
    check(first.startswith("From search_news"), f"the turn did not search: {first[:80]}")

    # A fresh agent and checkpointer, as a new process would build them.
    fresh, _ = agent.build_agent(tools, facts, memory_graph)
    resumed = agent.ask(fresh, thread, "Without searching again, what was my first question?")
    check(question in resumed, f"the thread did not resume: {resumed[:80]}")
    agent.ask(fresh, thread, "Remember that I prefer short answers.")
    recalled = agent.ask(fresh, "smoke-other", "What do you already know about me?")
    check("short answers" in recalled, f"the store did not carry the fact: {recalled[:80]}")

    history = AgensChatMessageHistory(thread, agens.get_engine(), graph_name=news_graph)
    history.add_messages(agent.conversation(fresh, thread))
    domains = agent.mentioned_domains(agens.make_graph(news_graph), history)
    check(domains, "cypher_subquery found no domain the conversation mentions")
    return (
        f"agent turn searched; resumed in a fresh agent after "
        f"{agent.checkpoint_count(saver, thread)} checkpoints; recalled across threads; "
        f"cypher_subquery matched {domains[0]['domain']}"
    )


def graph_retrievers() -> str:
    graph_name = "smoke_movies"
    ingest = load("05_graph_retrievers/ingest.py")
    retrieve = load("05_graph_retrievers/retrieve.py")
    graph = agens.make_graph(graph_name, create=True)
    store = ingest.movie_store(models.get_embeddings(), graph_name, create_graph=True)
    counts = ingest.build_catalog(graph, store)
    check(counts == {"vertices": 24, "edges": 38}, f"the catalog is {counts}")

    vector, context = retrieve.retrievers(store)
    title, _, plot = next(movie for movie in ingest.MOVIES if movie[0] == "The Lantern Divers")
    director = ingest.DIRECTED[title]
    plain = vector.invoke(plot)
    check(plain[0].metadata["title"] == title, "the plot did not find its film")
    check(director not in plain[0].page_content, "the vector hit already names the director")
    rendered = context.invoke(plot)
    check(director in rendered[0].page_content, "the graph context does not name the director")

    agens.grant(graph_name)
    rows = retrieve.text2cypher(graph, models.get_llm(), retrieve.COUNT_QUESTION)
    check([json.loads(d.page_content) for d in rows] == [{"films": 2}], f"rows {rows}")
    return (
        f"{counts['vertices']} vertices, {counts['edges']} edges; graph context named "
        f"{director}; text2cypher as the untrusted role counted 2 films"
    )


DEMOS_RUN: List[tuple[str, Callable[[], str], tuple[str, ...]]] = [
    ("01_arxiv_graphrag", arxiv_graphrag, ("smoke_arxiv",)),
    ("02_wikipedia_kg", wikipedia_kg, ("smoke_wikipedia_kg",)),
    ("03_news_vector_rag", news_vector_rag, ("smoke_news",)),
    (
        "04_chat_memory_agent",
        chat_memory_agent,
        ("smoke_agent_news", "smoke_agent_memory", "smoke_agent_facts"),
    ),
    ("05_graph_retrievers", graph_retrievers, ("smoke_movies",)),
]


def preflight() -> str:
    """Where the smoke runs, refused when the untrusted role is the owner's."""
    untrusted = config.require_untrusted_url()
    owner = conninfo_to_dict(config.url())
    role = conninfo_to_dict(untrusted).get("user")
    if role == owner.get("user"):
        raise SystemExit("AGENSGRAPH_UNTRUSTED_URL must name a role other than AGENSGRAPH_URL's")
    with psycopg.connect(config.url()) as conn:
        version = conn.execute("SELECT version()").fetchone()[0]
    server = version.split("(", 1)[1].split(")", 1)[0] if "(" in version else version
    where = f"{owner.get('host', '')}:{owner.get('port', '')}/{owner.get('dbname', '')}"
    return f"{server} at {where}; model-written Cypher as {role}"


def main() -> int:
    print(f"smoke: {preflight()}")
    failures = 0
    try:
        for name, run, graphs in DEMOS_RUN:
            drop(*graphs)
            output = io.StringIO()
            started = time.perf_counter()
            try:
                with contextlib.redirect_stdout(output), warnings.catch_warnings():
                    # Fake embeddings carry no meaning, so cosine relevance can fall below 0.
                    warnings.filterwarnings("ignore", message="Relevance scores must be between")
                    summary = run()
                status, failed = "ok", None
            except Exception as exc:  # noqa: BLE001 - every failure is reported, then the next runs
                status, failed = "FAIL", exc
                first = str(exc).splitlines()[0][:160] if str(exc) else ""
                summary = f"{type(exc).__name__}: {first}"
            finally:
                try:
                    drop(*graphs)
                except Exception as exc:  # noqa: BLE001
                    status, summary = "FAIL", f"{summary}; dropping its graphs failed: {exc}"
                    failed = failed or exc
            seconds = time.perf_counter() - started
            print(f"{status:<4} {name:<22} {seconds:5.1f}s  {summary}", flush=True)
            if failed is not None:
                failures += 1
                tail = output.getvalue().splitlines()[-20:]
                print("\n".join(f"     | {line}" for line in tail), file=sys.stderr)
                traceback.print_exception(failed, file=sys.stderr)
    finally:
        agens.close()
    print(f"smoke: {len(DEMOS_RUN) - failures} of {len(DEMOS_RUN)} demos passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
