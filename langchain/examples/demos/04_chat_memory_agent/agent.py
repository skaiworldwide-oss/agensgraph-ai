"""Conversational memory agent — durable across processes.

A LangGraph ReAct agent that answers from the news corpus (demo 03) via a
retriever tool, with its conversation state persisted in AgensGraph by AgensSaver
(the LangGraph checkpointer). The same thread_id resumes the conversation — even
in a brand-new process. The transcript is also mirrored to AgensChatMessageHistory.

    cd langchain
    # scripted multi-turn conversation + a resume-from-checkpoint demo:
    .venv/bin/python examples/demos/04_chat_memory_agent/agent.py

    # one turn on a thread (run repeatedly with the same id to see real resume):
    .venv/bin/python examples/demos/04_chat_memory_agent/agent.py my-thread "your message"

Prerequisite: demo 03's news store (run examples/demos/03_news_vector_rag/ingest.py).
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from langchain_agensgraph import AgensChatMessageHistory, AgensSaver, AgensgraphVector
from langchain_agensgraph.vectorstores.agensgraph_vector import SearchType

from _common import agens, config, console
from _common.models import get_embeddings, get_llm

MEMORY_GRAPH = "agent_memory"   # AgensSaver checkpoints live here
NEWS_GRAPH = "news"             # built by demo 03

_store = None


def _news_store():
    global _store
    if _store is None:
        _store = AgensgraphVector.from_existing_index(
            embedding=get_embeddings(),
            index_name="vector",
            search_type=SearchType.VECTOR,
            node_label="Article",
            graph_name=NEWS_GRAPH,
            engine=agens.get_engine(),
        )
    return _store


@tool
def search_news(query: str) -> str:
    """Search the news corpus and return the most relevant article snippets."""
    hits = _news_store().similarity_search(query, k=4)
    if not hits:
        return "No matching news found."
    return "\n\n".join(
        f"[{d.metadata.get('domain','?')} {d.metadata.get('date','')}] "
        f"{d.metadata.get('title','')}: {d.page_content[:200]}"
        for d in hits
    )


def build_agent():
    """A fresh agent + checkpointer — as a new process would build it."""
    saver = AgensSaver(graph=agens.make_graph(MEMORY_GRAPH, create=True, refresh_schema=False))
    agent = create_agent(get_llm(), [search_news], checkpointer=saver)
    return agent, saver


def ask(agent, thread_id: str, text: str) -> str:
    out = agent.invoke(
        {"messages": [{"role": "user", "content": text}]},
        config={"configurable": {"thread_id": thread_id}},
    )
    return out["messages"][-1].content


def main() -> None:
    config.require_openai_key()

    # one-turn mode: run repeatedly with the same thread id to see cross-process resume
    if len(sys.argv) > 2:
        thread, text = sys.argv[1], sys.argv[2]
        agent, _ = build_agent()
        print(ask(agent, thread, text))
        agens.close()
        return

    thread = "demo-conversation"
    console.section("multi-turn conversation (state persisted by AgensSaver)")
    agent, saver = build_agent()
    saver.delete_thread(thread)  # start clean for the scripted demo

    turns = [
        "Search the news for stories about artificial intelligence and summarize the main themes.",
        "Which of those themes relates most to jobs or hiring?",
        "Give one concrete example from the articles you found.",
    ]
    for t in turns:
        print(f"\n🧑  {t}")
        print(f"🤖  {ask(agent, thread, t)}")

    # Resume in a FRESH agent + checkpointer (as a new process would) — same thread.
    console.section("resume from checkpoint (new agent instance, same thread_id)")
    agent2, _ = build_agent()
    q = "Without searching again, what was my very first question in this conversation?"
    print(f"\n🧑  {q}")
    print(f"🤖  {ask(agent2, thread, q)}")

    # Bonus: AgensChatMessageHistory — a simple per-session message log.
    console.section("AgensChatMessageHistory — readable transcript for this session")
    history = AgensChatMessageHistory(
        thread, graph=agens.make_graph("chat_log", create=True, refresh_schema=False)
    )
    history.clear()
    for t in turns:
        history.add_message(HumanMessage(content=t))
        history.add_message(AIMessage(content="(answer stored)"))
    print(f"  stored {len(history.messages)} messages for session {thread!r}; "
          f"first: {history.messages[0].content[:60]!r}")

    agens.close()


if __name__ == "__main__":
    main()
