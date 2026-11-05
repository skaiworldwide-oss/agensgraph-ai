"""Conversational memory agent — durable across processes.

A LangGraph agent that answers from the news corpus (demo 03) through a retriever tool, with two
kinds of memory, both in AgensGraph:

* AgensSaver, the checkpointer, holds one conversation's state. The same thread_id resumes it —
  even in a brand-new process — and a different thread_id knows nothing about it.
* AgensStore holds what should outlive any one conversation. A fact learned in one thread is
  still there in the next, because it is keyed by who it is about rather than by which
  conversation mentioned it.

The transcript is also mirrored to AgensChatMessageHistory, beside the news graph, where a Cypher
statement reads it next to the Article chunks through ``cypher_subquery``.

    cd langchain
    # scripted multi-turn conversation + a resume-from-checkpoint demo:
    .venv/bin/python examples/demos/04_chat_memory_agent/agent.py

    # one turn on a thread (run repeatedly with the same id to see real resume):
    .venv/bin/python examples/demos/04_chat_memory_agent/agent.py my-thread "your message"

Prerequisite: demo 03's news store (run examples/demos/03_news_vector_rag/ingest.py).
"""

from __future__ import annotations

import asyncio
import pathlib
import sys
import time
from hashlib import md5
from typing import Any, Dict, List, Sequence, Tuple
from uuid import uuid4

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from langchain.agents import create_agent
from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool, tool

from _common import agens, config, console
from _common.models import get_embeddings, get_llm
from langchain_agensgraph import (
    AgensChatMessageHistory,
    AgensGraph,
    AgensgraphVector,
    AgensSaver,
    AgensStore,
)

MEMORY_GRAPH = "agent_memory"   # AgensSaver's tables live in "agent_memory_store"
FACTS_GRAPH = "agent_facts"     # AgensStore's tables live in "agent_facts_store"
NEWS_GRAPH = "news"             # built by demo 03; the chat log is kept beside it

USER = "demo-user"              # long-term memory is keyed by user, not by thread


def news_store(graph_name: str = NEWS_GRAPH) -> AgensgraphVector:
    """Demo 03's store, searched by embedding."""
    return AgensgraphVector.from_existing_index(
        get_embeddings(), engine=agens.get_engine(), graph_name=graph_name, label="Article"
    )


def facts_store(graph_name: str = FACTS_GRAPH) -> AgensStore:
    """Long-term memory, with semantic recall.

    `index` turns on embedding-backed search. The dimension is read from the embedding model
    rather than hard-coded, so switching models needs no edit here. The store's tables sit in
    the graph's companion schema; the graph itself need not exist.
    """
    embeddings = get_embeddings()
    return AgensStore(
        agens.get_engine(),
        graph_name=graph_name,
        index={
            "dims": len(embeddings.embed_query("dimension probe")),
            "embed": embeddings,
            "fields": ["text"],
        },
    )


def make_tools(news: AgensgraphVector, facts: AgensStore) -> List[BaseTool]:
    """The agent's tools: search the news, and remember or recall facts about the user."""

    @tool
    def search_news(query: str) -> str:
        """Search the news corpus and return the most relevant article snippets."""
        hits = news.similarity_search(query, k=4)
        if not hits:
            return "No matching news found."
        return "\n\n".join(
            f"[{d.metadata.get('domain', '?')} {d.metadata.get('date', '')}] "
            f"{d.metadata.get('title', '')}: {d.page_content[:200]}"
            for d in hits
        )

    @tool
    def remember_about_user(fact: str) -> str:
        """Save something durable about the user, to recall in later conversations."""
        # The key is a digest of the fact, so remembering the same thing twice overwrites one
        # memory instead of making a second. It is a stable digest rather than hash(), which is
        # seeded per process: the same fact would key differently in the next run, which is the
        # boundary this memory exists to cross.
        key = md5(fact.encode("utf-8")).hexdigest()
        facts.put(("memories", USER), key, {"text": fact})
        return f"Noted: {fact}"

    @tool
    def recall_about_user(query: str) -> str:
        """Look up what is known about the user from earlier conversations."""
        hits = facts.search(("memories", USER), query=query, limit=5)
        if not hits:
            return "Nothing recorded about the user yet."
        return "\n".join(f"- {h.value['text']}" for h in hits)

    return [search_news, remember_about_user, recall_about_user]


def build_agent(
    tools: Sequence[BaseTool], facts: AgensStore, memory_graph: str = MEMORY_GRAPH
) -> Tuple[Any, AgensSaver]:
    """A fresh agent — as a new process would build it.

    The checkpointer scopes state to a thread; the store is shared across all of them. One
    AgensSaver serves both invoke and ainvoke.
    """
    saver = AgensSaver(agens.get_engine(), graph_name=memory_graph)
    agent = create_agent(get_llm(), list(tools), checkpointer=saver, store=facts)
    return agent, saver


def ask(agent, thread_id: str, text: str, run_id: str | None = None) -> str:
    configurable: Dict[str, Any] = {"thread_id": thread_id}
    if run_id is not None:
        # Tagging the turn is what makes `delete_for_runs` able to find it later. It has
        # to travel in `configurable` or `metadata`: LangGraph merges those into the
        # checkpoint's metadata, and drops a `run_id` given at the top level of a config.
        configurable["run_id"] = run_id
    out = agent.invoke(
        {"messages": [{"role": "user", "content": text}]},
        config={"configurable": configurable},
    )
    return out["messages"][-1].text


def conversation(agent, thread_id: str) -> List[BaseMessage]:
    """The thread's messages as the checkpointer holds them: turns, tool calls and results."""
    return agent.get_state({"configurable": {"thread_id": thread_id}}).values["messages"]


def checkpoint_count(saver: AgensSaver, thread_id: str) -> int:
    return sum(1 for _ in saver.list({"configurable": {"thread_id": thread_id}}))


def mentioned_domains(graph: AgensGraph, history: AgensChatMessageHistory) -> List[Dict[str, Any]]:
    """The news domains the conversation mentions, read by one Cypher statement.

    ``cypher_subquery`` is SQL a Cypher statement can hold: here the session's messages as a
    list, unwound and matched against the Article chunks of the graph the history sits beside.
    """
    return graph.query(
        f"UNWIND [1] AS a WITH {history.cypher_subquery()} AS messages "
        'UNWIND messages AS m MATCH (n:"Article") WHERE m.text CONTAINS n.domain '
        "RETURN n.domain AS domain, count(DISTINCT m.seq) AS messages "
        "ORDER BY messages DESC LIMIT 5",
        (history.session_id,),
    )


async def _answer_all(
    agent, saver: AgensSaver, threads: Sequence[Tuple[str, str]]
) -> Tuple[List[Any], List[bool]]:
    """Answer several conversations at once, then read each one's checkpoint back.

    One loop for the whole thing: a pool's workers are tasks of the loop that opened it. The
    engine's awaiting pools are closed on that loop before it ends.
    """
    try:
        answers = await asyncio.gather(
            *(
                agent.ainvoke(
                    {"messages": [{"role": "user", "content": text}]},
                    config={"configurable": {"thread_id": thread_id}},
                )
                for thread_id, text in threads
            )
        )
        persisted = [
            (await saver.aget_tuple({"configurable": {"thread_id": t}})) is not None
            for t, _ in threads
        ]
        for thread_id, _text in threads:
            await saver.adelete_thread(thread_id)
        return answers, persisted
    finally:
        await agens.aclose()


def concurrent_conversations(agent, saver: AgensSaver) -> None:
    """Run separate conversations at the same time, on one engine.

    The same AgensSaver answers both surfaces. What the awaiting path buys is that these overlap:
    each conversation's checkpoint reads and writes take a connection while the others are
    waiting on the model, rather than queueing behind one. This closes the shared engine, so it
    runs last.
    """
    console.section("concurrent conversations (ainvoke, same checkpointer)")
    threads = [
        ("async-a", "In one word, name a technology in the news."),
        ("async-b", "In one word, name a place in the news."),
        ("async-c", "In one word, name a company in the news."),
    ]
    for thread_id, _text in threads:
        saver.delete_thread(thread_id)

    started = time.perf_counter()
    answers, persisted = asyncio.run(_answer_all(agent, saver, threads))
    elapsed = time.perf_counter() - started

    for (thread_id, _text), out in zip(threads, answers):
        print(f"  [{thread_id}] {out['messages'][-1].text.strip()[:60]}")
    print(f"  {len(threads)} conversations answered in {elapsed:.1f}s")
    print(f"  each resumable by the awaiting path: {persisted}")


def thread_lifecycle(agent, saver: AgensSaver, thread: str) -> None:
    """Fork a conversation, trim its history, and drop one run's checkpoints.

    A thread accumulates a checkpoint per step, which is what makes it resumable and
    what makes it grow. These are the operations for managing that.
    """
    console.section("thread lifecycle — fork, trim, and drop a run")

    forked = f"{thread}-forked"
    saver.delete_thread(forked)  # copying onto an occupied thread is refused
    saver.copy_thread(thread, forked)
    print(f"  copy_thread: {thread!r} has {checkpoint_count(saver, thread)} checkpoints, "
          f"{forked!r} now has {checkpoint_count(saver, forked)}")

    # A fork carries the whole parent chain, so it resumes on its own from here.
    answer = ask(agent, forked, "In one sentence, what have we been discussing?")
    print(f"  the fork resumes independently:\n    🤖  {answer}")
    print(f"  and only the fork grew: {thread!r}={checkpoint_count(saver, thread)}, "
          f"{forked!r}={checkpoint_count(saver, forked)}")

    # `keep_latest` keeps the current state, and any older checkpoint a channel still
    # needs to rebuild itself from. The thread stays resumable.
    saver.prune([forked])
    current = saver.get_tuple({"configurable": {"thread_id": forked}})
    print(f"  prune: {forked!r} trimmed to {checkpoint_count(saver, forked)}; "
          f"still resumable: {current is not None}")

    # What prune kept, per channel: the writes an ancestor still contributes and the
    # stored value they are applied to. A channel that records deltas rather than whole
    # values is rebuilt from these, which is why they outlive the checkpoints around them.
    history = saver.get_delta_channel_history(
        config=current.config, channels=sorted(current.checkpoint["channel_values"])
    )
    for channel, entry in sorted(history.items()):
        print(f"    {channel}: {len(entry.get('writes', []))} write(s), "
              f"seed {'kept' if 'seed' in entry else 'not needed'}")

    # A run is one invocation. Tagging it lets its checkpoints be found again, across
    # whatever threads it touched.
    run = str(uuid4())
    tagged = f"{thread}-tagged"
    saver.delete_thread(tagged)
    ask(agent, tagged, "Say hello in five words.", run_id=run)
    before = checkpoint_count(saver, tagged)
    saver.delete_for_runs([run])
    print(f"  delete_for_runs: run {run[:8]} left {before} checkpoints on {tagged!r}, "
          f"{checkpoint_count(saver, tagged)} after")

    saver.delete_thread(forked)
    saver.delete_thread(tagged)


def main() -> None:
    config.require_openai_key()
    try:
        run(sys.argv[1:])
    finally:
        agens.close()


def run(args: Sequence[str]) -> None:
    news = news_store()
    facts = facts_store()
    tools = make_tools(news, facts)

    # one-turn mode: run repeatedly with the same thread id to see cross-process resume
    if len(args) > 1:
        thread, text = args[0], args[1]
        agent, _ = build_agent(tools, facts)
        print(ask(agent, thread, text))
        return

    thread = "demo-conversation"
    console.section("multi-turn conversation (state persisted by AgensSaver)")
    agent, saver = build_agent(tools, facts)
    saver.delete_thread(thread)  # start clean for the scripted demo

    turns = [
        "Search the news for stories about artificial intelligence and summarize the main themes.",
        "Which of those themes relates most to jobs or hiring?",
        "Give one concrete example from the articles you found.",
    ]
    for t in turns:
        answer = ask(agent, thread, t)
        print(f"\n🧑  {t}")
        print(f"🤖  {answer}")
    transcript = conversation(agent, thread)

    # Resume in a FRESH agent + checkpointer (as a new process would) — same thread.
    console.section("resume from checkpoint (new agent instance, same thread_id)")
    agent2, _ = build_agent(tools, facts)
    q = "Without searching again, what was my very first question in this conversation?"
    print(f"\n🧑  {q}")
    print(f"🤖  {ask(agent2, thread, q)}")

    # AgensStore — memory that is NOT scoped to a conversation.
    console.section("long-term memory (AgensStore) — a second, unrelated thread")
    for existing in facts.search(("memories", USER), limit=100):
        facts.delete(("memories", USER), existing.key)

    agent3, _ = build_agent(tools, facts)
    other = "a-different-conversation"
    saver.delete_thread(other)
    tell = "Remember that I work on graph databases and prefer short answers."
    print(f"\n🧑  [thread {thread!r}] {tell}")
    print(f"🤖  {ask(agent3, thread, tell)}")

    # A different thread_id: the checkpointer knows nothing about the exchange above,
    # so anything recalled here came from the store rather than the conversation.
    recall = "What do you already know about me? Do not search the news."
    print(f"\n🧑  [thread {other!r}] {recall}")
    print(f"🤖  {ask(agent3, other, recall)}")

    console.section("what is actually stored")
    for item in facts.search(("memories", USER), limit=10):
        print(f"  {item.namespace} {item.key}: {item.value['text']}")
    print(f"  namespaces in use: {facts.list_namespaces()}")

    thread_lifecycle(agent, saver, thread)

    # AgensChatMessageHistory — a per-session message log, kept beside the news graph.
    console.section("AgensChatMessageHistory — the transcript, read next to the graph")
    history = AgensChatMessageHistory(thread, agens.get_engine(), graph_name=NEWS_GRAPH)
    history.clear()
    history.add_messages(transcript)
    print(f"  stored {len(history.messages)} messages for session {thread!r}; "
          f"first: {history.messages[0].text[:60]!r}")
    rows = mentioned_domains(agens.make_graph(NEWS_GRAPH), history)
    print("  news domains the conversation mentions (one Cypher statement over messages "
          "and Article chunks):")
    for row in rows:
        print(f"    {row['domain']}: {row['messages']} message(s)")

    concurrent_conversations(agent, saver)


if __name__ == "__main__":
    main()
