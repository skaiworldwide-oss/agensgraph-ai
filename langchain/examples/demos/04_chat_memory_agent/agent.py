"""Conversational memory agent — durable across processes.

A LangGraph ReAct agent that answers from the news corpus (demo 03) via a
retriever tool, with two kinds of memory, both in AgensGraph:

* AgensSaver, the checkpointer, holds one conversation's state. The same thread_id
  resumes it — even in a brand-new process — and a different thread_id knows nothing
  about it.
* AgensStore holds what should outlive any one conversation. A fact learned in one
  thread is still there in the next, because it is keyed by who it is about rather
  than by which conversation mentioned it.

The transcript is also mirrored to AgensChatMessageHistory.

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
from typing import List, Sequence, Tuple
from uuid import uuid4

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from _common import agens, config, console
from _common.models import get_embeddings, get_llm
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from langchain_agensgraph import (
    AgensChatMessageHistory,
    AgensgraphVector,
    AgensSaver,
    AgensStore,
    AsyncAgensSaver,
)
from langchain_agensgraph.vectorstores.agensgraph_vector import SearchType

MEMORY_GRAPH = "agent_memory"   # AgensSaver checkpoints live here
FACTS_GRAPH = "agent_facts"     # AgensStore long-term memory lives here
NEWS_GRAPH = "news"             # built by demo 03

USER = "demo-user"              # long-term memory is keyed by user, not by thread

_store = None
_facts = None


def _news_store():
    global _store
    if _store is None:
        _store = AgensgraphVector.from_existing_index(
            embedding=get_embeddings(),
            index_name="Article_embedding_idx",
            search_type=SearchType.VECTOR,
            node_label="Article",
            graph_name=NEWS_GRAPH,
            engine=agens.get_engine(),
        )
    return _store


def facts_store() -> AgensStore:
    """Long-term memory, with semantic recall.

    `index` turns on embedding-backed search. The dimension is read from the embedding
    model rather than hard-coded, so switching models needs no edit here. Embeddings are
    kept out of the property map, in a companion-schema table, so recall never reads a
    vector out of jsonb.
    """
    global _facts
    if _facts is None:
        embeddings = get_embeddings()
        _facts = AgensStore(
            graph=agens.make_graph(FACTS_GRAPH, create=True, refresh_schema=False),
            index={
                "dims": len(embeddings.embed_query("dimension probe")),
                "embed": embeddings,
                "fields": ["text"],
            },
        )
    return _facts


@tool
def remember_about_user(fact: str) -> str:
    """Save something durable about the user, to recall in later conversations.

    The key is a digest of the fact, so remembering the same thing twice overwrites one
    memory instead of making a second. It has to be a stable digest rather than
    ``hash()``, which is seeded per process: the same fact would key differently in the
    next run, which is precisely the boundary this memory exists to cross.
    """
    key = md5(fact.encode("utf-8")).hexdigest()
    facts_store().put(("memories", USER), key, {"text": fact})
    return f"Noted: {fact}"


@tool
def recall_about_user(query: str) -> str:
    """Look up what is known about the user from earlier conversations."""
    hits = facts_store().search(("memories", USER), query=query, limit=5)
    if not hits:
        return "Nothing recorded about the user yet."
    return "\n".join(f"- {h.value['text']}" for h in hits)


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
    """A fresh agent — as a new process would build it.

    The checkpointer scopes state to a thread; the store is shared across all of them.
    """
    saver = AgensSaver(graph=agens.make_graph(MEMORY_GRAPH, create=True, refresh_schema=False))
    agent = create_agent(
        get_llm(),
        [search_news, remember_about_user, recall_about_user],
        checkpointer=saver,
        store=facts_store(),
    )
    return agent, saver


def ask(agent, thread_id: str, text: str, run_id: str | None = None) -> str:
    configurable = {"thread_id": thread_id}
    if run_id is not None:
        # Tagging the turn is what makes `delete_for_runs` able to find it later. It has
        # to travel in `configurable` or `metadata`: LangGraph merges those into the
        # checkpoint's metadata, and drops a `run_id` given at the top level of a config.
        configurable["run_id"] = run_id
    out = agent.invoke(
        {"messages": [{"role": "user", "content": text}]},
        config={"configurable": configurable},
    )
    return out["messages"][-1].content


def checkpoint_count(saver: AgensSaver, thread_id: str) -> int:
    return sum(1 for _ in saver.list({"configurable": {"thread_id": thread_id}}))


async def _answer_all(
    agent, saver: AsyncAgensSaver, threads: Sequence[Tuple[str, str]]
) -> Tuple[List[str], List[bool]]:
    """Answer several conversations at once, then read each one's checkpoint back.

    One loop for the whole thing: a pool's workers are tasks of the loop that opened it,
    so an engine used across two `asyncio.run` calls has to rebuild it for the second.
    """
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
    return answers, persisted


def concurrent_conversations(saver: AsyncAgensSaver) -> None:
    """Run separate conversations at the same time, on one pool.

    ``AsyncAgensSaver`` is ``AgensSaver`` -- the same object answers both surfaces. What
    the awaiting path buys is that these overlap: each conversation's checkpoint reads
    and writes take a connection from the pool while the others are waiting on the model,
    rather than queueing behind one.
    """
    console.section("concurrent conversations (async checkpointer + store)")

    agent, _ = build_agent()
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
        print(f"  [{thread_id}] {out['messages'][-1].content.strip()[:60]}")
    print(f"  {len(threads)} conversations answered in {elapsed:.1f}s, on one pool")
    print(f"  each resumable by the awaiting path: {persisted}")

    for thread_id, _text in threads:
        saver.delete_thread(thread_id)


def thread_lifecycle(saver: AgensSaver, thread: str) -> None:
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
    agent, _ = build_agent()
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

    # AgensStore — memory that is NOT scoped to a conversation.
    console.section("long-term memory (AgensStore) — a second, unrelated thread")
    facts = facts_store()
    for existing in facts.search(("memories", USER), limit=100):
        facts.delete(("memories", USER), existing.key)

    agent3, _ = build_agent()
    other = "a-different-conversation"
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

    thread_lifecycle(saver, thread)
    concurrent_conversations(saver)

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
