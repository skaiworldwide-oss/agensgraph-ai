# 04 · Conversational memory agent

A durable conversational agent: a LangGraph ReAct agent
(`langchain.agents.create_agent`) that answers from the news corpus (demo 03) via
a retriever tool, with its full conversation state persisted in AgensGraph by
**`AgensSaver`** (the LangGraph checkpointer). The same `thread_id` resumes the
conversation — even in a brand-new process. The transcript is also mirrored to
**`AgensChatMessageHistory`**.

## Run

```bash
cd langchain
# scripted multi-turn conversation + a resume-from-checkpoint demo:
.venv/bin/python examples/demos/04_chat_memory_agent/agent.py

# one turn on a thread — run it AGAIN with the same id (new process) to see resume:
.venv/bin/python examples/demos/04_chat_memory_agent/agent.py my-thread "Find news about AI"
.venv/bin/python examples/demos/04_chat_memory_agent/agent.py my-thread "What did you just tell me?"
```

Prerequisite: demo 03's news store (`03_news_vector_rag/ingest.py`).

## What it demonstrates

- **`AgensSaver`** — a LangGraph checkpointer backed by AgensGraph. Passed as
  `create_agent(model, tools, checkpointer=AgensSaver(...))`, it persists the
  agent's full state per `thread_id`, so:
  - later turns see earlier ones without re-sending them, and
  - a **fresh agent instance** (as a new process) resumes the exact conversation
    from the checkpoint — the demo asks "what was my first question?" and it
    answers from persisted state, without searching again.
- **`AgensStore`** — the other half of memory. The checkpointer scopes state to one
  conversation; the store holds what should outlive any of them, keyed by who it is
  about rather than by which thread mentioned it. Passed as
  `create_agent(..., store=AgensStore(...))` and reached through `remember_about_user`
  / `recall_about_user` tools, so the demo can state a preference on one `thread_id`
  and have it recalled on a **different** one — something the checkpointer cannot do
  by design.
  - `index={"dims": ..., "embed": ..., "fields": ["text"]}` turns on semantic recall.
    The dimension is read from the embedding model rather than hard-coded.
  - `put` embeds the one memory it writes, which is what an agent learning a single fact
    wants. Seeding or importing should use `batch`, which embeds the whole set in one
    request — measured at 2.1 memories a second against 21.
  - Items are ordinary vertices, so a memory can be linked to other memories or to
    domain data with ordinary edges. Embeddings are held outside the property map, in
    a companion-schema table, so recall never reads a vector out of jsonb.
- **Grounded tool use** — a `search_news` tool runs `AgensgraphVector`
  similarity search over the news store, so graph store, vector store, and agent
  memory all live in one AgensGraph database.
- **`AgensChatMessageHistory`** — a lighter per-session message log
  (`add_messages` / `.messages` / `clear`), shown alongside the checkpointer.

See [`chat_memory_agent.ipynb`](chat_memory_agent.ipynb) for a pre-executed,
end-to-end walkthrough.

## Notes

- Uses OpenAI for the agent's reasoning and the tool's embeddings.
- `delete_thread(thread_id)` clears a conversation's checkpoints. `prune([thread_id])`
  trims a thread to its current state while keeping whatever a channel still needs to
  rebuild; `copy_thread(src, dst)` forks one; `delete_for_runs([...])` removes the
  checkpoints of particular runs across whatever threads they touched.
- The two memories are independent: clearing a conversation leaves the store's facts,
  and clearing the store leaves the conversations.
- The agent loop (reason → call tool → answer) issues a few LLM calls per turn.
