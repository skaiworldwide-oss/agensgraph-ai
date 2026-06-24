# 04 · Memory — an agent's evolving knowledge graph

The **memory** server is a persistent knowledge graph an assistant grows over a
conversation: **entities** (each with free-text observations) and **relations** between
them. This demo seeds a small, realistic "what the assistant remembers about a traveler"
memory and then exercises the complete tool set.

📓 **Guided tour:** [`memory.ipynb`](./memory.ipynb).

## Run

```bash
# from mcp-agensgraph/examples/demos
.venv/bin/python 04_memory/build.py    # seed (rebuilds each run)
.venv/bin/python 04_memory/ask.py      # read + search + evolve + forget
```

Knobs: `MEM_DB` (default `mcp_memory`), `MEM_GRAPH` (default `memory`).
`ask.py` mutates the memory at the end (to show deletes) — re-run `build.py` to reset.

## Capabilities shown

| Tool | In the demo |
|------|-------------|
| `create_entities` / `create_relations` | seed 5 entities + 5 relations |
| `read_graph` | read the whole memory; `limit=2` shows the `truncated` flag |
| `search_memories` | full-text over name/type/observations (`"Seoul"`, `"airport"`, `"March 2026"`) |
| `find_memories_by_name` | exact lookup of an entity **plus its relations** |
| `add_observations` | the assistant learns a new fact about an entity |
| `delete_observations` / `delete_relations` / `delete_entities` | forget specific facts, edges, or whole entities |

## What it shows

```python
async with clients.memory_client("mcp_memory", "memory") as mem:
    await mem.call_tool("search_memories", {"query": "Seoul"})
    await mem.call_tool("read_graph", {"limit": 2})            # → {"entities", "relations", "truncated"}
    await mem.call_tool("add_observations", {"observations": [{"entityName": "Alex Kim", "observations": ["..."]}]})
```

Relations reference entities by name, so a capped `read_graph` stays coherent — and
`search_memories` is the way to narrow a large memory instead of dumping all of it.
