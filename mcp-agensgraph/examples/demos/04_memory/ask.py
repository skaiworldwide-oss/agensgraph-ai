"""04 · memory ask — the full knowledge-graph memory lifecycle.

Reads and evolves the memory that build.py seeded: read the whole graph (with a
bounded `limit` + `truncated` flag), full-text search, exact name lookup, append
observations, and delete observations / relations / entities — the complete tool set
of the memory MCP server.

    cd mcp-agensgraph/examples/demos
    .venv/bin/python 04_memory/build.py    # seed first
    .venv/bin/python 04_memory/ask.py

(ask.py mutates the memory at the end to show deletes — re-run build.py to reset.)
"""

from __future__ import annotations

import asyncio
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from _common import clients, console

DB = os.getenv("MEM_DB", "mcp_memory")
GRAPH = os.getenv("MEM_GRAPH", "memory")


def _names(graph):
    return sorted(e["name"] for e in graph["entities"])


async def main() -> None:
    async with clients.memory_client(DB, GRAPH) as mem:
        # ---- read the whole memory ----
        console.section("read_graph — the whole memory")
        g = clients.data(await mem.call_tool("read_graph", {}))
        console.kv("entities", _names(g))
        console.kv("relations", [f"{r['source']} -{r['relationType']}-> {r['target']}" for r in g["relations"]])
        console.kv("truncated", g["truncated"])

        # ---- bounded read: limit + truncated flag ----
        console.section("read_graph with a limit — the `truncated` flag")
        capped = clients.data(await mem.call_tool("read_graph", {"limit": 2}))
        console.kv("limit=2 → entities", len(capped["entities"]))
        console.kv("truncated", capped["truncated"])

        # ---- full-text search ----
        console.section("search_memories — full-text across name/type/observations")
        for q in ("Seoul", "airport", "March 2026"):
            hit = clients.data(await mem.call_tool("search_memories", {"query": q}))
            console.kv(f"search '{q}'", _names(hit))

        # ---- exact lookup by name (+ its relations) ----
        console.section("find_memories_by_name — exact lookup + connections")
        found = clients.data(await mem.call_tool("find_memories_by_name", {"names": ["Alex Kim"]}))
        console.kv("Alex Kim observations", found["entities"][0]["observations"])
        console.kv("Alex Kim relations",
                   [f"-{r['relationType']}-> {r['target']}" for r in found["relations"]])

        # ---- evolve: add observations ----
        console.section("add_observations — the assistant learns something new")
        await mem.call_tool("add_observations", {"observations": [
            {"entityName": "Alex Kim", "observations": ["Speaks Korean and English"]}]})
        obs = clients.data(await mem.call_tool("find_memories_by_name", {"names": ["Alex Kim"]}))["entities"][0]
        console.kv("Alex Kim observations now", obs["observations"])

        # ---- forget: delete observation, relation, entity ----
        console.section("Deletes — observations, relations, entities")
        await mem.call_tool("delete_observations", {"deletions": [
            {"entityName": "Alex Kim", "observations": ["Prefers window seats"]}]})
        await mem.call_tool("delete_relations", {"relations": [
            {"source": "Tokyo Trip 2026", "target": "Tokyo Haneda", "relationType": "ARRIVES_AT"}]})
        await mem.call_tool("delete_entities", {"entityNames": ["Tokyo Haneda"]})

        g2 = clients.data(await mem.call_tool("read_graph", {}))
        console.kv("entities after deletes", _names(g2))
        console.kv("relations after deletes",
                   [f"{r['source']} -{r['relationType']}-> {r['target']}" for r in g2["relations"]])
        console.kv("Alex Kim obs after delete",
                   clients.data(await mem.call_tool("find_memories_by_name", {"names": ["Alex Kim"]}))["entities"][0]["observations"])


if __name__ == "__main__":
    asyncio.run(main())
