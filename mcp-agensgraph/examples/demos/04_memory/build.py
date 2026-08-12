"""04 · memory build — populate an agent's knowledge graph.

The memory MCP server is a persistent knowledge graph an assistant grows over a
conversation: entities (with observations) + relations. This seeds a small, realistic
"what the assistant remembers about a traveler" memory in the `mcp_memory` database.

    cd mcp-agensgraph/examples/demos
    .venv/bin/python 04_memory/build.py        # rebuilds the memory each run

Knobs: MEM_DB (default mcp_memory), MEM_GRAPH (default memory).
"""

from __future__ import annotations

import asyncio
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from _common import clients, config, console
from _common.memory_seed import ENTITIES, RELATIONS

DB = os.getenv("MEM_DB", "mcp_memory")
GRAPH = os.getenv("MEM_GRAPH", "memory")


async def main() -> None:
    config.ensure_db(DB)
    # clean rebuild — drop the graph so the populate is deterministic
    import psycopg

    with psycopg.connect(config.dsn(DB), autocommit=True) as conn:
        conn.execute(f'DROP GRAPH IF EXISTS "{GRAPH}" CASCADE')

    console.section(f"Seeding the agent memory in {DB}/{GRAPH}")
    async with clients.memory_client(DB, GRAPH) as mem:
        created = clients.data(await mem.call_tool("create_entities", {"entities": ENTITIES}))
        console.kv("entities created", len(created))
        # create_relations answers `{"created": [...], "skipped": [...]}`, so the count of
        # relationships is the length of one of those and not of the reply.
        rels = clients.data(await mem.call_tool("create_relations", {"relations": RELATIONS}))
        console.kv("relations created", len(rels["created"]))
        console.kv("relations skipped", len(rels["skipped"]))

        graph = clients.data(await mem.call_tool("read_graph", {}))
        console.kv("memory now", f"{len(graph['entities'])} entities, {len(graph['relations'])} relations")
        print("\n  Seeded. Explore it with: .venv/bin/python 04_memory/ask.py")


if __name__ == "__main__":
    asyncio.run(main())
