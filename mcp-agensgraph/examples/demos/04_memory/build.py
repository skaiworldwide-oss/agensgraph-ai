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

DB = os.getenv("MEM_DB", "mcp_memory")
GRAPH = os.getenv("MEM_GRAPH", "memory")

ENTITIES = [
    {"name": "Alex Kim", "type": "person",
     "observations": ["Frequent flyer", "Based in Seoul", "Prefers window seats"]},
    {"name": "Korean Air", "type": "airline",
     "observations": ["SkyTeam member", "Hub at Incheon"]},
    {"name": "Incheon International", "type": "airport",
     "observations": ["IATA code ICN", "Serves Seoul"]},
    {"name": "Tokyo Haneda", "type": "airport",
     "observations": ["IATA code HND"]},
    {"name": "Tokyo Trip 2026", "type": "trip",
     "observations": ["Business trip", "Planned for March 2026"]},
]
RELATIONS = [
    {"source": "Alex Kim", "target": "Incheon International", "relationType": "LIVES_NEAR"},
    {"source": "Alex Kim", "target": "Korean Air", "relationType": "FLIES_WITH"},
    {"source": "Korean Air", "target": "Incheon International", "relationType": "HUB_AT"},
    {"source": "Tokyo Trip 2026", "target": "Incheon International", "relationType": "DEPARTS_FROM"},
    {"source": "Tokyo Trip 2026", "target": "Tokyo Haneda", "relationType": "ARRIVES_AT"},
]


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
        rels = clients.data(await mem.call_tool("create_relations", {"relations": RELATIONS}))
        console.kv("relations created", len(rels))

        graph = clients.data(await mem.call_tool("read_graph", {}))
        console.kv("memory now", f"{len(graph['entities'])} entities, {len(graph['relations'])} relations")
        print("\n  Seeded. Explore it with: .venv/bin/python 04_memory/ask.py")


if __name__ == "__main__":
    asyncio.run(main())
