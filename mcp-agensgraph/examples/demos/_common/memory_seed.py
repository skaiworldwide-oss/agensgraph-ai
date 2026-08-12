"""The memory the 04 demo starts from, and how to put it back.

One copy of the seed, because two scripts need the same one: ``build.py`` writes it into an
empty graph, and ``ask.py`` -- which ends by forgetting an entity, a relationship and an
observation -- puts it back before it starts, so that a second run demonstrates what the
first one did rather than a memory the first one already consumed.
"""

from __future__ import annotations

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

# What the demo goes on to learn. Removed by the restore, so that adding it is visible as a
# change rather than as a list that was already this long.
LEARNED = {"entityName": "Alex Kim", "observations": ["Speaks Korean and English"]}


async def restore(mem) -> None:
    """Put the seed back, using the tools rather than SQL.

    ``create_entities`` merges by name and ``create_relations`` skips a relationship already
    there, so this is the same work whether the memory is empty, seeded, or left the way a
    previous run left it -- which is the only reason the demo can be run twice.
    """
    await mem.call_tool("create_entities", {"entities": ENTITIES})
    await mem.call_tool("create_relations", {"relations": RELATIONS})
    await mem.call_tool("delete_observations", {"deletions": [LEARNED]})
