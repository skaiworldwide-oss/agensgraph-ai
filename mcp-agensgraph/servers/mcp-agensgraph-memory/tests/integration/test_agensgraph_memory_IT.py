import pytest

from mcp_agensgraph_memory.agensgraph_memory import (
    BY_RECENCY,
    AgensGraphMemory,
    Entity,
    KnowledgeGraph,
    ObservationAddition,
    ObservationDeletion,
    Relation,
)


@pytest.mark.asyncio
async def test_create_and_read_entities(memory: AgensGraphMemory):
    """Test creating and reading entities from the knowledge graph."""
    # Create test entities
    test_entities = [
        Entity(
            name="Alice",
            type="Person",
            observations=["Likes reading", "Works at Company X"],
        ),
        Entity(name="Bob", type="Person", observations=["Enjoys hiking"]),
    ]

    # Create entities in the graph
    created_entities = await memory.create_entities(test_entities)
    assert len(created_entities) == 2

    # Read the graph
    graph = await memory.read_graph()

    # Verify entities were created
    assert len(graph.entities) == 2

    # Check if entities have correct data
    entities_by_name = {entity.name: entity for entity in graph.entities}
    assert "Alice" in entities_by_name
    assert "Bob" in entities_by_name
    assert entities_by_name["Alice"].type == "Person"
    assert "Likes reading" in entities_by_name["Alice"].observations
    assert "Enjoys hiking" in entities_by_name["Bob"].observations


@pytest.mark.asyncio
async def test_create_and_read_relations(memory: AgensGraphMemory):
    """Test creating and reading relationships between entities."""
    # Create test entities
    test_entities = [
        Entity(name="Alice", type="Person", observations=[]),
        Entity(name="Bob", type="Person", observations=[]),
    ]
    await memory.create_entities(test_entities)

    # Create test relation
    test_relations = [Relation(source="Alice", target="Bob", relationType="KNOWS")]

    # Create relation in the graph
    created_relations = await memory.create_relations(test_relations)
    assert len(created_relations["created"]) == 1
    assert created_relations["skipped"] == []

    # Read the graph
    graph: KnowledgeGraph = await memory.read_graph()

    # Verify relation was created
    assert len(graph.relations) == 1
    relation = graph.relations[0]
    assert relation.source == "Alice"
    assert relation.target == "Bob"
    assert relation.relationType == "KNOWS"


@pytest.mark.asyncio
async def test_add_observations(memory: AgensGraphMemory):
    """Test adding observations to existing entities."""
    # Create test entity
    test_entity = Entity(
        name="Charlie", type="Person", observations=["Initial observation"]
    )
    await memory.create_entities([test_entity])

    # Add observations
    observation_additions = [
        ObservationAddition(
            entityName="Charlie",
            observations=["New observation 1", "New observation 2"],
        )
    ]

    result = await memory.add_observations(observation_additions)
    assert len(result) == 1

    # Read the graph
    graph = await memory.read_graph()

    # Find Charlie
    charlie = next((e for e in graph.entities if e.name == "Charlie"), None)
    assert charlie is not None

    # Verify observations were added
    assert "Initial observation" in charlie.observations
    assert "New observation 1" in charlie.observations
    assert "New observation 2" in charlie.observations


@pytest.mark.asyncio
async def test_delete_observations(memory: AgensGraphMemory):
    """Test deleting specific observations from entities."""
    # Create test entity with observations
    test_entity = Entity(
        name="Dave",
        type="Person",
        observations=["Observation 1", "Observation 2", "Observation 3"],
    )
    await memory.create_entities([test_entity])

    # Delete specific observations
    observation_deletions = [
        ObservationDeletion(entityName="Dave", observations=["Observation 2"])
    ]

    await memory.delete_observations(observation_deletions)

    # Read the graph
    graph = await memory.read_graph()

    # Find Dave
    dave = next((e for e in graph.entities if e.name == "Dave"), None)
    assert dave is not None

    # Verify observation was deleted
    assert "Observation 1" in dave.observations
    assert "Observation 2" not in dave.observations
    assert "Observation 3" in dave.observations


@pytest.mark.asyncio
async def test_delete_entities(memory: AgensGraphMemory):
    """Test deleting entities from the knowledge graph."""
    # Create test entities
    test_entities = [
        Entity(name="Eve", type="Person", observations=[]),
        Entity(name="Frank", type="Person", observations=[]),
    ]
    await memory.create_entities(test_entities)

    # Delete one entity
    await memory.delete_entities(["Eve"])

    # Read the graph
    graph = await memory.read_graph()

    # Verify Eve was deleted but Frank remains
    entity_names = [e.name for e in graph.entities]
    assert "Eve" not in entity_names
    assert "Frank" in entity_names


@pytest.mark.asyncio
async def test_delete_relations(memory: AgensGraphMemory):
    """Test deleting relationships between entities."""
    # Create test entities
    test_entities = [
        Entity(name="Grace", type="Person", observations=[]),
        Entity(name="Hank", type="Person", observations=[]),
    ]
    await memory.create_entities(test_entities)

    # Create test relations
    test_relations = [
        Relation(source="Grace", target="Hank", relationType="KNOWS"),
        Relation(source="Grace", target="Hank", relationType="WORKS_WITH"),
    ]
    await memory.create_relations(test_relations)

    # Delete one relation
    relations_to_delete = [
        Relation(source="Grace", target="Hank", relationType="KNOWS")
    ]
    await memory.delete_relations(relations_to_delete)

    # Read the graph
    graph: KnowledgeGraph = await memory.read_graph()

    # Verify only the WORKS_WITH relation remains
    assert len(graph.relations) == 1
    assert graph.relations[0].relationType == "WORKS_WITH"


@pytest.mark.asyncio
async def test_search_nodes(memory: AgensGraphMemory):
    """Test fulltext search functionality across entities."""
    # Create test entities
    test_entities = [
        Entity(name="Ian", type="Person", observations=["Likes coffee"]),
        Entity(name="Jane", type="Person", observations=["Likes tea"]),
        Entity(name="coffee", type="Beverage", observations=["Hot drink"]),
    ]
    await memory.create_entities(test_entities)

    # Search for coffee-related nodes
    result = await memory.search_memories("coffee")

    # Verify search results
    entity_names = [e.name for e in result.entities]
    assert "Ian" in entity_names
    assert "coffee" in entity_names
    assert "Jane" not in entity_names


@pytest.mark.asyncio
async def test_find_nodes(memory: AgensGraphMemory):
    """Test finding entities by exact names."""
    # Create test entities
    test_entities = [
        Entity(name="Kevin", type="Person", observations=[]),
        Entity(name="Laura", type="Person", observations=[]),
        Entity(name="Mike", type="Person", observations=[]),
    ]
    await memory.create_entities(test_entities)

    # Find specific nodes by name
    result = await memory.find_memories_by_name(["Kevin", "Laura"])

    # Verify only requested nodes are returned
    entity_names = [e.name for e in result.entities]
    assert "Kevin" in entity_names
    assert "Laura" in entity_names
    assert "Mike" not in entity_names


@pytest.mark.asyncio
async def test_read_graph_limit_and_truncation(memory: AgensGraphMemory):
    """read_graph caps entities at `limit` and flags truncation; relations (by name) stay coherent."""
    await memory.create_entities(
        [Entity(name=f"N{i}", type="thing", observations=[f"obs {i}"]) for i in range(4)]
    )
    await memory.create_relations(
        [Relation(source="N0", target="N1", relationType="LINKS")]
    )

    capped = await memory.read_graph(limit=2)
    assert len(capped.entities) == 2
    assert capped.truncated is True

    full = await memory.read_graph()  # the class default is its cap
    assert len(full.entities) == 4
    assert full.truncated is False
    assert any(r.relationType == "LINKS" for r in full.relations)

    found = await memory.search_memories("thing", limit=3)
    assert len(found.entities) == 3
    assert found.truncated is True


class TestALimitBoundsWhatComesBack:
    """The limit bounds the response, not the request.

    Bounding the names alone left the response to whatever those names were connected to: one
    entity with three hundred neighbours answered with 301 entities and ninety-eight kilobytes,
    saying it had truncated nothing. Names dropped for being past the limit went unreported too,
    so a caller heard about entities it had not asked after and not about the ones it had.
    """

    @pytest.mark.asyncio
    async def test_a_hub_is_cut_to_the_limit_and_says_so(self, memory):
        await memory.create_entities(
            [Entity(name="hub", type="Person", observations=[])]
            + [Entity(name=f"n{i}", type="Person", observations=[]) for i in range(60)]
        )
        await memory.create_relations(
            [Relation(source="hub", target=f"n{i}", relationType="KNOWS") for i in range(60)]
        )
        found = await memory.find_memories_by_name(["hub"], limit=10)
        assert len(found.relations) == 10
        assert len(found.entities) == 11
        assert found.truncated is True

    @pytest.mark.asyncio
    async def test_a_name_dropped_for_being_past_the_limit_is_reported(self, memory):
        await memory.create_entities(
            [Entity(name=f"e{i}", type="Person", observations=[]) for i in range(20)]
        )
        found = await memory.find_memories_by_name([f"e{i}" for i in range(20)], limit=5)
        assert len(found.entities) == 5
        assert found.truncated is True

    @pytest.mark.asyncio
    async def test_nothing_cut_is_not_reported_as_cut(self, memory):
        await memory.create_entities(
            [Entity(name="solo", type="Person", observations=[])]
        )
        found = await memory.find_memories_by_name(["solo"], limit=10)
        assert [e.name for e in found.entities] == ["solo"]
        assert found.truncated is False


class TestWhenSomethingWasWritten:
    """A capped read had no answer to "what did I learn most recently".

    Nothing recorded when an entity was written, so a page was the alphabetically-first slice
    and the memory could only be read in one arbitrary order.
    """

    @pytest.mark.asyncio
    async def test_writing_an_entity_records_when(self, memory):
        await memory.create_entities([Entity(name="a", type="t", observations=[])])
        written = (await memory.read_graph(limit=1)).entities[0]
        assert written.updated is not None
        assert written.updated.endswith("+00:00")

    @pytest.mark.asyncio
    async def test_adding_an_observation_moves_it(self, memory):
        await memory.create_entities([Entity(name="a", type="t", observations=[])])
        first = (await memory.read_graph(limit=1)).entities[0].updated
        await memory.add_observations(
            [ObservationAddition(entityName="a", observations=["later"])]
        )
        second = (await memory.read_graph(limit=1)).entities[0].updated
        assert second > first

    @pytest.mark.asyncio
    async def test_a_page_can_be_the_most_recent_rather_than_the_first(self, memory):
        for name in ("zebra", "apple", "mango"):
            await memory.create_entities([Entity(name=name, type="t", observations=[])])
        by_name = [e.name for e in (await memory.read_graph(limit=3)).entities]
        by_recency = [
            e.name for e in (await memory.read_graph(limit=3, order=BY_RECENCY)).entities
        ]
        assert by_name == ["apple", "mango", "zebra"]
        assert by_recency[0] == "mango"
        assert by_recency[-1] == "zebra"

    @pytest.mark.asyncio
    async def test_an_order_nobody_offers_is_refused(self, memory):
        with pytest.raises(ValueError, match="ordered"):
            await memory.read_graph(order="sideways")
