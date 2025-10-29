"""
Integration tests for AgensGraph database schema verification.
Tests that generated constraints and queries actually work in AgensGraph.
"""
import pytest
from mcp_agensgraph_data_modeling.data_model import Node, Property, Relationship, DataModel
from mcp_agensgraph_data_modeling.utils import _quote_identifiers
from psycopg.types.json import Jsonb


@pytest.mark.asyncio
@pytest.mark.database
class TestConstraintCreation:
    """Test constraint creation in AgensGraph database."""

    async def test_create_node_constraint(self, db_connection, graphname):
        """Test creating a UNIQUE constraint on a node's key property."""
        conn, cursor = db_connection

        # Create a node definition
        node = Node(
            label="Person",
            key_property=Property(name="personId", type="STRING"),
            properties=[
                Property(name="firstName", type="STRING"),
                Property(name="lastName", type="STRING")
            ]
        )

        # Get the constraint query
        constraint_query = node.get_cypher_constraint_query()
        print(f"\nConstraint Query: {constraint_query}")

        # Execute the constraint creation
        await cursor.execute(constraint_query)
        await conn.commit()

        # Verify constraint was created using the helper function
        await cursor.execute("SELECT property_has_unique_constraint('personId')")
        result = await cursor.fetchone()
        assert result[0] is True, "Constraint should be created"

    async def test_create_relationship_constraint(self, db_connection, graphname):
        """Test creating a UNIQUE constraint on a relationship's key property."""
        conn, cursor = db_connection

        # Create a relationship definition with key property
        relationship = Relationship(
            type="WORKS_AT",
            start_node_label="Person",
            end_node_label="Company",
            key_property=Property(name="employmentId", type="STRING"),
            properties=[Property(name="startDate", type="DATE")]
        )

        # Get the constraint query
        constraint_query = relationship.get_cypher_constraint_query()
        print(f"\nRelationship Constraint Query: {constraint_query}")

        # Execute the constraint creation
        await cursor.execute(constraint_query)
        await conn.commit()

        # Verify constraint was created
        await cursor.execute("SELECT property_has_unique_constraint('employmentId')")
        result = await cursor.fetchone()
        assert result[0] is True, "Relationship constraint should be created"

    async def test_constraint_prevents_duplicates(self, db_connection, graphname):
        """Test that UNIQUE constraint prevents duplicate values."""
        conn, cursor = db_connection

        # Create node with constraint
        node = Node(
            label="User",
            key_property=Property(name="userId", type="STRING")
        )

        constraint_query = node.get_cypher_constraint_query()
        await cursor.execute(constraint_query)
        await conn.commit()

        # Create first node
        create_query1 = _quote_identifiers("""
            CREATE (u: User {userId: 'user123', name: 'Alice'})
        """)
        await cursor.execute(create_query1)
        await conn.commit()

        # Try to create duplicate - should fail
        create_query2 = _quote_identifiers("""
            CREATE (u: User {userId: 'user123', name: 'Bob'})
        """)

        with pytest.raises(Exception) as exc_info:
            await cursor.execute(create_query2)
            await conn.commit()

        # Verify it's a constraint violation
        error_msg = str(exc_info.value).lower()
        assert "unique" in error_msg or "duplicate" in error_msg or "violates" in error_msg


@pytest.mark.asyncio
@pytest.mark.database
class TestNodeIngestion:
    """Test node ingestion with generated queries."""

    async def test_ingest_single_node(self, db_connection, graphname):
        """Test ingesting a single node using generated query."""
        conn, cursor = db_connection

        # Create node definition
        node = Node(
            label="Product",
            key_property=Property(name="productId", type="STRING"),
            properties=[
                Property(name="name", type="STRING"),
                Property(name="price", type="FLOAT")
            ]
        )

        # Get ingest query
        ingest_query = node.get_cypher_ingest_query_for_many_records()
        print(f"\nNode Ingest Query: {ingest_query}")

        # Prepare data
        records = [
            {"productId": "prod-001", "name": "Laptop", "price": 999.99}
        ]

        # Execute ingest
        await cursor.execute(ingest_query, {"records": Jsonb(records)})
        await conn.commit()

        # Verify node was created
        verify_query = _quote_identifiers("""
            MATCH (p: Product {productId: 'prod-001'})
            RETURN p.name, p.price
        """)
        await cursor.execute(verify_query)
        result = await cursor.fetchone()

        assert result is not None
        assert result[0] == "Laptop"
        assert float(result[1]) == 999.99

    async def test_ingest_multiple_nodes(self, db_connection, graphname):
        """Test ingesting multiple nodes at once."""
        conn, cursor = db_connection

        node = Node(
            label="Employee",
            key_property=Property(name="empId", type="STRING"),
            properties=[
                Property(name="name", type="STRING"),
                Property(name="department", type="STRING")
            ]
        )

        ingest_query = node.get_cypher_ingest_query_for_many_records()

        # Prepare multiple records
        records = [
            {"empId": "emp-001", "name": "Alice", "department": "Engineering"},
            {"empId": "emp-002", "name": "Bob", "department": "Sales"},
            {"empId": "emp-003", "name": "Charlie", "department": "Marketing"}
        ]

        # Execute ingest
        await cursor.execute(ingest_query, {"records": Jsonb(records)})
        await conn.commit()

        # Verify all nodes were created
        verify_query = _quote_identifiers("""
            MATCH (e: Employee)
            RETURN count(e)
        """)
        await cursor.execute(verify_query)
        result = await cursor.fetchone()

        assert result[0] == 3

    async def test_ingest_updates_existing_nodes(self, db_connection, graphname):
        """Test that MERGE updates existing nodes."""
        conn, cursor = db_connection

        node = Node(
            label="Customer",
            key_property=Property(name="customerId", type="STRING"),
            properties=[
                Property(name="email", type="STRING"),
                Property(name="status", type="STRING")
            ]
        )

        ingest_query = node.get_cypher_ingest_query_for_many_records()

        # First ingest
        records1 = [{"customerId": "cust-001", "email": "alice@example.com", "status": "active"}]
        await cursor.execute(ingest_query, {"records": Jsonb(records1)})
        await conn.commit()

        # Second ingest with updated properties
        records2 = [{"customerId": "cust-001", "email": "alice@example.com", "status": "inactive"}]
        await cursor.execute(ingest_query, {"records": Jsonb(records2)})
        await conn.commit()

        # Verify node was updated
        verify_query = _quote_identifiers("""
            MATCH (c: Customer {customerId: 'cust-001'})
            RETURN c.status
        """)
        await cursor.execute(verify_query)
        result = await cursor.fetchone()

        assert result[0] == "inactive"


@pytest.mark.asyncio
@pytest.mark.database
class TestRelationshipIngestion:
    """Test relationship ingestion with generated queries."""

    async def test_ingest_relationship_basic(self, db_connection, graphname):
        """Test ingesting relationships between existing nodes."""
        conn, cursor = db_connection

        # First, create nodes
        create_nodes = _quote_identifiers("""
            CREATE (p1: Person {personId: 'p1', name: 'Alice'}),
                   (p2: Person {personId: 'p2', name: 'Bob'}),
                   (c: Company {companyId: 'c1', name: 'TechCorp'})
        """)
        await cursor.execute(create_nodes)
        await conn.commit()

        # Create relationship definition
        relationship = Relationship(
            type="WORKS_AT",
            start_node_label="Person",
            end_node_label="Company",
            properties=[Property(name="since", type="INTEGER")]
        )

        # Get ingest query
        ingest_query = relationship.get_cypher_ingest_query_for_many_records(
            start_node_key_property_name="personId",
            end_node_key_property_name="companyId"
        )
        print(f"\nRelationship Ingest Query: {ingest_query}")

        # Prepare relationship data
        records = [
            {"sourceId": "p1", "targetId": "c1", "since": 2020},
            {"sourceId": "p2", "targetId": "c1", "since": 2021}
        ]

        # Execute ingest
        await cursor.execute(ingest_query, {"records": Jsonb(records)})
        await conn.commit()

        # Verify relationships were created
        verify_query = _quote_identifiers("""
            MATCH (p: Person)-[r: WORKS_AT]->(c: Company)
            RETURN count(r)
        """)
        await cursor.execute(verify_query)
        result = await cursor.fetchone()

        assert result[0] == 2

    async def test_ingest_relationship_with_key_property(self, db_connection, graphname):
        """Test ingesting relationships with key property."""
        conn, cursor = db_connection

        # Create nodes
        create_nodes = _quote_identifiers("""
            CREATE (a: Account {accountId: 'a1'}),
                   (t: Transaction {transactionId: 't1'})
        """)
        print(f"\nCreate Nodes Query: {create_nodes}")
        await cursor.execute(create_nodes)
        await conn.commit()

        # Create relationship with key property
        relationship = Relationship(
            type="HAS_TRANSACTION",
            start_node_label="Account",
            end_node_label="Transaction",
            key_property=Property(name="linkId", type="STRING"),
            properties=[Property(name="timestamp", type="INTEGER")]
        )

        ingest_query = relationship.get_cypher_ingest_query_for_many_records(
            start_node_key_property_name="accountId",
            end_node_key_property_name="transactionId"
        )
        print(f"\nRelationship Ingest Query: {ingest_query}")
        records = [{"sourceId": "a1", "targetId": "t1", "linkId": "link-001", "timestamp": 1234567890}]

        await cursor.execute(ingest_query, {"records": Jsonb(records)})
        await conn.commit()

        # Verify relationship with properties
        verify_query = _quote_identifiers("""
            MATCH (a: Account)-[r: HAS_TRANSACTION]->(t: Transaction)
            RETURN r.linkId, t.timestamp
        """)
        print(f"\nVerify Query: {verify_query}")
        await cursor.execute(verify_query)
        result = await cursor.fetchone()

        assert result[0] == "link-001"
        assert result[1] == 1234567890


@pytest.mark.asyncio
@pytest.mark.database
class TestCaseSensitivity:
    """Test that case sensitivity is properly preserved."""

    async def test_case_sensitivity_preservation(self, db_connection, graphname):
        """Test that PascalCase labels and camelCase properties are preserved."""
        conn, cursor = db_connection

        # Create node with mixed case
        node = Node(
            label="ProductCategory",
            key_property=Property(name="categoryId", type="STRING"),
            properties=[Property(name="categoryName", type="STRING")]
        )

        # Create constraint
        constraint_query = node.get_cypher_constraint_query()
        await cursor.execute(constraint_query)
        await conn.commit()

        # Ingest data
        ingest_query = node.get_cypher_ingest_query_for_many_records()
        records = [{"categoryId": "cat-001", "categoryName": "Electronics"}]
        await cursor.execute(ingest_query, {"records": Jsonb(records)})
        await conn.commit()

        # Query with exact case - should work
        verify_query = _quote_identifiers("""
            MATCH (pc: ProductCategory {categoryId: 'cat-001'})
            RETURN pc.categoryName
        """)
        await cursor.execute(verify_query)
        result = await cursor.fetchone()

        assert result is not None
        assert result[0] == "Electronics"


@pytest.mark.asyncio
@pytest.mark.database
class TestCompleteDataModel:
    """Test complete data model workflow."""

    async def test_complete_workflow(self, db_connection, graphname):
        """Test creating constraints and ingesting data for a complete data model."""
        conn, cursor = db_connection

        # Define a complete data model
        data_model = DataModel(
            nodes=[
                Node(
                    label="Person",
                    key_property=Property(name="id", type="STRING"),
                    properties=[
                        Property(name="name", type="STRING"),
                        Property(name="age", type="INTEGER")
                    ]
                ),
                Node(
                    label="City",
                    key_property=Property(name="id", type="STRING"),
                    properties=[Property(name="name", type="STRING")]
                )
            ],
            relationships=[
                Relationship(
                    type="LIVES_IN",
                    start_node_label="Person",
                    end_node_label="City"
                )
            ]
        )

        # Step 1: Create constraints
        constraint_queries = data_model.get_cypher_constraints_query()
        for constraint_query in constraint_queries:
            print(f"\nExecuting constraint: {constraint_query}")
            await cursor.execute(constraint_query)
        await conn.commit()

        # Step 2: Ingest nodes
        person_ingest = data_model.nodes_dict["Person"].get_cypher_ingest_query_for_many_records()
        person_records = [
            {"id": "p1", "name": "Alice", "age": 30},
            {"id": "p2", "name": "Bob", "age": 25}
        ]
        await cursor.execute(person_ingest, {"records": Jsonb(person_records)})

        city_ingest = data_model.nodes_dict["City"].get_cypher_ingest_query_for_many_records()
        city_records = [
            {"id": "c1", "name": "New York"},
            {"id": "c2", "name": "San Francisco"}
        ]
        await cursor.execute(city_ingest, {"records": Jsonb(city_records)})
        await conn.commit()

        # Step 3: Ingest relationships
        rel_ingest = data_model.relationships_dict["(:Person)-[:LIVES_IN]->(:City)"].get_cypher_ingest_query_for_many_records(
            start_node_key_property_name="id",
            end_node_key_property_name="id"
        )
        rel_records = [
            {"sourceId": "p1", "targetId": "c1"},
            {"sourceId": "p2", "targetId": "c2"}
        ]
        await cursor.execute(rel_ingest, {"records": Jsonb(rel_records)})
        await conn.commit()

        # Step 4: Verify complete graph
        verify_query = _quote_identifiers("""
            MATCH (p: Person)-[r: LIVES_IN]->(c: City)
            RETURN p.name as person_name, c.name as city_name
            ORDER BY person_name
        """)
        await cursor.execute(verify_query)
        results = await cursor.fetchall()

        assert len(results) == 2
        assert results[0][0] == "Alice"
        assert results[0][1] == "New York"
        assert results[1][0] == "Bob"
        assert results[1][1] == "San Francisco"
