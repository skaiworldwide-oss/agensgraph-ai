import os
from typing import Any
import pytest
import pytest_asyncio

from mcp_agensgraph_cypher.server import create_mcp_server, get_pool_connection
from psycopg.rows import namedtuple_row  # type: ignore
from psycopg_pool import AsyncConnectionPool, PoolTimeout  # type: ignore

@pytest.fixture(scope="module")
def graphname():
    return os.getenv("AGENSGRAPH_GRAPH_NAME", "test")

@pytest_asyncio.fixture(scope="module", autouse=True)
async def setup():
    db_name = os.getenv("AGENSGRAPH_DB")
    db_user = os.getenv("AGENSGRAPH_USERNAME")
    db_password = os.getenv("AGENSGRAPH_PASSWORD")
    db_host = os.getenv("AGENSGRAPH_HOST", "localhost")
    db_port = os.getenv("AGENSGRAPH_PORT", "5432")

    if not db_name or not db_user or not db_password:
        raise ValueError("Environment variables AGENSGRAPH_DB, AGENSGRAPH_USERNAME, and AGENSGRAPH_PASSWORD must be set.")

    db_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    agensgraph_driver = AsyncConnectionPool(db_url, open=False)

    await agensgraph_driver.open()
    
    yield agensgraph_driver

    await agensgraph_driver.close()


@pytest_asyncio.fixture(scope="function")
async def mcp_server(setup, graphname):
    mcp = create_mcp_server(setup, graphname=graphname)

    return mcp


@pytest_asyncio.fixture(scope="function")
async def init_data(setup, clear_data: Any):
    async with get_pool_connection(setup) as conn:
        async with conn.cursor(row_factory=namedtuple_row) as cursor:
            # Create test data
            await cursor.execute("""
                SET graph_path = 'test'
            """)
            await cursor.execute("""
                CREATE (a:Person {name: 'Alice', age: 30}),
                       (b:Person {name: 'Bob', age: 25}),
                       (c:Person {name: 'Charlie', age: 35}),
                       (a)-[:FRIEND]->(b),
                       (b)-[:FRIEND]->(c)
            """)
            await conn.commit()


@pytest_asyncio.fixture(scope="function")
async def clear_data(setup):
    async with get_pool_connection(setup) as conn:
        async with conn.cursor(row_factory=namedtuple_row) as cursor:
            # Clear existing data
            await cursor.execute("""
                SET graph_path = 'test'
            """)
            await cursor.execute("""
                MATCH (n) DETACH DELETE n
            """)
            await conn.commit()
