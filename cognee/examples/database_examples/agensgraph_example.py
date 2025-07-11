import os
import pathlib
import asyncio
import cognee
from cognee.modules.search.types import SearchType


async def main():
    """
    Example script demonstrating how to use Cognee with agensgraph

    This example:
    1. Configures Cognee to use agensgraph as graph database
    2. Sets up data directories
    3. Adds sample data to Cognee
    4. Processes (cognifies) the data
    5. Performs different types of searches
    """

    # Set up agensgraph credentials in .env file and get the values from environment variables
    agensgraph_url = os.getenv("GRAPH_DATABASE_URL")

    # Configure agensgraph as the graph database provider
    cognee.config.set_graph_db_config(
        {
            "graph_database_url": agensgraph_url,  # agensgraph connection DSN
            "graph_database_provider": "agensgraph",  # Specify agensgraph as provider
        }
    )

    # Set up data directories for storing documents and system files
    # You should adjust these paths to your needs
    current_dir = pathlib.Path(__file__).parent
    data_directory_path = str(current_dir / "data_storage")
    cognee.config.data_root_directory(data_directory_path)

    cognee_directory_path = str(current_dir / "cognee_system")
    cognee.config.system_root_directory(cognee_directory_path)

    # Clean any existing data (optional)
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    # Create a dataset
    dataset_name = "agensgraph_example"

    # Add sample text to the dataset
    sample_text = """AgensGraph is a cutting-edge multi-model graph database designed for modern
    complex data environments. By supporting both relational and graph data models simultaneously,
    AgensGraph allows developers to seamlessly integrate legacy relational data with the flexible
    graph data model within a single database. AgensGraph is built on the robust PostgreSQL RDBMS,
    providing a highly reliable, fully-featured platform ready for enterprise use."""

    # Add the sample text to the dataset
    await cognee.add([sample_text], dataset_name)

    # Process the added document to extract knowledge
    await cognee.cognify([dataset_name])

    # Now let's perform some searches
    # 1. Search for insights related to "agensgraph"
    insights_results = await cognee.search(query_type=SearchType.INSIGHTS, query_text="agensgraph")
    print("\nInsights about agensgraph:")
    for result in insights_results:
        print(f"- {result}")

    # 2. Search for text chunks related to "graph database"
    chunks_results = await cognee.search(
        query_type=SearchType.CHUNKS, query_text="graph database", datasets=[dataset_name]
    )
    print("\nChunks about graph database:")
    for result in chunks_results:
        print(f"- {result}")

    # 3. Get graph completion related to databases
    graph_completion_results = await cognee.search(
        query_type=SearchType.GRAPH_COMPLETION, query_text="database"
    )
    print("\nGraph completion for databases:")
    for result in graph_completion_results:
        print(f"- {result}")

    # Clean up (optional)
    # await cognee.prune.prune_data()
    # await cognee.prune.prune_system(metadata=True)


if __name__ == "__main__":
    asyncio.run(main())
