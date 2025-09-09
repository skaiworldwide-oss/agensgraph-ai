from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cognee_agensgraph.infrastructure.databases.graph.agensgraph.adapter import AgensgraphAdapter


async def get_edge_density(adapter: AgensgraphAdapter):
    """
    Calculate the edge density of a graph in an Agensgraph database.

    This function executes a Cypher query to determine the ratio of edges to the maximum
    possible edges in a graph, based on the number of nodes. If there are fewer than two
    nodes, it returns an edge density of zero.

    Parameters:
    -----------

        - adapter (AgensgraphAdapter): An instance of AgensgraphAdapter used to interface with the
          Agensgraph database.

    Returns:
    --------

        Returns the calculated edge density as a float, or 0 if no results are found.
    """
    query = """
    MATCH (n)
    WITH count(n) AS num_nodes
    MATCH ()-[r]->()
    WITH num_nodes, count(r) AS num_edges
    RETURN CASE
        WHEN num_nodes < 2 THEN 0
        ELSE num_edges * 1.0 / (num_nodes * (num_nodes - 1))
    END AS edge_density;
    """
    result = await adapter.query(query)
    return result[0]["edge_density"] if result else 0


async def count_self_loops(adapter: AgensgraphAdapter):
    """
    Count the number of self-loop relationships in the Agensgraph database.

    This function executes a Cypher query to find and count all edge relationships that
    begin and end at the same node (self-loops). It returns the count of such relationships
    or 0 if no results are found.

    Parameters:
    -----------

        - adapter (AgensgraphAdapter): An instance of AgensgraphAdapter used to interact with the
          Agensgraph database.

    Returns:
    --------

        The count of self-loop relationships found in the database, or 0 if none were found.
    """
    query = """
    MATCH (n)-[r]->(n)
    RETURN count(r) AS adapter_loop_count;
    """
    result = await adapter.query(query)
    return result[0]["adapter_loop_count"] if result else 0


