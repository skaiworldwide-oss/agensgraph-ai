"""LangChain integration for AgensGraph."""

from langchain_agensgraph._core.engine import AgensEngine, package_version
from langchain_agensgraph.chains import AgensCypherQAChain, create_cypher_tool
from langchain_agensgraph.chat_message_histories import AgensChatMessageHistory
from langchain_agensgraph.graph_transformers import LLMGraphTransformer
from langchain_agensgraph.graphs import AgensGraph, GraphDocument, Node, Relationship
from langchain_agensgraph.observability import log_queries
from langchain_agensgraph.retrievers import (
    AgensGraphContextRetriever,
    AgensText2CypherRetriever,
    AgensVectorRetriever,
)
from langchain_agensgraph.vectorstores import AgensgraphVector

__version__ = package_version()

__all__ = [
    "AgensChatMessageHistory",
    "AgensCypherQAChain",
    "AgensEngine",
    "AgensGraph",
    "AgensGraphContextRetriever",
    "AgensText2CypherRetriever",
    "AgensVectorRetriever",
    "AgensgraphVector",
    "GraphDocument",
    "LLMGraphTransformer",
    "Node",
    "Relationship",
    "create_cypher_tool",
    "log_queries",
    "__version__",
]

# The LangGraph checkpointer and stores need the langgraph extra; the rest imports without it.
try:
    from langchain_agensgraph.checkpoint import AgensSaver
    from langchain_agensgraph.store import AgensStore, AsyncAgensStore
except ImportError:  # pragma: no cover
    pass
else:
    __all__ += ["AgensSaver", "AgensStore", "AsyncAgensStore"]
