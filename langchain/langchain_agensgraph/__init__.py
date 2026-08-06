"""LangChain integration for AgensGraph."""

from langchain_agensgraph.chat_message_histories import AgensChatMessageHistory
from langchain_agensgraph.engine import AgensEngine
from langchain_agensgraph.graph_transformers import LLMGraphTransformer
from langchain_agensgraph.graphs.agensgraph import AgensGraph
from langchain_agensgraph.graphs.graph_document import (
    GraphDocument,
    Node,
    Relationship,
)
from langchain_agensgraph.vectorstores.agensgraph_vector import AgensgraphVector

__version__ = "0.2.0"

__all__ = [
    "AgensChatMessageHistory",
    "AgensEngine",
    "AgensGraph",
    "AgensgraphVector",
    "GraphDocument",
    "LLMGraphTransformer",
    "Node",
    "Relationship",
    "__version__",
]

# The LangGraph checkpointer and store are re-exported lazily so the package still
# imports if langgraph is not installed.
try:  # pragma: no cover - import guard
    from langchain_agensgraph.checkpoint import AgensSaver, AsyncAgensSaver
    from langchain_agensgraph.store import AgensStore

    __all__ += ["AgensSaver", "AgensStore", "AsyncAgensSaver"]
except ImportError:  # pragma: no cover
    pass
