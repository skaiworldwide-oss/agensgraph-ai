from cognee.infrastructure.databases.graph.config import get_graph_config
from cognee.infrastructure.databases.graph.get_graph_engine import get_graph_engine
from cognee.infrastructure.databases.graph.use_graph_adapter import use_graph_adapter
from cognee_agensgraph.infrastructure.databases.graph.agensgraph.adapter import AgensgraphAdapter

use_graph_adapter("agensgraph", AgensgraphAdapter)
