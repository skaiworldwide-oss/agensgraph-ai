import sys, types
from lightrag_agensgraph.kg.agensgraph_impl import AgensgraphStorage
import lightrag.kg

lightrag.kg.STORAGE_IMPLEMENTATIONS["GRAPH_STORAGE"]["implementations"].append("AgensgraphStorage")
lightrag.kg.STORAGES["AgensgraphStorage"] = "lightrag_agensgraph.kg.agensgraph_impl"
