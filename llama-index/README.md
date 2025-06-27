# LlamaIndex Graph Store Integration: AgensGraph

This plugin provides support for AgensGraphStore integration in [llamaindex](https://www.llamaindex.ai/), for persisting graphs directly in [AgensGraph](https://github.com/skaiworldwide-oss/agensgraph). Additionally, we support the PropertyGraphIndex, which allows you to store and query property graphs in AgensGraph.

See the associated guides below:

- [Agens Graph Store](./examples/index_structs/knowledge_graph/AgensgraphDemo.ipynb)
- [Agens Property Graph Store](./examples//property_graph/property_graph_agensgraph.ipynb)


## Build

You can use the following commands to build the plugin from source.

```bash
pip install poetry
poetry install
poetry build
```

## Usage

```
pip install llama-index llama_index_graph_stores_agensgraph
```

### AgensGraphStore
```python
from llama_index.graph_stores.agensgraph import AgensGraphStore

conf = {
    "database": "",
    "user": "",
    "password": "",
    "host": "",
    "port": ,
}

graph = AgensGraphStore(graph_name="", conf=conf, create=True)
```

### AgensPropertyGraphStore
```python
from llama_index.graph_stores.agensgraph import AgensPropertyGraphStore

conf = {
    "database": "",
    "user": "",
    "password": "",
    "host": "",
    "port": ,
}

graph = AgensPropertyGraphStore(graph_name="", conf=conf, create=True)
```