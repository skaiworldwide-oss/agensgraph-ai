# Langchain Graph Store Integration: AgensGraph

This plugin provides support for GraphStore integration in [langchain](https://www.langchain.com/), for persisting graphs directly in [AgensGraph](https://github.com/bitnine-oss/agensgraph).

See the associated guide below:

- [Agens Graph Store](./examples/agensgraph.ipynb)

## Build

You can use the following commands to build the plugin from source.

```bash
pip install poetry
poetry install
poetry build
```

## Usage

```
pip install langchain langchain_graph_store_agensgraph-0.1.0-py3-none-any.whl
```

```python
from langchain.graphs.agensgraph import AgensGraph

conf = {
    "database": "",
    "user": "",
    "password": "",
    "host": "",
    "port": ,
}

graph = AgensGraph(graph_name="", conf=conf)
```