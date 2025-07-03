# LightRag Knowledge Graph Integration: AgensGraph

This plugin adds support for storing and querying knowledge graphs in [AgensGraph](https://github.com/skaiworldwide-oss/agensgraph) with [LightRAG](https://lightrag.github.io/).

## Build

You can use the following commands to build the plugin from source.

```bash
pip install poetry
poetry install
poetry build
```

## Usage

```
pip install lightrag-hku lightrag_agensgraph-0.1.0-py3-none-any.whl
```

```python
from lightrag import LightRag
import lighrag_agensgraph

os.environ["AGENSGRAPH_DB"] = ""
os.environ["AGENSGRAPH_USER"] = ""
os.environ["AGENSGRAPH_PASSWORD"] = ""
os.environ["AGENSGRAPH_HOST"] = ""
os.environ["AGENSGRAPH_PORT"] = ""
os.environ["AGENSGRAPH_GRAPHNAME"] = ""

rag = LightRAG(
    graph_storage="AgensgraphStorage",
    ...
)
```

See [examples](./examples/) and [tests](./tests/) for more details on how to use the plugin with LightRAG.
