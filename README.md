# AgensGraph AI

Integrations, tools and starter material for building AI applications on
[AgensGraph](https://github.com/skaiworldwide-oss/agensgraph), the PostgreSQL-based graph
database. Every integration here runs on the
[`agensgraph-python`](https://pypi.org/project/agensgraph-python/) 2.0 driver.

| Integration | Package | What it gives you |
| --- | --- | --- |
| [LangChain](https://github.com/skaiworldwide-oss/agensgraph-ai/tree/main/langchain) | `langchain-agensgraph` | a graph store and a vector store, three retrievers, a text2cypher chain, a LangGraph checkpointer and long-term memory store, chat message history |
| [LlamaIndex](https://github.com/skaiworldwide-oss/agensgraph-ai/tree/main/llama-index) | `llama-index-agensgraph` | a property graph store for `PropertyGraphIndex` and a vector store for `VectorStoreIndex` |
| [LightRAG](https://github.com/skaiworldwide-oss/agensgraph-ai/tree/main/lightrag) | `lightrag-agensgraph` | all four LightRAG storages — graph, vectors, key-value, document status — in one database |
| [cognee](https://github.com/skaiworldwide-oss/agensgraph-ai/tree/main/cognee) | `cognee-agensgraph` | cognee's graph store and vector store in one database |
| [MCP](https://github.com/skaiworldwide-oss/agensgraph-ai/tree/main/mcp-agensgraph) | `mcp-agensgraph-cypher`, `mcp-agensgraph-memory`, `mcp-agensgraph-data-modeling` | three Model Context Protocol servers: Cypher over a graph, a knowledge-graph memory, and graph data modeling |

Each directory has a README of its own with setup and usage, and an `examples/demos/` suite
that runs on real datasets.

## Installation

The `agensgraph-ai` package installs any combination of the integrations under one name.
Pick the ones you need:

```bash
pip install "agensgraph-ai[langchain]"
pip install "agensgraph-ai[langchain,lightrag]"
pip install "agensgraph-ai[all]"
```

| Extra | Installs | Import |
| --- | --- | --- |
| `langchain` | `langchain-agensgraph` | `langchain_agensgraph` |
| `llama-index` | `llama-index-agensgraph` | `llama_index_agensgraph` |
| `lightrag` | `lightrag-agensgraph` | `lightrag_agensgraph` |
| `cognee` | `cognee-agensgraph` | `cognee_agensgraph` |
| `mcp` | the three `mcp-agensgraph-*` servers | run as commands |
| `all` | all of the above | |

Name at least one extra. `pip install agensgraph-ai` on its own installs no integrations,
and an extra that is misspelled installs none either — `pip` warns about that, `uv` does
not.

Each integration is also released on its own, so it can be installed by name instead. The
two forms produce the same environment — `agensgraph-ai` ships no code, and is a
convenience rather than a layer:

```bash
pip install langchain-agensgraph
pip install llama-index-agensgraph
pip install lightrag-agensgraph
pip install cognee-agensgraph
```

The MCP servers are commands rather than libraries, and an MCP client normally launches
them itself with `uvx`, which needs no install at all:

```json
"mcpServers": {
  "agensgraph-cypher": {
    "command": "uvx",
    "args": ["mcp-agensgraph-cypher@0.3.0", "--transport", "stdio"]
  }
}
```

The `mcp` extra is for the other case: hosting a server yourself over HTTP or SSE.

## Requirements

- Python 3.11 or later.
- A running AgensGraph 2.17 or later; the driver refuses an older server at connect.
  `SHOW agversion` tells you which you have. From 2.18 a release reports four numbers, such
  as `2.18.4.0`, with `-rc1` on a release candidate; the first two are the line, so
  `2.18.6.0-rc1` is a 2.18 server.
- The vector-backed features need the `pgvector` extension, and schema introspection is
  faster with the `meta` extension. AgensGraph bundles neither; see
  [how to build them](https://github.com/skaiworldwide-oss/agensgraph-ai/blob/main/langchain/README.md#agensgraph-requirements).

## License

Apache License 2.0 — see
[LICENSE](https://github.com/skaiworldwide-oss/agensgraph-ai/blob/main/LICENSE).

## Contact

Open an [issue](https://github.com/skaiworldwide-oss/agensgraph-ai/issues) or a pull request.
