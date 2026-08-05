🧠 AgensGraph AI Tools and Libraries
====================================

Welcome to [AgensGraph AI](https://github.com/skaiworldwide-oss/agensgraph-ai) — a curated collection of tools, integrations, and starter templates for building AI-powered applications that work with [AgensGraph](https://github.com/skaiworldwide-oss/agensgraph), a PostgreSQL-based multi-model graph database.

This repository currently includes:
* [LangChain](./langchain/) — LLM agents, tools, and chains
* [LlamaIndex](./llama-index/) — custom data indexing and retrieval
* [LightRag](./lightrag/) — graph-aware RAG for accurate, context-rich retrieval
* [cognee](./cognee) — dynamic memory for Agents
* [mcp](./mcp-agensgraph/) — Model Context Protocol server for AgensGraph enabling database access and graph exploration.

> ✅ Each library has its own subfolder with a dedicated README to guide you through setup and usage.

# 📦 Installation

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
    "args": ["mcp-agensgraph-cypher@0.2.0", "--transport", "stdio"]
  }
}
```

The `mcp` extra is for the other case: hosting a server yourself over HTTP or SSE.

## Database requirements

The Python install is only half of the setup. These integrations talk to a running
AgensGraph, 2.17 or newer recommended, and the vector-backed features need the `pgvector`
and `meta` extensions, which AgensGraph does not bundle — see
[langchain/README.md](./langchain/README.md#agensgraph-requirements) for how to build and
enable them.

# 🎯 Purpose
This repository is designed to help developers:
* Integrate AgensGraph with modern LLM frameworks
* Leverage graph data in conversational and intelligent apps
* Explore Retrieval-Augmented Generation (RAG), agents, and graph reasoning

Everything is open-source and modular — feel free to use, fork, or contribute.

# 📄 License
This repository is licensed under the [Apache License 2.0](./LICENSE).

# 📬 Contact
For questions, feature requests, or collaboration:
* Open an [Issue](https://github.com/skaiworldwide-oss/agensgraph-ai/issues) or Pull Request