# 🧠🕸️ AgensGraph Knowledge Graph Memory MCP Server

## 🌟 Overview

A Model Context Protocol (MCP) server implementation that provides persistent memory capabilities through AgensGraph graph database integration.

By storing information in a graph structure, this server maintains complex relationships between entities as memory nodes and enables long-term retention of knowledge that can be queried and analyzed across multiple conversations or sessions.

The MCP server leverages AgensGraph's graph database capabilities to create an interconnected knowledge base that serves as an external memory system. Through Cypher queries, it allows exploration and retrieval of stored information, relationship analysis between different data points, and generation of insights from the accumulated knowledge.

### 🕸️ Graph Schema

* `Memory` - a vertex label holding an entity's `name`, `type` and `observations`.
  A **unique** property index on `name` is what makes an entity's name identify one
  vertex, so two callers writing the same entity write one.
* One edge label per relationship type, named by the type upper-cased — `WORKS_AT`,
  `LIVES_IN`. A type given in any other case names the same relationship. Each label
  carries a unique index on the pair of ends it joins, so one relationship of a type
  between two entities is one edge.
* A GIN index over `to_tsvector('english', ...)` of the name, type and observations,
  which `search_memories` reads. It is built out of PostgreSQL's own functions; the
  server installs nothing in the database.

All of these are made at startup and checked afterwards. Starting up over a store
written before they existed merges entities that share a name (unioning their
observations and keeping their relationships), moves relationships off a miscased
type, collapses duplicate relationships onto the one written first, and rebuilds the
full-text index. What it collapsed is written to the log.

## 📦 Components

### 🔧 Tools

The server offers these core tools:

#### 🔎 Query Tools
- `read_graph`
   - Read a page of the knowledge graph
   - Input:
     - `limit` (int, optional): max entities to return (default 1000, most 1000)
   - Returns: `{ "entities": [...], "relations": [...], "truncated": bool }`. The page is
     the first `limit` entities by name, and the relations are those whose **both** ends
     are on it — nothing in the result names an entity the result does not contain. When
     `truncated` is true the memory has more; narrow with `search_memories`. (The default
     is configurable via `AGENSGRAPH_MEMORY_LIMIT`, up to the same ceiling.)

- `search_memories`
   - Find entities by words in their name, type or observations
   - Input:
     - `query` (string): words to search for. **Every** word must match — they are joined
       with AND, not OR — and they are stemmed rather than matched as prefixes, so
       `engineers` finds `engineering` but `eng` finds neither. A query of only stop words
       matches nothing. `"*"` asks for everything, which is `read_graph` by another name.
     - `limit` (int, optional): max matching entities to return (default 1000, most 1000)
   - Returns: matching subgraph as `{ "entities", "relations", "truncated" }`

- `find_memories_by_name`
   - Find entities by their exact names, with what they are connected to
   - Input:
     - `names` (array of strings): entity names to retrieve
     - `limit` (int, optional): max names to look up (default 1000, most 1000)
   - Returns: the named entities, every relationship touching them in either direction,
     and the entities at the other end of those relationships

#### ♟️ Entity Management Tools
- `create_entities`
   - Write entities, merging into any whose name is already in the memory
   - Input:
     - `entities`: Array of objects with:
       - `name` (string): Name of the entity
       - `type` (string): Type of the entity
       - `observations` (array of strings): Observations about the entity
   - An entity already in the memory **keeps its observations and gains the ones given**;
     its type is set to the one given.
   - Returns: the entities as they now stand, read back from the memory

- `delete_entities`
   - Delete entities and every relationship they take part in
   - Input:
     - `entityNames` (array of strings): Names of entities to delete
   - Returns: `{ "deleted": [...], "notFound": [...], "deletedRelations": int }`

#### 🔗 Relation Management Tools
- `create_relations`
   - Write relationships between entities that are already in the memory
   - Input:
     - `relations`: Array of objects with:
       - `source` (string): Name of source entity
       - `target` (string): Name of target entity
       - `relationType` (string): Type of relation, stored upper-cased
   - A relationship whose source or target is not in the memory is not written.
   - Returns: `{ "created": [...], "skipped": [...] }`

- `delete_relations`
   - Delete relationships, keeping the entities
   - Input:
     - `relations`: Array of objects with the same schema as create_relations
   - Returns: `{ "requested": int, "deletedRelations": int }`

#### 📝 Observation Management Tools
- `add_observations`
   - Add observations to entities already in the memory
   - Input:
     - `observations`: Array of objects with:
       - `entityName` (string): Entity to add to
       - `observations` (array of strings): Observations to add
   - An observation the entity already holds is not stored again.
   - Returns: per entity, `{ "entityName", "addedObservations", "found" }`

- `delete_observations`
   - Delete specific observations from entities
   - Input:
     - `deletions`: Array of objects with:
       - `entityName` (string): Entity to delete from
       - `observations` (array of strings): Observations to remove
   - Returns: per entity, `{ "entityName", "deletedObservations", "found" }`

## 🔧 Usage with Claude Desktop

### 💾 Installation

```bash
pip install mcp-agensgraph-memory
```

### ⚙️ Configuration

Add the server to your `claude_desktop_config.json`:

```json
"mcpServers": {
  "agensgraph": {
    "command": "uvx",
    "args": [
      "mcp-agensgraph-memory",
      "--db-url",
      "postgresql://localhost:5432",
      "--username",
      "<your-username>",
      "--database",
      "<your-database>",
      "--graphname",
      "memory"
    ]
  }
}
```

There is no `--password` here on purpose: a password on the command line is readable by every
process on the machine through `ps`. Leave it out and libpq resolves it — `PGPASSWORD`,
`.pgpass`, `PGSERVICE`, or an authentication method that needs none — or pass it in the
environment as below.

Alternatively, you can set environment variables:

```json
"mcpServers": {
  "agensgraph": {
    "command": "uvx",
    "args": [ "mcp-agensgraph-memory" ],
    "env": {
      "AGENSGRAPH_URL": "postgresql://localhost:5432",
      "AGENSGRAPH_USERNAME": "<your-username>",
      "AGENSGRAPH_PASSWORD": "<your-password>",
      "AGENSGRAPH_DATABASE": "<your-database>",
      "AGENSGRAPH_GRAPH_NAME": "memory"
    }
  }
}
```

#### Namespacing
For multi-tenant deployments, add `--namespace` to prefix tool names:
```json
"args": [ "mcp-agensgraph-memory", "--namespace", "myapp", "--db-url", "..." ]
```
Tools become: `myapp-read_graph`, `myapp-create_entities`, etc.

Can also use `AGENSGRAPH_NAMESPACE` environment variable.

### 🌐 HTTP Transport Mode

The server supports HTTP transport for web-based deployments and microservices:

```bash
# Basic HTTP mode (defaults: host=127.0.0.1, port=8000, path=/mcp/)
mcp-agensgraph-memory --transport http

# Custom HTTP configuration
mcp-agensgraph-memory --transport http --server-host 127.0.0.1 --server-port 8080 --server-path /api/mcp/
```

Environment variables for HTTP configuration:

```bash
export AGENSGRAPH_TRANSPORT=http
export AGENSGRAPH_MCP_SERVER_HOST=127.0.0.1
export AGENSGRAPH_MCP_SERVER_PORT=8080
export AGENSGRAPH_MCP_SERVER_PATH=/api/mcp/
export AGENSGRAPH_NAMESPACE=myapp
mcp-agensgraph-memory
```

### 🔄 Transport Modes

The server supports three transport modes:

- **STDIO** (default): Standard input/output for local tools and Claude Desktop
- **SSE**: Server-Sent Events for web-based deployments
- **HTTP**: Streamable HTTP for modern web deployments and microservices

## 🔒 Security Protection

The server includes comprehensive security protection with **secure defaults** that protect against common web-based attacks while preserving full MCP functionality when using HTTP transport.

### 🛡️ DNS Rebinding Protection

**TrustedHost Middleware** validates Host headers to prevent DNS rebinding attacks:

**Secure by Default:**
- Only `localhost` and `127.0.0.1` hosts are allowed by default

**Environment Variable:**
```bash
export AGENSGRAPH_MCP_SERVER_ALLOWED_HOSTS="example.com,www.example.com"
```

### 🌐 CORS Protection

**Cross-Origin Resource Sharing (CORS)** protection blocks browser-based requests by default:

**Environment Variable:**
```bash
export AGENSGRAPH_MCP_SERVER_ALLOW_ORIGINS="https://example.com,https://app.example.com"
```

### 🔧 Complete Security Configuration

**Development Setup:**
```bash
mcp-agensgraph-memory --transport http \
  --allowed-hosts "localhost,127.0.0.1" \
  --allow-origins "http://localhost:3000"
```

**Production Setup:**
```bash
mcp-agensgraph-memory --transport http \
  --allowed-hosts "example.com,www.example.com" \
  --allow-origins "https://example.com,https://app.example.com"
```

## 🚀 Development

### 📦 Prerequisites

A running AgensGraph 2.17 or later. The driver refuses an older server at connect.

1. Install `uv` (Universal Virtualenv):
```bash
# Using pip
pip install uv

# Using Homebrew on macOS
brew install uv

# Using cargo (Rust package manager)
cargo install uv
```

2. Clone the repository and set up development environment:
```bash
# Clone the repository
git clone https://github.com/skaiworldwide-oss/agensgraph-ai.git
cd agensgraph-ai/mcp-agensgraph/servers/mcp-agensgraph-memory

# Create and activate virtual environment using uv
uv venv
source .venv/bin/activate  # On Unix/macOS
.venv\Scripts\activate     # On Windows

# Install dependencies including dev dependencies
uv sync
```

### 🔧 Environment Variables

| Variable                                | Default                                 | Description                                        |
| --------------------------------------- | --------------------------------------- | -------------------------------------------------- |
| `AGENSGRAPH_URL`                        | `postgresql://localhost:5432`           | AgensGraph connection URL (host:port only)         |
| `AGENSGRAPH_USERNAME`                   | `agens`                                 | AgensGraph username                                |
| `AGENSGRAPH_PASSWORD`                   | _(unset — libpq resolves it)_           | AgensGraph password                                |
| `AGENSGRAPH_DATABASE`                   | `agens`                                 | AgensGraph database name (`AGENSGRAPH_DB` also read) |
| `AGENSGRAPH_GRAPH_NAME`                 | `memory`                                | AgensGraph graph name                              |
| `AGENSGRAPH_TRANSPORT`                  | `stdio` (local), `http` (remote)        | Transport protocol (`stdio`, `http`, or `sse`)     |
| `AGENSGRAPH_MCP_SERVER_HOST`            | `127.0.0.1` (local)                     | Host to bind to                                    |
| `AGENSGRAPH_MCP_SERVER_PORT`            | `8000`                                  | Port for HTTP/SSE transport                        |
| `AGENSGRAPH_MCP_SERVER_PATH`            | `/mcp/`                                 | Path for accessing MCP server                      |
| `AGENSGRAPH_MCP_SERVER_ALLOW_ORIGINS`   | _(empty - secure by default)_           | Comma-separated list of allowed CORS origins       |
| `AGENSGRAPH_MCP_SERVER_ALLOWED_HOSTS`   | `localhost,127.0.0.1`                   | Comma-separated list of allowed hosts (DNS rebinding protection) |
| `AGENSGRAPH_NAMESPACE`                  | _(empty - no prefix)_                   | Namespace prefix for tool names (e.g., `myapp-read_graph`) |
| `AGENSGRAPH_MEMORY_LIMIT`               | `1000`                                  | Default entities per read, bounded by the same ceiling |
| `AGENSGRAPH_ALLOW_SERVER_PROGRAMS`      | `false`                                 | Serve as a role that can run a command on the server's host |

Leaving `AGENSGRAPH_PASSWORD` unset is the useful default rather than an omission: an empty
password is not sent as an empty one, so libpq resolves it the way it resolves everything else
— `PGPASSWORD`, `.pgpass`, `PGSERVICE`, or an authentication method that needs none. A password
passed as `--password` on the command line is readable by every process on the machine through
`ps`; the environment variable or `.pgpass` is not.

## 📄 License

This MCP server is licensed under the Apache License 2.0, which is what `LICENSE`, `NOTICE` and the package metadata declare. You are free to use, modify and distribute it subject to that licence; see `LICENSE` for the terms.
