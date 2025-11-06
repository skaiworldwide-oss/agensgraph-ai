# 🧠🕸️ AgensGraph Knowledge Graph Memory MCP Server

## 🌟 Overview

A Model Context Protocol (MCP) server implementation that provides persistent memory capabilities through AgensGraph graph database integration.

By storing information in a graph structure, this server maintains complex relationships between entities as memory nodes and enables long-term retention of knowledge that can be queried and analyzed across multiple conversations or sessions.

The MCP server leverages AgensGraph's graph database capabilities to create an interconnected knowledge base that serves as an external memory system. Through Cypher queries, it allows exploration and retrieval of stored information, relationship analysis between different data points, and generation of insights from the accumulated knowledge.

### 🕸️ Graph Schema

* `Memory` - A node representing an entity with a name, type, and observations.
* `Relationship` - A relationship between two entities with a type.

## 📦 Components

### 🔧 Tools

The server offers these core tools:

#### 🔎 Query Tools
- `read_graph`
   - Read the entire knowledge graph
   - No input required
   - Returns: Complete graph with entities and relations

- `search_memories`
   - Search for nodes based on a query
   - Input:
     - `query` (string): Search query matching names, types, observations
   - Returns: Matching subgraph

- `find_memories_by_name`
   - Find specific nodes by name
   - Input:
     - `names` (array of strings): Entity names to retrieve
   - Returns: Subgraph with specified nodes

#### ♟️ Entity Management Tools
- `create_entities`
   - Create multiple new entities in the knowledge graph
   - Input:
     - `entities`: Array of objects with:
       - `name` (string): Name of the entity
       - `type` (string): Type of the entity
       - `observations` (array of strings): Initial observations about the entity
   - Returns: Created entities

- `delete_entities`
   - Delete multiple entities and their associated relations
   - Input:
     - `entityNames` (array of strings): Names of entities to delete
   - Returns: Success confirmation

#### 🔗 Relation Management Tools
- `create_relations`
   - Create multiple new relations between entities
   - Input:
     - `relations`: Array of objects with:
       - `source` (string): Name of source entity
       - `target` (string): Name of target entity
       - `relationType` (string): Type of relation
   - Returns: Created relations

- `delete_relations`
   - Delete multiple relations from the graph
   - Input:
     - `relations`: Array of objects with same schema as create_relations
   - Returns: Success confirmation

#### 📝 Observation Management Tools
- `add_observations`
   - Add new observations to existing entities
   - Input:
     - `observations`: Array of objects with:
       - `entityName` (string): Entity to add to
       - `contents` (array of strings): Observations to add
   - Returns: Added observation details

- `delete_observations`
   - Delete specific observations from entities
   - Input:
     - `deletions`: Array of objects with:
       - `entityName` (string): Entity to delete from
       - `observations` (array of strings): Observations to remove
   - Returns: Success confirmation

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
      "--password",
      "<your-password>",
      "--database",
      "<your-database>",
      "--graphname",
      "memory"
    ]
  }
}
```

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
      "AGENSGRAPH_DB": "<your-database>",
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
| `AGENSGRAPH_PASSWORD`                   | `agens`                                 | AgensGraph password                                |
| `AGENSGRAPH_DB`                         | `agens`                                 | AgensGraph database name                           |
| `AGENSGRAPH_GRAPH_NAME`                 | `memory`                                | AgensGraph graph name                              |
| `AGENSGRAPH_TRANSPORT`                  | `stdio` (local), `http` (remote)        | Transport protocol (`stdio`, `http`, or `sse`)     |
| `AGENSGRAPH_MCP_SERVER_HOST`            | `127.0.0.1` (local)                     | Host to bind to                                    |
| `AGENSGRAPH_MCP_SERVER_PORT`            | `8000`                                  | Port for HTTP/SSE transport                        |
| `AGENSGRAPH_MCP_SERVER_PATH`            | `/mcp/`                                 | Path for accessing MCP server                      |
| `AGENSGRAPH_MCP_SERVER_ALLOW_ORIGINS`   | _(empty - secure by default)_           | Comma-separated list of allowed CORS origins       |
| `AGENSGRAPH_MCP_SERVER_ALLOWED_HOSTS`   | `localhost,127.0.0.1`                   | Comma-separated list of allowed hosts (DNS rebinding protection) |
| `AGENSGRAPH_NAMESPACE`                  | _(empty - no prefix)_                   | Namespace prefix for tool names (e.g., `myapp-read_graph`) |

## 📄 License

This MCP server is licensed under the MIT License. This means you are free to use, modify, and distribute the software, subject to the terms and conditions of the MIT License. For more details, please see the LICENSE file in the project repository.
