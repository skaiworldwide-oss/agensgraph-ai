# 🔍⁉️ Agensgraph MCP Server

## 🌟 Overview

A Model Context Protocol (MCP) server implementation that provides database interaction and allows graph exploration capabilities through Agensgraph. This server enables running Cypher graph queries, analyzing complex domain data, and automatically generating business insights that can be enhanced with Claude's analysis.

## 🧩 Components

### 🛠️ Tools

The server offers these core tools:

#### 📊 Query Tools
- `read-agensgraph-cypher`
   - Execute Cypher read queries to read data from the database
   - Input: 
     - `query` (string): The Cypher query to execute
     - `params` (dictionary, optional): Parameters to pass to the Cypher query
   - Returns: Query results as JSON serialized array of objects

- `write-agensgraph-cypher`
   - Execute updating Cypher queries
   - Input:
     - `query` (string): The Cypher update query
     - `params` (dictionary, optional): Parameters to pass to the Cypher query
   - Returns: A JSON serialized result summary counter with `{ insertedvertices: number, insertededges: number, ... }`

#### 🕸️ Schema Tools
- `get-agensgraph-schema`
   - Get a list of all nodes types in the graph database, their attributes with name, type and relationships to other node types
   - No input required
   - Returns: JSON serialized list of node labels with two dictionaries: one for attributes and one for relationships

### 🏷️ Namespacing

The server supports namespacing to allow multiple Agensgraph MCP servers to be used simultaneously. When a namespace is provided, all tool names are prefixed with the namespace followed by a hyphen (e.g., `mydb-read-agensgraph-cypher`).

This is useful when you need to connect to multiple Agensgraph databases or instances from the same session.

## 🔧 Usage with Claude Desktop

### 💾 Released Package

Can be found on PyPi https://pypi.org/project/mcp-agensgraph-cypher/

Add the server to your `claude_desktop_config.json` with the database connection configuration through environment variables. You may also specify the transport method and namespace with cli arguments or environment variables.

```json
"mcpServers": {
  "agensgraph-cypher": {
    "command": "uvx",
    "args": [ "mcp-agensgraph-cypher@0.1.0", "--transport", "stdio"  ],
    "env": {
      "AGENSGRAPH_USERNAME": "<your-username>",
      "AGENSGRAPH_PASSWORD": "<your-password>",
      "AGENSGRAPH_DB": "<dbname>",
      "AGENSGRAPH_HOST": "localhost",
      "AGENSGRAPH_PORT": "5432",
      "AGENSGRAPH_GRAPH_NAME": "graph",
    }
  }
}
```

#### Multiple Graphs Example

Here's an example of connecting to multiple Graphs within a single agensgraph db using namespaces:

```json
{
  "mcpServers": {
    "graph1-agensgraph": {
      "command": "uvx",
      "args": [ "mcp-agensgraph-cypher@0.1.0", "--namespace", "graph1" ],
      "env": {
        "AGENSGRAPH_USERNAME": "<your-username>",
        "AGENSGRAPH_PASSWORD": "<your-password>",
        "AGENSGRAPH_DB": "<dbname>",
        "AGENSGRAPH_HOST": "localhost",
        "AGENSGRAPH_PORT": "5432",
        "AGENSGRAPH_GRAPH_NAME": "graph1"
      }
    },
    "graph2-agensgraph": {
      "command": "uvx",
      "args": [ "mcp-agensgraph-cypher@0.1.0", "--namespace", "graph2" ],
      "env": {
        "AGENSGRAPH_USERNAME": "<your-username>",
        "AGENSGRAPH_PASSWORD": "<your-password>",
        "AGENSGRAPH_DB": "<dbname>",
        "AGENSGRAPH_HOST": "localhost",
        "AGENSGRAPH_PORT": "5432",
        "AGENSGRAPH_GRAPH_NAME": "graph2"
      }
    }
  }
}
```

In this setup:
- The graph1 graph tools will be prefixed with `graph1-` (e.g., `graph1-read-agensgraph-cypher`)
- The graph2 database tools will be prefixed with `graph2-` (e.g., `graph2-get-agensgraph-schema`)

Syntax with `--db-url`, `--username`, `--password` and other command line arguments is still supported but environment variables are preferred.

<details>
  <summary>Legacy Syntax</summary>

```json
"mcpServers": {
  "agensgraph": {
    "command": "uvx",
    "args": [
      "mcp-agensgraph-cypher@0.1.0",
      "--db-name",
      "<your-db-name>",
      "--username"
      "<your-username>",
      "--password",
      "<your-password>",
      "--db-host",
      "localhost",
      "--db-port",
      "5432",
      "--namespace",
      "mydb",
      "--transport",
      "sse",
      "--server-host",
      "0.0.0.0",
      "--server-port",
      "8000"
    ]
  }
}
```

</details>

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
# Clone the repository
git clone https://github.com/yourusername/agensgraph-ai.git
cd mcp-agensgraph/servers/mcp-agensgraph-cypher

# Create and activate virtual environment using uv
uv venv
source .venv/bin/activate  # On Unix/macOS
.venv\Scripts\activate     # On Windows

# Install dependencies including dev dependencies
uv pip install -e ".[dev]"
```

3. Run Integration Tests

```bash
./tests.sh
```

### 🔧 Development Configuration

```json
# Add the server to your claude_desktop_config.json
"mcpServers": {
  "agensgraph": {
    "command": "uv",
    "args": [
      "--directory", 
      "parent_of_servers_repo/servers/mcp-agensgraph-cypher/src",
      "run", 
      "mcp-agensgraph-cypher", 
      "--transport", 
      "stdio", 
      "--namespace", 
      "dev",
    ],
    "env": {
      "AGENSGRAPH_USERNAME": "<your-username>",
      "AGENSGRAPH_PASSWORD": "<your-password>",
      "AGENSGRAPH_DB": "<dbname>",
      "AGENSGRAPH_HOST": "localhost",
      "AGENSGRAPH_PORT": "5432",
      "AGENSGRAPH_GRAPH_NAME": "graph"
    }
  }
}
```

## 📄 License

This MCP server is licensed under the MIT License. This means you are free to use, modify, and distribute the software, subject to the terms and conditions of the MIT License. For more details, please see the LICENSE file in the project repository.
