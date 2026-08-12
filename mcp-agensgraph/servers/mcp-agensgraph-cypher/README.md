# 🔍⁉️ Agensgraph MCP Server

## 🌟 Overview

A Model Context Protocol (MCP) server implementation that provides database interaction and allows graph exploration capabilities through Agensgraph. This server enables running Cypher graph queries, analyzing complex domain data, and automatically generating business insights that can be enhanced with Claude's analysis.

## 🧩 Components

### 🛠️ Tools

The server offers these core tools:

#### 📊 Query Tools
- `read-agensgraph-cypher`
   - Execute Cypher read queries to read data from the database. Runs in a read-only
     transaction (the database rejects writes even if the keyword check is bypassed),
     and results are **paginated** so an unbounded query can't flood the context.
   - Input:
     - `query` (string): The Cypher query to execute
     - `params` (dictionary, optional): Parameters to pass to the Cypher query
     - `limit` (int, optional): max rows for this page (default 100; clamped to a
       server max, default 1000)
     - `offset` (int, optional): rows to skip — pass the response's `next_offset` to
       fetch the next page. Skipping produces and discards the rows before the page, so
       the cost grows with the offset: measured over 50,000 nodes, offset 0 took 0.062 s,
       10,000 took 1.19 s and 40,000 took 6.21 s.
     - `walk` (bool, optional): read the whole result in one pass from a server-side
       cursor instead of a page of it, and report the exact `total_rows`. Paging through a
       whole result costs the sum of the offsets: measured over 200,000 rows, pages of
       1,000 took 7.90 s and one walk took 0.70 s.
   - Returns: a JSON object
     `{ "rows": [...], "row_count", "offset", "limit", "has_more", "next_offset" }`.
     When `has_more` is true, call again with `offset = next_offset` for the next page.
     (Page sizes are configurable via `AGENSGRAPH_PAGE_SIZE` / `AGENSGRAPH_MAX_PAGE_SIZE`.)

   A response is bounded by a token budget — 10,000 tokens by default, set with
   `AGENSGRAPH_RESPONSE_TOKEN_LIMIT` or `--token-limit`, and turned off with `0`. Whole
   rows are dropped for it, never part of one, and `rows_omitted` says how many: an
   unbounded read of a graph holding text returned 874,043 bytes in a single result.

   A list of 128 items or more is replaced with `<omitted: list of N items>` rather than
   removed, so a suppressed embedding cannot be read as an absent property.

   A query that both ends in its own `SKIP`/`OFFSET`/`LIMIT` and uses `FILTER`, `NEXT` or
   `CALL ... YIELD` is refused by name: the grammar keeps those three for the top of a
   statement, so neither appending paging nor reading the query as a subquery works. Write
   the page into the query itself and leave `limit`/`offset` alone.

- `write-agensgraph-cypher`
   - Execute updating Cypher queries
   - Input:
     - `query` (string): The Cypher update query
     - `params` (dictionary, optional): Parameters to pass to the Cypher query
   - Returns: A JSON object carrying the five write counters —
     `{ insertedvertices, insertededges, deletedvertices, deletededges, updatedproperties }`
     — plus the `rows` the statement returned. A counter the statement cannot be held to is
     `null` rather than `0`. The counters are read inside the same transaction as the write,
     and nothing is raised once the commit has landed: a reply is a reply about work that
     was committed, and a failure means nothing was.

#### 🕸️ Schema Tools
- `get-agensgraph-schema`
   - What is in the graph: one entry per node label, with its exact node count, its
     properties, and the relationship types leaving it.
   - No input required
   - Returns: `{ "<Label>": { "type": "node", "count", "properties": { "<key>": { "type",
     "declared", "indexed", "unique" } }, "relationships": { "<TYPE>": { "direction",
     "labels": [...], "properties", "count" } } } }`. A label holding nothing is left out.
   - Counts, relationships, indexes and constraints are exact. Property types come from a
     bounded sample of each label (`AGENSGRAPH_SCHEMA_SAMPLE`, default 1000) except where a
     property has a column of its own, which the catalog answers exactly. Nothing scans the
     graph: the triples come from the catalog the planner keeps, falling back to a scan of
     the edges only when nothing has gathered it since the last write. Measured on a graph
     of 138,619 vertices and 274,748 edges: 0.86 s reading it by hand, 0.22 s this way.
   - Both places uniqueness is kept are read — a property index and a uniqueness assertion,
     which the server keeps as an exclusion constraint and the property-index view hides.
   - **Nothing is installed.** A full start leaves `pg_proc` and `pg_class` the size they
     were, which the tests assert.

#### ⚡ Performance Tools
- `explain-agensgraph-cypher`
   - Show how AgensGraph would run a Cypher statement. `EXPLAIN` accepts Cypher directly,
     and without `analyze` the statement is only planned, never executed.
   - Input:
     - `query` (string): The Cypher statement to plan
     - `analyze` (bool, optional): also run it and report actual timings. It runs inside a
       read-only transaction either way, so a statement that would write is refused by the
       server with `25006` and leaves nothing behind.
   - Returns: the plan as JSON

- `recommend-property-indexes`
   - Suggest property indexes and rewrites for a query, from its plan and the label
     catalogs. Reports DDL to consider; it never runs it.
   - Input: `query` (string), `params` (dictionary, optional — a list bound as a parameter
     plans as a jsonb containment test rather than as an index lookup, so advice about a
     query with parameters is advice about a different plan without them)
   - Returns: `{ "findings": [...], "verified": false, "note": ..., "existing_indexes": [...] }`

   A property tested only with `STARTS WITH` is never also recommended an index: with the
   index built the same query still plans `Seq Scan ... Filter: string_starts_with(...)`,
   so the two pieces of advice would contradict each other. Another property in the same
   filter still gets its index.

   Alongside missing indexes it flags two shapes that cannot reach an index at all:
   `STARTS WITH`, which compiles to `string_starts_with` and reads the whole label
   however selective the prefix is (AGV2-514), and the jsonb containment test an `IN`
   list becomes once bound as a parameter (AGV2-515). Both are reported with the rewrite
   that does use an index.

   **Recommendations are reasoned, not measured**, and say so. A property index cannot be
   costed before it is built: a hypothetical index has to be supplied as a plain
   `CREATE INDEX`, and the expression a property index carries —
   `properties.'key'::text` — cannot be written that way. An index written with the jsonb
   operators instead is not the expression a Cypher filter matches, so the planner ignores
   it. Build the index and compare `EXPLAIN` before and after to confirm.

- `agensgraph-health`
   - Cache hit ratio, unused indexes, vacuum backlog, connection use, `auto_gather_graphmeta`,
     and which optional extensions are installed. Each check stands on its own — one whose
     extension is absent reports that rather than failing the others.
   - No input required

- `top-cypher-queries`
   - The Cypher statements costing the most total time, from `pg_stat_statements`. Literals
     appear as parameters, since Cypher is normalised the same way SQL is.
   - Input: `limit` (int, optional, default 20)
   - Requires `pg_stat_statements`; reports how to enable it when absent.

### 🏷️ Namespacing

The server supports namespacing to allow multiple Agensgraph MCP servers to be used simultaneously. When a namespace is provided, all tool names are prefixed with the namespace followed by a hyphen (e.g., `mydb-read-agensgraph-cypher`).

This is useful when you need to connect to multiple Agensgraph databases or instances from the same session.

## 🔧 Usage with Claude Desktop

### 💾 Released Package

Can be found on PyPi https://pypi.org/project/mcp-agensgraph-cypher/

Add the server to your `claude_desktop_config.json` with the database connection configuration through environment variables. You may also specify the transport method and namespace with cli arguments or environment variables.

If running locally, use the following configuration after running `uv sync` in the server directory:

```json
"mcpServers": {
  "agensgraph-cypher": {
    "command": "uv",
    "args": [
      "--directory",
      "/path/to/agensgraph-ai/mcp-agensgraph/servers/mcp-agensgraph-cypher",
      "run",
      "mcp-agensgraph-cypher"
    ],
    "env": {
      "AGENSGRAPH_URL": "postgresql://<host>:<port>",
      "AGENSGRAPH_USERNAME": "<your-username>",
      "AGENSGRAPH_PASSWORD": "<your-password>",
      "AGENSGRAPH_DATABASE": "<dbname>",
      "AGENSGRAPH_GRAPHNAME": "<graphname>"
    }
  }
}
```

Alternatively, using the released package(not available at the moment):
```json
"mcpServers": {
  "agensgraph-cypher": {
    "command": "uvx",
    "args": [ "mcp-agensgraph-cypher@0.2.0", "--transport", "stdio"  ],
    "env": {
      "AGENSGRAPH_URL": "postgresql://<host>:<port>",
      "AGENSGRAPH_USERNAME": "<your-username>",
      "AGENSGRAPH_PASSWORD": "<your-password>",
      "AGENSGRAPH_DATABASE": "<dbname>",
      "AGENSGRAPH_GRAPHNAME": "<graphname>"
    }
  }
}
```

#### Multiple Graphs Example

Here's an example of connecting to multiple Graphs within a single agensgraph db using namespaces:

If running locally, use the following configuration after running `uv sync` in the server directory:

```json
{
  "mcpServers": {
    "graph1-agensgraph": {
      "command": "uv",
      "args": [
        "--directory",
        "/path/to/agensgraph-ai/mcp-agensgraph/servers/mcp-agensgraph-cypher",
        "run",
        "mcp-agensgraph-cypher"
      ],
      "env": {
        "AGENSGRAPH_URL": "postgresql://<host>:<port>",
        "AGENSGRAPH_USERNAME": "<your-username>",
        "AGENSGRAPH_PASSWORD": "<your-password>",
        "AGENSGRAPH_DATABASE": "<dbname>",
        "AGENSGRAPH_GRAPHNAME": "graph1"
      }
    },
    "graph2-agensgraph": {
      "command": "uv",
      "args": [
        "--directory",
        "/path/to/agensgraph-ai/mcp-agensgraph/servers/mcp-agensgraph-cypher",
        "run",
        "mcp-agensgraph-cypher"
      ],
      "env": {
        "AGENSGRAPH_URL": "postgresql://<host>:<port>",
        "AGENSGRAPH_USERNAME": "<your-username>",
        "AGENSGRAPH_PASSWORD": "<your-password>",
        "AGENSGRAPH_DATABASE": "<dbname>",
        "AGENSGRAPH_GRAPHNAME": "graph2"
      }
    }
  }
}
```

Alternatively, using the released package(not available at the moment) with namespaces:
```json
{
  "mcpServers": {
    "graph1-agensgraph": {
      "command": "uvx",
      "args": [ "mcp-agensgraph-cypher@0.2.0", "--namespace", "graph1" ],
      "env": {
        "AGENSGRAPH_URL": "postgresql://<host>:<port>",
        "AGENSGRAPH_USERNAME": "<your-username>",
        "AGENSGRAPH_PASSWORD": "<your-password>",
        "AGENSGRAPH_DATABASE": "<dbname>",
        "AGENSGRAPH_GRAPHNAME": "graph1"
      }
    },
    "graph2-agensgraph": {
      "command": "uvx",
      "args": [ "mcp-agensgraph-cypher@0.2.0", "--namespace", "graph2" ],
      "env": {
        "AGENSGRAPH_URL": "postgresql://<host>:<port>",
        "AGENSGRAPH_USERNAME": "<your-username>",
        "AGENSGRAPH_PASSWORD": "<your-password>",
        "AGENSGRAPH_DATABASE": "<dbname>",
        "AGENSGRAPH_GRAPHNAME": "graph2"
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
      "mcp-agensgraph-cypher@0.2.0",
      "--db-url",
      "postgresql://<host>:<port>",
      "--db-name",
      "<your-db-name>",
      "--username",
      "<your-username>",
      "--password",
      "<your-password>",
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
      "AGENSGRAPH_DATABASE": "<dbname>",
      "AGENSGRAPH_HOST": "localhost",
      "AGENSGRAPH_PORT": "5432",
      "AGENSGRAPH_GRAPH_NAME": "graph"
    }
  }
}
```

## 📄 License

This MCP server is licensed under the MIT License. This means you are free to use, modify, and distribute the software, subject to the terms and conditions of the MIT License. For more details, please see the LICENSE file in the project repository.
