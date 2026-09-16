# 🔍⁉️ Agensgraph MCP Server

## 🌟 Overview

A Model Context Protocol (MCP) server implementation that provides database interaction and allows graph exploration capabilities through Agensgraph. This server enables running Cypher graph queries, analyzing complex domain data, and automatically generating business insights that can be enhanced with Claude's analysis.

## 🧩 Components

### 🛠️ Tools

The server offers these core tools:

#### 📊 Query Tools
- `read_agensgraph_cypher`
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

- `write_agensgraph_cypher`
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

   **A parameter is written `%(name)s`, not `$name`.** Both tools take `params` as a map and
   bind it the way the database driver binds one, so the placeholder in the statement is the
   driver's: `UNWIND %(records)s AS record` with `params: {"records": [...]}`. `$records` is
   not a placeholder this server rewrites — it reaches the server as written and comes back
   `ERROR: syntax error at or near "$"`. The data-modeling server generates ingest queries in
   this form already, so a generated query and a hand-written one take parameters the same way.
   A list or a map bound this way arrives as JSONB, which is what `UNWIND` and `IN` expect.

#### 🕸️ Schema Tools
- `get_agensgraph_schema`
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
- `explain_agensgraph_cypher`
   - Show how AgensGraph would run a Cypher statement. `EXPLAIN` accepts Cypher directly,
     and without `analyze` the statement is only planned, never executed.
   - Input:
     - `query` (string): The Cypher statement to plan
     - `analyze` (bool, optional): also run it and report actual timings. It runs inside a
       read-only transaction either way, so a statement that would write is refused by the
       server with `25006` and leaves nothing behind.
   - Returns: the plan as JSON

- `recommend_property_indexes`
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

- `agensgraph_health`
   - Cache hit ratio, unused indexes, vacuum backlog, connection use, `auto_gather_graphmeta`,
     and which optional extensions are installed. Each check stands on its own — one whose
     extension is absent reports that rather than failing the others.
   - No input required

- `top_cypher_queries`
   - The Cypher statements costing the most total time, from `pg_stat_statements`. Literals
     appear as parameters, since Cypher is normalised the same way SQL is.
   - Input: `limit` (int, optional, default 20)
   - Requires `pg_stat_statements`; reports how to enable it when absent.

### 🔒 Read-only mode

`--read-only`, or `AGENSGRAPH_READ_ONLY=true`, starts the server without the write tool. Six
tools remain — `get_agensgraph_schema`, `read_agensgraph_cypher`, `explain_agensgraph_cypher`,
`recommend_property_indexes`, `agensgraph_health` and `top_cypher_queries` — and the write tool
is not registered, so a client cannot call it by name either. A read-only server also does not
create the graph it is pointed at, because creating one is a write.

The tools that are left run their statements inside a read-only transaction, which is where the
guarantee actually lives: a write is refused by the database with `25006` whatever the text
looked like. `explain_agensgraph_cypher` with `analyze` is the case that shows the difference —
it executes what it is given without reading it, and the transaction refuses the write.

One privilege escapes it, so the server refuses to start holding it: a role that can run
`COPY ... TO PROGRAM` runs a command on the database server's host, which takes rows out rather
than putting any in, so a read-only transaction has no write to refuse. Connect as a role that
is neither a superuser nor a member of `pg_execute_server_program`, or pass
`--allow-server-programs` to accept it deliberately.

### 🏷️ Namespacing

The server supports namespacing to allow multiple Agensgraph MCP servers to be used simultaneously. When a namespace is provided, all tool names are prefixed with the namespace followed by a hyphen (e.g., `mydb-read_agensgraph_cypher`).

This is useful when you need to connect to multiple Agensgraph databases or instances from the same session.

## 🔧 Usage with Claude Desktop

### 💾 Released Package

On PyPI: https://pypi.org/project/mcp-agensgraph-cypher/

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

Alternatively, using the released package:
```json
"mcpServers": {
  "agensgraph-cypher": {
    "command": "uvx",
    "args": [ "mcp-agensgraph-cypher@0.3.0", "--transport", "stdio"  ],
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

Alternatively, using the released package with namespaces:
```json
{
  "mcpServers": {
    "graph1-agensgraph": {
      "command": "uvx",
      "args": [ "mcp-agensgraph-cypher@0.3.0", "--namespace", "graph1" ],
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
      "args": [ "mcp-agensgraph-cypher@0.3.0", "--namespace", "graph2" ],
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
- The graph1 graph tools will be prefixed with `graph1-` (e.g., `graph1-read_agensgraph_cypher`)
- The graph2 database tools will be prefixed with `graph2-` (e.g., `graph2-get_agensgraph_schema`)

Syntax with `--db-url`, `--username`, `--password` and other command line arguments is still supported but environment variables are preferred.

<details>
  <summary>Command-line syntax</summary>

```json
"mcpServers": {
  "agensgraph": {
    "command": "uvx",
    "args": [
      "mcp-agensgraph-cypher@0.3.0",
      "--db-url",
      "postgresql://<host>:<port>",
      "--database",
      "<your-db-name>",
      "--username",
      "<your-username>",
      "--graphname",
      "<graphname>",
      "--namespace",
      "mydb"
    ]
  }
}
```

A password on the command line is readable by every process on the machine, through `ps`, so
there is no `--password` here: leave it out and libpq resolves it the way it resolves
everything else -- `PGPASSWORD`, `.pgpass`, `PGSERVICE`, or the authentication method that
needs none.

`--transport` is left out too. It defaults to `stdio`, which is what Claude Desktop spawns and
the only transport that is not reachable by another process on the machine. The HTTP and SSE
transports have **no authentication**, and none is available to configure: `initialize` and
`tools/list` are answered without a credential, so anything that can reach the port can run
every tool this server exposes with its database credentials. Binding them to `0.0.0.0`
publishes that to the network and also defeats the one check that is there, since the trusted-
host middleware is satisfied by a `Host: localhost` header the caller writes themselves. If you
serve HTTP, keep the loopback default and put something that authenticates in front of it.

</details>

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
git clone https://github.com/yourusername/agensgraph-ai.git
cd mcp-agensgraph/servers/mcp-agensgraph-cypher

# Create and activate virtual environment using uv
uv venv
source .venv/bin/activate  # On Unix/macOS
.venv\Scripts\activate     # On Windows

# Install dependencies including dev dependencies
uv sync
```

3. Run Integration Tests

```bash
./test.sh
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
      "AGENSGRAPH_URL": "postgresql://localhost:5432",
      "AGENSGRAPH_USERNAME": "<your-username>",
      "AGENSGRAPH_PASSWORD": "<your-password>",
      "AGENSGRAPH_DATABASE": "<dbname>",
      "AGENSGRAPH_GRAPHNAME": "graph"
    }
  }
}
```

## 📄 License

This MCP server is licensed under the Apache License 2.0, which is what `LICENSE`, `NOTICE` and the package metadata declare. You are free to use, modify and distribute it subject to that licence; see `LICENSE` for the terms.
