# mcp-agensgraph-common

Shared core for the AgensGraph MCP servers
([cypher](../mcp-agensgraph-cypher), [memory](../mcp-agensgraph-memory),
[data-modeling](../mcp-agensgraph-data-modeling)). It exists so the three servers
don't each re-implement (and drift on) the same plumbing.

## What's in it

| Module | Responsibility |
|--------|----------------|
| `config` | Merge CLI args + env vars → config dicts (`connection_config`, `transport_config`, `read_controls`). Standardized env var names. |
| `connection` | DSN building, `AsyncConnectionPool` lifecycle, `ensure_graph`, and `run_query` with **database-side read-only enforcement** and identifier-quoted `graph_path`. *(needs the `db` extra)* |
| `safety` | `quote_identifiers` / `quote_label` (AgensGraph case-sensitivity; safe rel-type quoting) and a comment-aware `is_write_query`. |
| `results` | Parse AgensGraph `vertex`/`edge` result strings → JSON (`record_to_dict`), `value_sanitize`, `truncate_to_tokens`. |
| `transport` | One `run_server` for stdio / Streamable HTTP / SSE with CORS + TrustedHost middleware. |

## Read-only enforcement

`run_query(..., read_only=True)` runs the statement in a `READ ONLY` transaction, so
AgensGraph rejects any Cypher write at the database level — independent of (and
stronger than) the client-side `is_write_query` keyword check, which only exists to
return a friendly error.

## Install

The DB-backed servers depend on `mcp-agensgraph-common[db]`; the DB-less
data-modeling server depends on `mcp-agensgraph-common` (no psycopg).
