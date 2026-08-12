# mcp-agensgraph-common

Shared core for the AgensGraph MCP servers
([cypher](../mcp-agensgraph-cypher), [memory](../mcp-agensgraph-memory),
[data-modeling](../mcp-agensgraph-data-modeling)). It exists so the three servers
don't each re-implement (and drift on) the same plumbing.

## What's in it

| Module | Responsibility |
|--------|----------------|
| `config` | Merge CLI args + env vars → config dicts (`connection_config`, `transport_config`, `read_controls`, `pool_config`). Standardized env var names. |
| `connection` | DSN building, the pool's lifetime, `ensure_graph`, `run_query` and `read_page`. *(needs the `db` extra)* |
| `safety` | `quote_identifiers`, which keeps the case of a label a model wrote. |
| `results` | A result as JSON a model can read: `rows_of` / `mapping_rows`, `value_sanitize`, `fit_rows`. |
| `transport` | One `run_server` for stdio / Streamable HTTP / SSE with CORS + TrustedHost middleware. |

Nothing is re-exported from the package root; each server imports the module it needs, which
is what keeps the DB-less data-modeling server from importing psycopg to read a namespace out
of the environment.

## What the pool is told once

The graph to read, how long a statement may take, and how many connections there may be. All
three were per call before, and the first two were a round trip each: a read tool call is 5.00
client-to-server flushes now against 6.00. The graph has to exist before the pool opens, since
each connection selects it as it is made — `ensure_graph(dsn, name)` is that, on a connection
of its own.

`AGENSGRAPH_POOL_MIN_SIZE` and `AGENSGRAPH_POOL_MAX_SIZE` are how the size is found. Both ends
are given values on purpose: a pool told only the lower one is that wide at the top as well,
and a trivial read was measured waiting 7.77 s behind eighteen slow ones for it.

## What a read is allowed to do

`run_query(..., read_only=True)` and `read_page` run the statement inside the driver's
`read_only_transaction`, so the refusal is the **server's**: a Cypher write, an `INSERT`, a
`TRUNCATE` and a `DROP` are each refused with `25006` and leave nothing behind. The block ends
by rolling back, because a transaction that could not write has nothing to commit and the one
thing it *can* do is `SET` — committed, that would belong to whoever borrows the connection
next.

**One thing a read-only transaction does not stop is `COPY ... TO PROGRAM`**, which runs a
command on the database server's host: it takes rows out rather than putting any in, so there
is no write to refuse, and reading the statement does not stop it either. What stops it is not
holding the privilege. `check_role_cannot_run_programs` asks once, at startup, and a server
connected as a superuser or a member of `pg_execute_server_program` refuses to start unless
`--allow-server-programs` accepts it.

Deciding by reading the statement is not done here at all. `agensgraph.cypher`'s
`writable_counters` and `check_single_statement` are what the servers ask about text.

## Install

The DB-backed servers depend on `mcp-agensgraph-common[db]`; the DB-less
data-modeling server depends on `mcp-agensgraph-common` (no psycopg).
