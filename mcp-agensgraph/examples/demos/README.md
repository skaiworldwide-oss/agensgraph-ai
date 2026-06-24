# AgensGraph MCP servers — demos

Runnable demos for the three [Model Context Protocol](https://modelcontextprotocol.io)
servers in this repo — proof they're real, fast, and usable, on real data:

- **`mcp-agensgraph-cypher`** — run Cypher against AgensGraph (schema, read with
  pagination, write), with read-only mode, namespacing, and stdio/HTTP transports.
- **`mcp-agensgraph-memory`** — a persistent knowledge-graph "memory" (entities,
  relations, observations) an agent grows over a conversation.
- **`mcp-agensgraph-data-modeling`** — design graph schemas and generate the
  constraint + ingest Cypher (no database needed).

**The MCP servers are pure tools — no LLM, no API key.** So these demos are
deterministic, free, and reproducible: a FastMCP client drives the servers exactly as an
agent would. The real *LLM-using-the-tools* story is Claude Desktop — see
[`claude_desktop/`](./claude_desktop).

## Setup

```bash
cd mcp-agensgraph/examples/demos
uv venv .venv
uv pip install --python .venv/bin/python \
    -e ../../servers/mcp-agensgraph-common \
    -e ../../servers/mcp-agensgraph-cypher \
    -e ../../servers/mcp-agensgraph-memory \
    -e ../../servers/mcp-agensgraph-data-modeling \
    -r requirements-demos.txt
```

Connection defaults to the local AgensGraph (`127.0.0.1:55432`, trust auth); override via
`AGENS_HOST/PORT/USER/PASSWORD` or a `.env` (copy `.env.example`). Each demo uses its own
database (`mcp_flights`, `mcp_memory`), so nothing collides with other suites.

## Quickstart

```bash
.venv/bin/python 01_model_and_load/build.py    # design a schema + load OpenFlights
.venv/bin/python 02_cypher_query/ask.py        # query it through the cypher server
```

## The demos

| # | Server(s) | What it shows | Run |
|---|-----------|---------------|-----|
| [**01_model_and_load**](01_model_and_load) | data-modeling → cypher | design Airports/ROUTEs, generate ingest Cypher, load ~67k OpenFlights routes through the write tool | `build.py` |
| [**02_cypher_query**](02_cypher_query) | cypher | schema introspection, multi-hop reads, vertex/edge parsing, pagination, read-only, timeout/token-limit | `ask.py` |
| [**03_cypher_operate**](03_cypher_operate) | cypher | write + stats, read-only mode, namespacing, real **stdio + Streamable HTTP** transports + CORS | `ask.py` |
| [**04_memory**](04_memory) | memory | entities/relations/observations, search, find-by-name, `read_graph` limit/`truncated`, deletes | `build.py` → `ask.py` |
| [**05_cypher_scale**](05_cypher_scale) | cypher | read-only schema + aggregations + pagination on a pre-existing ~150k-node graph | `ask.py` |

Each folder has a README and a **pre-executed notebook** (real outputs, no setup needed).
Track B — [`claude_desktop/`](./claude_desktop) — wires all three servers into Claude
Desktop with guided prompts.

## How the demos drive the servers

```python
from _common import clients
async with clients.cypher_client("mcp_flights", "flights") as cy:        # in-memory client
    schema = clients.data(await cy.call_tool("get_agensgraph_schema", {}))
    page = await cy.call_tool("read_agensgraph_cypher", {"query": "...", "limit": 1000, "offset": 0})
# stdio / HTTP variants: clients.stdio_client(...), clients.http_client(url)
```

`_common/` wires the FastMCP client to each server (in-memory + stdio + HTTP), the
AgensGraph connection, and the OpenFlights loader.
