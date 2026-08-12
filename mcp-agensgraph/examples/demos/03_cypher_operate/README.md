# 03 · Cypher operate — writes, safety modes, and transports

The operating side of the **cypher** server — what you configure when you actually
deploy it:

- **`write_agensgraph_cypher`** — mutate the graph (here: MERGE a route) and get back
  write stats (`insertedvertices`/`insertededges`/…). The demo adds a `DEMO` route and
  deletes it again so the graph stays as loaded.
- **Read-only mode** — start the server `--read-only` and the write tool is not registered,
  so a client cannot call it by name either. The other six stay: `get_agensgraph_schema`,
  `read_agensgraph_cypher`, `explain_agensgraph_cypher`, `recommend_property_indexes`,
  `agensgraph_health` and `top_cypher_queries`. Defense in depth: their statements run in a
  `READ ONLY` transaction, which is what actually refuses a write.
- **Namespacing** — `--namespace ops` prefixes every tool (`ops-read_agensgraph_cypher`),
  so several servers (e.g. one per database) coexist in one client.
- **Transports** — the same tools driven over the **real server process**: **stdio**
  (what Claude Desktop spawns) and **Streamable HTTP** (what remote/containerized
  deployments use), the latter with CORS + DNS-rebinding (TrustedHost) middleware.

📓 **Guided tour:** [`cypher_operate.ipynb`](./cypher_operate.ipynb).

## Run

```bash
# from mcp-agensgraph/examples/demos  (run 01_model_and_load/build.py first)
.venv/bin/python 03_cypher_operate/ask.py
```

## What it shows

```python
# in-memory: read-only + namespacing are construction options
async with clients.cypher_client("mcp_flights", "flights", read_only=True) as cy: ...
async with clients.cypher_client("mcp_flights", "flights", namespace="ops") as cy: ...

# real process over stdio / HTTP
async with clients.stdio_client("mcp-agensgraph-cypher", ["--transport", "stdio"], env) as cy: ...
async with clients.http_client(f"http://127.0.0.1:{port}/mcp/") as cy: ...   # port from the kernel
```

The HTTP server is launched with `--allow-origins` / `--allowed-hosts`, so CORS and
TrustedHost middleware are active. (SSE is also supported but deprecated — prefer HTTP.)
