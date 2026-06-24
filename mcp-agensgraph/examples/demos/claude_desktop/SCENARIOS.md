# Claude Desktop — driving the AgensGraph MCP servers with a real LLM

The demos in `01`–`05` drive the servers with a deterministic FastMCP client (no LLM).
This track is the other half: **Claude Desktop is MCP's native client**, so it's how a
real LLM uses these tools. You run it; the servers do the work.

> These scenarios are run by the user (they need the Claude Desktop app). The
> deterministic demos (`01`–`05`) prove the servers work; this proves they're useful to
> an agent. We'll capture real transcripts here together after a run.

## Setup

1. Build the flights graph + seed the memory (once):
   ```bash
   cd mcp-agensgraph/examples/demos
   .venv/bin/python 01_model_and_load/build.py
   .venv/bin/python 04_memory/build.py
   ```
2. Copy [`claude_desktop_config.json`](./claude_desktop_config.json) into Claude Desktop's
   config (macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`;
   Linux/Windows: the equivalent Claude config dir). Adjust the absolute paths if this
   repo lives elsewhere. Restart Claude Desktop — the three servers' tools should appear.
3. (Optional, no Claude needed) Validate any server with the **MCP Inspector**:
   ```bash
   npx @modelcontextprotocol/inspector \
     .venv/bin/mcp-agensgraph-cypher --transport stdio
   ```
   The Inspector lists tools, runs them, and shows raw results — a quick, LLM-free check.

## Scenarios

### A · Cypher — ask the flights graph in natural language
The agent uses `get_agensgraph_schema` to learn the model, then `read_agensgraph_cypher`
(paginated, read-only) to answer:
- *"What's in this graph? Which airports have the most outgoing routes?"*
- *"How can I get from a small island airport to JFK with one stopover?"*
- *"Which airlines fly out of Frankfurt, and where to?"*

### B · Data modeling — design a schema from scratch
Use the `create_new_data_model` **prompt** (Claude Desktop surfaces MCP prompts as
slash-commands / starters):
- *"Design a graph data model for a movie-recommendation app (users, movies, ratings,
  actors). Validate it, show me the Mermaid diagram, and give me the ingest Cypher."*

The agent calls `validate_data_model`, `get_mermaid_config_str` (Claude renders the
diagram), and `get_*_cypher_ingest_query` — then you can hand that Cypher to the cypher
server's `write` tool in the same conversation.

### C · Memory — remember across the conversation
With the seeded `mcp_memory` graph (or starting empty):
- *"Remember that I'm planning a trip to Tokyo in March and I prefer aisle seats."*
  → the agent calls `create_entities` / `add_observations`.
- Later: *"What do you know about my travel preferences?"*
  → the agent calls `search_memories` / `read_graph` and answers from the graph.

## Capturing results
After running a scenario, paste the notable turns (and any `get_mermaid_config_str`
diagram) here, so this file becomes a real transcript of an LLM using the servers.
