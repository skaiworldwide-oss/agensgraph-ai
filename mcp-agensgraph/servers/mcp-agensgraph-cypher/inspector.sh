#!/bin/bash
# Drive mcp-agensgraph-cypher from the MCP Inspector against a local database.
#
# Run it from this directory. The flags are the server's own: `--database` and `--graphname`,
# not `--db-name` and `--graph-name`, which it answers with "unrecognized arguments" and exits
# before the Inspector has anything to talk to.
#
# No `--password`: one on the command line is readable by every process on the machine through
# `ps`. Set PGPASSWORD, use .pgpass, or connect with an authentication method that needs none.
npx @modelcontextprotocol/inspector \
    uv run mcp-agensgraph-cypher \
    --db-url "${AGENSGRAPH_URL:-postgresql://localhost:5432}" \
    --database "${AGENSGRAPH_DATABASE:-test}" \
    --username "${AGENSGRAPH_USERNAME:-$USER}" \
    --graphname "${AGENSGRAPH_GRAPHNAME:-test}"
