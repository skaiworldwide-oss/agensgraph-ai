"""Shared core for the AgensGraph MCP servers.

Five modules, imported by name:

* ``config``     -- what the CLI and the environment say
* ``connection`` -- the pool's lifetime, and running one statement on it
* ``results``    -- a result as JSON a model can read, bounded by what it can hold
* ``safety``     -- quoting an identifier in a statement written by a model
* ``transport``  -- stdio, Streamable HTTP and SSE

The package root re-exports none of it. Each server imports the module it needs, which is what
keeps the DB-less data-modeling server from importing psycopg to read a namespace out of the
environment.
"""
