"""Server settings more than one component asks of its statements."""

from __future__ import annotations

from collections.abc import Mapping

# An UNWIND-keyed MATCH probes the key's index once per row only when the planner may not choose a
# hash or merge join, which it otherwise does by estimate and which reads the whole label.
NESTED_LOOPS: Mapping[str, object] = {"enable_hashjoin": "off", "enable_mergejoin": "off"}

EF_SEARCH_MAX = 1000
"""The largest ``hnsw.ef_search`` pgvector takes; an iterative scan reads on past it."""
