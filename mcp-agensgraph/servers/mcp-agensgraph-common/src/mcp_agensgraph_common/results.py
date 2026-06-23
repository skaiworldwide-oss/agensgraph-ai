"""Parse AgensGraph query results into JSON-friendly Python values.

AgensGraph's graph values come back (via psycopg) as ``vertex`` / ``edge`` typed
strings:

    vertex: ``label[gid]{...json props...}``
    edge:   ``label[gid][start_gid, end_gid]{...json props...}``

plus ordinary scalars. These helpers turn a psycopg ``namedtuple`` row into a plain
dict, sanitize oversized lists (e.g. embeddings) that waste LLM context, and truncate
a serialized response to a token budget.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, NamedTuple, Optional, Pattern

logger = logging.getLogger("mcp_agensgraph_common")

VERTEX_REGEX: Pattern = re.compile(r"(\w+)\[(\d+\.\d+)\](\{.*\})")
EDGE_REGEX: Pattern = re.compile(
    r"(\w+)\[(\d+\.\d+)\]\[(\d+\.\d+),\s*(\d+\.\d+)\](\{.*\})"
)


def _loads(properties: str) -> Any:
    """json.loads that degrades to the raw string instead of crashing the query."""
    try:
        return json.loads(properties)
    except (json.JSONDecodeError, TypeError):
        logger.debug("Could not JSON-decode vertex/edge properties: %.80s", properties)
        return properties


def record_to_dict(record: NamedTuple) -> dict[str, Any]:
    """Convert an AgensGraph result row (namedtuple) to a dict.

    Vertices become their property maps; edges become a
    ``(start_props, type, end_props)`` triple, resolving endpoints against the
    vertices seen in the same row.
    """
    result: dict[str, Any] = {}
    vertices: dict[str, Any] = {}

    for field_name in record._fields:
        value = getattr(record, field_name)
        if isinstance(value, str):
            vertex_match = VERTEX_REGEX.match(value)
            if vertex_match:
                _, vertex_id, properties = vertex_match.groups()
                vertices[str(vertex_id)] = _loads(properties)

    for field_name in record._fields:
        value = getattr(record, field_name)
        if isinstance(value, str):
            vertex_match = VERTEX_REGEX.match(value)
            edge_match = EDGE_REGEX.match(value)
            if vertex_match:
                result[field_name] = _loads(vertex_match.group(3))
            elif edge_match:
                label, _eid, start_id, end_id, _props = edge_match.groups()
                result[field_name] = (
                    vertices.get(start_id, {}),
                    label,
                    vertices.get(end_id, {}),
                )
            else:
                result[field_name] = value
        else:
            result[field_name] = value
    return result


def value_sanitize(value: Any, list_limit: int = 128) -> Any:
    """Drop oversized lists (e.g. embeddings) that bloat LLM context.

    Adapted from neo4j-graphrag-python's schema sanitizer.
    """
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, val in value.items():
            if isinstance(val, dict):
                sanitized = value_sanitize(val, list_limit)
                if sanitized is not None:
                    out[key] = sanitized
            elif isinstance(val, list):
                if len(val) < list_limit:
                    sanitized = value_sanitize(val, list_limit)
                    if sanitized is not None:
                        out[key] = sanitized
                # oversized list: drop the key
            else:
                out[key] = val
        return out
    if isinstance(value, list):
        if len(value) < list_limit:
            return [
                value_sanitize(item, list_limit)
                for item in value
                if value_sanitize(item, list_limit) is not None
            ]
        return None
    return value


def truncate_to_tokens(text: str, token_limit: int, model: str = "gpt-4o") -> str:
    """Truncate ``text`` to at most ``token_limit`` tokens for the given model.

    Falls back to a generic encoding for unknown models so an unusual model name
    never crashes a response.
    """
    import tiktoken

    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    tokens = encoding.encode(text)
    if len(tokens) <= token_limit:
        return text
    return encoding.decode(tokens[:token_limit])
