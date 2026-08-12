"""Turn AgensGraph query results into JSON-friendly Python values.

The driver decodes a vertex, an edge and a path into types of its own, so a row becomes a
dict by asking the driver for the JSON shape of each value rather than by matching the text
the server printed. An edge carries both of its endpoint identities, so an edge read on its
own reports them: ``MATCH ()-[r]->() RETURN r`` names what is at each end instead of two
empty maps.

A list too long to be worth a model's context is replaced by a marker saying how many items
it held, so a suppressed value cannot be read as an absent one.

A response is bounded by whole rows. Rows are measured one at a time and only the ones that
fit are kept, so what reaches the model is JSON it can parse.
"""

from __future__ import annotations

import json
import logging
from typing import Any, NamedTuple

from agensgraph import to_builtins

logger = logging.getLogger("mcp_agensgraph_common")

# Values that are already JSON, checked by exact type so the common case costs one lookup.
_PLAIN = (str, int, float, bool, type(None))

# What a list dropped for its size leaves behind. A model reading a result has to be able to
# tell a value that was suppressed from one that was not there.
OMITTED = "<omitted: list of {count} items>"


def as_builtins(value: Any) -> Any:
    """A query value as dicts, lists and scalars, graph values included.

    Containers are walked, because a vertex can arrive inside a list or a map -- ``collect(n)``
    and ``{a: n}`` both do it -- and the driver's own conversion takes one value at a time.
    """
    kind = type(value)
    if kind in _PLAIN:
        return value
    if kind is dict:
        return {key: as_builtins(item) for key, item in value.items()}
    if kind is list or kind is tuple:
        return [as_builtins(item) for item in value]
    try:
        return to_builtins(value)
    except TypeError:
        # Not a graph value: a date, a decimal, anything else the server sends. Left as it is
        # for the serializer to render.
        return value


def record_to_dict(record: NamedTuple) -> dict[str, Any]:
    """Convert an AgensGraph result row (namedtuple) to a dict."""
    return {name: as_builtins(getattr(record, name)) for name in record._fields}


def value_sanitize(value: Any, list_limit: int = 128) -> Any:
    """Replace lists too long for a model's context with a marker naming their length.

    An embedding is the case this exists for: a thousand-odd floats cost more context than the
    whole rest of the answer and say nothing a model can act on. What it must not do is delete
    the key, which reads exactly like the property not being there -- a model asking for
    ``n.embedding`` and receiving ``{}`` cannot tell that it asked for something and got it.
    """
    if isinstance(value, dict):
        return {
            key: OMITTED.format(count=len(item))
            if isinstance(item, list) and len(item) >= list_limit
            else value_sanitize(item, list_limit)
            for key, item in value.items()
        }
    if isinstance(value, list):
        if len(value) >= list_limit:
            return OMITTED.format(count=len(value))
        return [value_sanitize(item, list_limit) for item in value]
    return value


_ENCODINGS: dict[str, Any] = {}


def token_encoding(model: str = "gpt-4o") -> Any:
    """The tokenizer for a model, loaded once per process.

    Loading one is four tenths of a second, which is not a thing to pay per tool call. An
    unknown model name falls back to a generic encoding rather than failing a response.
    """
    if model not in _ENCODINGS:
        import tiktoken

        try:
            _ENCODINGS[model] = tiktoken.encoding_for_model(model)
        except KeyError:
            _ENCODINGS[model] = tiktoken.get_encoding("cl100k_base")
    return _ENCODINGS[model]


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """How many tokens a string costs the model it is going to."""
    return len(token_encoding(model).encode(text))


def fit_rows(
    rows: list[Any], token_limit: int, *, reserve: int = 0, model: str = "gpt-4o"
) -> tuple[list[Any], int]:
    """The rows that fit a token budget, and how many were left out.

    Whole rows, measured before anything is serialized as one document. Cutting the serialized
    document instead ends it in the middle of a string or a brace, and a model handed JSON it
    cannot parse has been given nothing -- it cannot even see which rows it received.

    *reserve* is what the envelope around the rows costs, so the budget the rows are measured
    against is what is left after it.

    Measuring stops at the first row that does not fit, so the work is bounded by the budget
    rather than by the size of the result: a thousand rows of abstracts cost a second to
    measure in full and twenty-five milliseconds to measure as far as a ten-thousand-token cap.
    """
    encoding = token_encoding(model)
    budget = max(0, token_limit - reserve)
    used = 0
    for index, row in enumerate(rows):
        used += len(encoding.encode(json.dumps(row, default=str)))
        if used > budget:
            return rows[:index], len(rows) - index
    return rows, 0


__all__ = [
    "OMITTED",
    "as_builtins",
    "count_tokens",
    "fit_rows",
    "record_to_dict",
    "token_encoding",
    "value_sanitize",
]
