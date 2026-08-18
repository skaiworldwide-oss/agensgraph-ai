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
from typing import Any, Sequence

from agensgraph import Result, to_builtins

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


def column_names(keys: Sequence[str]) -> list[str]:
    """The column names, each one usable as a key of the same map.

    A result's columns are a list rather than a set, and nothing stops two of them sharing a
    name: ``RETURN n.a AS x, n.b AS x`` names both ``x``. Keeping the first and dropping the
    second would answer a question with half of it, so a repeat is named by the position it
    came from and both values survive.

    Nothing else is renamed. A column called ``class`` or ``_x`` is what the caller asked for
    and is a perfectly good key -- building rows as namedtuples was what could not hold those,
    with ``ValueError: Type names and field names cannot be a keyword`` for the first and a
    silent rename to ``f_x`` for the second.
    """
    taken: set[str] = set()
    names: list[str] = []
    for position, key in enumerate(keys, 1):
        name = key
        while name in taken:
            name = f"{name} (column {position})"
        taken.add(name)
        names.append(name)
    return names


def mapping_rows(cursor: Any) -> Any:
    """A psycopg row factory building the same maps as :func:`rows_of`.

    For a stream, which yields rows rather than a result and so has no column names to hand
    afterwards. The names are read once, when the factory is built for a result.
    """
    names = column_names([column.name for column in cursor.description or ()])

    def build(values: Sequence[Any]) -> dict[str, Any]:
        return {name: as_builtins(value) for name, value in zip(names, values)}

    return build


def naming_rows(into: list[str]) -> Any:
    """A psycopg row factory that records the column names and hands the values back untouched.

    For a walk, which reads every row and keeps a few. Building a map for each row on the way
    past converts the ones about to be dropped: measured, walking 2,000 rows to keep 10 converted
    4,000 values instead of 20, and a walk of 50,000 whole vertices cost 30.4 seconds against
    21.0 for keeping the rows as they arrived. The names are what a stream otherwise has no way
    to recover afterwards, so they are taken once here and the conversion happens on the rows
    that are kept.
    """

    def build(cursor: Any) -> Any:
        into[:] = column_names([column.name for column in cursor.description or ()])

        def identity(values: Sequence[Any]) -> Sequence[Any]:
            return values

        return identity

    return build


def named_row(names: Sequence[str], values: Sequence[Any]) -> dict[str, Any]:
    """One row as a map, for a caller that kept the values and the names separately."""
    return {name: as_builtins(value) for name, value in zip(names, values)}


def rows_of(result: Result) -> list[dict[str, Any]]:
    """A result's rows as maps of column name to a JSON-shaped value.

    The values arrive decoded: the driver reads the wire form, so a vertex is a vertex and an
    edge carries its own id, its properties and the identity at each of its ends -- which
    matching the text the server printed could not do, since an edge read on its own reported
    an empty map at each end.
    """
    names = column_names(result.keys)
    return [
        {name: as_builtins(value) for name, value in zip(names, record)}
        for record in result.records
    ]


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

BYTES_PER_TOKEN = 2
"""What a token is taken to be worth when there is no tokenizer to ask.

Measured against a graph of 138,619 papers, over four shapes of result: 4.67 bytes per token
for titles and abstracts, 2.14 for whole vertices, 2.00 for a label and an id, 1.75 for ids
alone -- 1.50 to 5.96 across every row. Two is at the dense end of that, so the estimate is
generous with text and can be about a third low for a result of nothing but ids.
"""


def token_encoding(model: str = "gpt-4o") -> Any:
    """The tokenizer for a model, loaded once per process, or ``None`` if there is not one.

    tiktoken is an extra rather than a dependency, and what makes that worth doing is what
    loading it does: the first call fetches the model's BPE file over the network. Measured
    with an empty cache directory, 5,445 ms -- and where there is no route out, the shipped
    default could not serve a single read at all. Every way that can fail is caught here, the
    package being absent and the fetch failing alike, and the answer is then an estimate rather
    than a failed response.

    An unknown model name falls back to a generic encoding rather than failing a response.
    """
    if model in _ENCODINGS:
        return _ENCODINGS[model]
    encoding = None
    try:
        import tiktoken

        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
    except ImportError:
        logger.info(
            "tiktoken is not installed, so a response is bounded by an estimate of %d bytes "
            "per token. Install mcp-agensgraph-common[tokens] to count them exactly.",
            BYTES_PER_TOKEN,
        )
    except Exception as exc:  # pragma: no cover - needs a broken or unreachable cache
        logger.warning(
            "tiktoken could not load an encoding for %r (%s), so a response is bounded by an "
            "estimate of %d bytes per token. Loading one fetches a file over the network.",
            model,
            exc,
            BYTES_PER_TOKEN,
        )
    _ENCODINGS[model] = encoding
    return encoding


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """How many tokens a string costs the model it is going to, counted or estimated."""
    encoding = token_encoding(model)
    if encoding is None:
        return -(-len(text.encode()) // BYTES_PER_TOKEN)
    return len(encoding.encode(text))


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

    With no tokenizer installed the measure is the row's size in bytes over
    :data:`BYTES_PER_TOKEN`. A budget is still a budget -- what changes is that it is
    approximate, not that it is gone.
    """
    encoding = token_encoding(model)
    if encoding is None:
        def measure(text: str) -> int:
            return -(-len(text.encode()) // BYTES_PER_TOKEN)
    else:
        def measure(text: str) -> int:
            return len(encoding.encode(text))

    budget = max(0, token_limit - reserve)
    used = 0
    for index, row in enumerate(rows):
        used += measure(json.dumps(row, default=str))
        if used > budget:
            return rows[:index], len(rows) - index
    return rows, 0


__all__ = [
    "named_row",
    "naming_rows",
    "BYTES_PER_TOKEN",
    "OMITTED",
    "as_builtins",
    "column_names",
    "count_tokens",
    "fit_rows",
    "mapping_rows",
    "rows_of",
    "token_encoding",
    "value_sanitize",
]
