"""Metadata filters, as LangChain writes them, turned into one SQL predicate.

A chunk's metadata is stored in its vertex's property map, so a filter reads ``properties``::

    {"year": {"$gte": 2020}, "lang": {"$in": ["en", "ko"]}}

becomes, for the label aliased ``l``::

    (jsonb_typeof(l.properties->'year') = 'number' AND (l.properties->'year') >= %s::jsonb)
    AND l.properties @> ANY(%s::jsonb[])

Equality and membership are containment, which the label's GIN index serves whatever the key;
comparisons read the key, which a declared filter key's btree serves. A comparison is bracketed by
the value's JSON type, so ``{"$gt": 5}`` never matches a string and ``{"$gt": "b"}`` never matches a
number, where jsonb's own ordering would compare across types. ``$ne`` and ``$nin`` match a chunk
that lacks the key, as "not equal" reads.

Keys are written into the statement as literals, after checking that one is a string the server
will read as given, because an expression index is matched against the statement's text and a
parameter in its place would never match it. Every value is a parameter.

Operators: ``$eq``, ``$ne``, ``$gt``, ``$gte``, ``$lt``, ``$lte``, ``$in``, ``$nin``, ``$between``,
``$like``, ``$ilike``, ``$exists``, and the combinators ``$and``, ``$or``, ``$not``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from agensgraph import Jsonb

from .._core.names import literal

MAX_KEY_BYTES = 255

COMPARISONS = {"$gt": ">", "$gte": ">=", "$lt": "<", "$lte": "<="}
FIELD_OPERATORS = frozenset(
    {"$eq", "$ne", "$in", "$nin", "$between", "$like", "$ilike", "$exists", *COMPARISONS}
)
COMBINATORS = frozenset({"$and", "$or", "$not"})

Scalar = str | int | float | bool


def translate(where: Mapping[str, Any] | None, alias: str = "l") -> tuple[str, list[Any]]:
    """The predicate for a filter, and its parameters; ``TRUE`` for no filter."""
    if not where:
        return "TRUE", []
    params: list[Any] = []
    return _clause(where, f"{alias}.properties", params), params


def names_a_field(where: Mapping[str, Any] | None) -> bool:
    """Whether a filter tests any metadata field; one that does not, ``{}`` or
    ``{"$and": [{}]}``, holds for every chunk."""
    if not where:
        return False
    for key, value in where.items():
        if key not in COMBINATORS:
            return True
        parts = [value] if key == "$not" else value
        if any(isinstance(part, Mapping) and names_a_field(part) for part in parts):
            return True
    return False


def _clause(where: Mapping[str, Any], properties: str, params: list[Any]) -> str:
    if not isinstance(where, Mapping):
        raise ValueError(f"a filter must be a mapping, not {where!r}")
    if not where:
        return "TRUE"
    parts = []
    for key, value in where.items():
        if key in COMBINATORS:
            parts.append(_combined(key, value, properties, params))
        elif isinstance(key, str) and key.startswith("$"):
            raise ValueError(f"unknown filter operator {key!r}")
        else:
            parts.append(_field(_key(key), value, properties, params))
    return parts[0] if len(parts) == 1 else "(" + " AND ".join(parts) + ")"


def _combined(operator: str, value: Any, properties: str, params: list[Any]) -> str:
    if operator == "$not":
        return f"NOT ({_clause(value, properties, params)})"
    if not isinstance(value, Sequence) or isinstance(value, str) or not value:
        raise ValueError(f"{operator} takes a non-empty list of filters, not {value!r}")
    joiner = " AND " if operator == "$and" else " OR "
    return "(" + joiner.join(_clause(each, properties, params) for each in value) + ")"


def _field(key: str, value: Any, properties: str, params: list[Any]) -> str:
    if not isinstance(value, Mapping):
        return _field(key, {"$eq": value}, properties, params)
    if not value:
        raise ValueError(f"the filter on {key!r} names no operator")
    parts = [_operator(key, op, arg, properties, params) for op, arg in value.items()]
    return parts[0] if len(parts) == 1 else "(" + " AND ".join(parts) + ")"


def _operator(key: str, op: str, arg: Any, properties: str, params: list[Any]) -> str:
    field = f"({properties}->{literal(key)})"
    if op == "$eq":
        params.append(Jsonb({key: _scalar(key, op, arg)}))
        return f"{properties} @> %s::jsonb"
    if op == "$ne":
        params.append(Jsonb({key: _scalar(key, op, arg)}))
        return f"NOT ({properties} @> %s::jsonb)"
    if op in ("$in", "$nin"):
        values = _list(key, op, arg)
        if not values:
            return "FALSE" if op == "$in" else "TRUE"
        params.append([Jsonb({key: value}) for value in values])
        found = f"{properties} @> ANY(%s::jsonb[])"
        return found if op == "$in" else f"NOT ({found})"
    if op in COMPARISONS:
        kind = _ordered(key, op, arg)
        params.append(Jsonb(arg))
        return f"(jsonb_typeof({field}) = '{kind}' AND {field} {COMPARISONS[op]} %s::jsonb)"
    if op == "$between":
        if not isinstance(arg, Sequence) or isinstance(arg, str) or len(arg) != 2:
            raise ValueError(f"$between on {key!r} takes [low, high], not {arg!r}")
        low, high = arg
        kind = _ordered(key, op, low)
        if _ordered(key, op, high) != kind:
            raise ValueError(f"$between on {key!r} needs two values of one type, not {arg!r}")
        params += [Jsonb(low), Jsonb(high)]
        return (
            f"(jsonb_typeof({field}) = '{kind}' AND {field} >= %s::jsonb AND {field} <= %s::jsonb)"
        )
    if op in ("$like", "$ilike"):
        if not isinstance(arg, str):
            raise ValueError(f"{op} on {key!r} takes a pattern string, not {arg!r}")
        params.append(arg)
        word = "LIKE" if op == "$like" else "ILIKE"
        return f"(jsonb_typeof({field}) = 'string' AND ({properties}->>{literal(key)}) {word} %s)"
    if op == "$exists":
        if not isinstance(arg, bool):
            raise ValueError(f"$exists on {key!r} takes true or false, not {arg!r}")
        found = f"({properties} ? {literal(key)})"
        return found if arg else f"NOT {found}"
    raise ValueError(f"unknown filter operator {op!r} on {key!r}")


def _key(key: object) -> str:
    if not isinstance(key, str) or not key:
        raise ValueError(f"a filter key must be a non-empty string, not {key!r}")
    if "\x00" in key or len(key.encode()) > MAX_KEY_BYTES:
        raise ValueError(f"the filter key {key!r} holds a null byte or is too long")
    return key


def _scalar(key: str, op: str, value: Any) -> Scalar:
    if isinstance(value, bool | int | float | str) and not _nan(value):
        return value
    if value is None:
        raise ValueError(
            f"{op} on {key!r} compares with None, and a None is never stored: use "
            f'{{"$exists": false}} for a chunk without {key!r}'
        )
    raise ValueError(f"{op} on {key!r} takes a string, number or boolean, not {value!r}")


def _list(key: str, op: str, values: Any) -> list[Scalar]:
    if not isinstance(values, Sequence) or isinstance(values, str):
        raise ValueError(f"{op} on {key!r} takes a list, not {values!r}")
    return [_scalar(key, op, value) for value in values]


def _ordered(key: str, op: str, value: Any) -> str:
    """The JSON type a comparison brackets by: a number or a string."""
    if isinstance(value, bool) or not isinstance(value, int | float | str) or _nan(value):
        raise ValueError(f"{op} on {key!r} compares numbers or strings, not {value!r}")
    return "string" if isinstance(value, str) else "number"


def _nan(value: object) -> bool:
    return isinstance(value, float) and value != value


__all__ = ["names_a_field", "translate"]
