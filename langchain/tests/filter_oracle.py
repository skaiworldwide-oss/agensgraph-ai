"""A metadata filter evaluated in Python, with the meaning ``vectorstores.filters.translate``
gives it in SQL: the oracle the translation is tested against."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from langchain_agensgraph.vectorstores.filters import COMPARISONS


def matches(where: Mapping[str, Any] | None, metadata: Mapping[str, Any]) -> bool:
    """Whether a chunk's metadata passes a filter."""
    if not where:
        return True
    for key, value in where.items():
        if key == "$and":
            if not all(matches(each, metadata) for each in value):
                return False
        elif key == "$or":
            if not any(matches(each, metadata) for each in value):
                return False
        elif key == "$not":
            if matches(value, metadata):
                return False
        elif not isinstance(value, Mapping):
            if not _equal(metadata, key, value):
                return False
        elif not all(_holds(metadata, key, op, arg) for op, arg in value.items()):
            return False
    return True


def _holds(metadata: Mapping[str, Any], key: str, op: str, arg: Any) -> bool:
    present = key in metadata
    stored = metadata.get(key)
    if op == "$eq":
        return _equal(metadata, key, arg)
    if op == "$ne":
        return not _equal(metadata, key, arg)
    if op == "$in":
        return any(_equal(metadata, key, each) for each in arg)
    if op == "$nin":
        return not any(_equal(metadata, key, each) for each in arg)
    if op == "$exists":
        return present is arg
    if op in COMPARISONS or op == "$between":
        bounds = arg if op == "$between" else [arg]
        if not present or _kind(stored) != _kind(bounds[0]):
            return False
        if op == "$between":
            return bool(arg[0] <= stored <= arg[1])
        return bool(
            {
                "$gt": stored > arg,
                "$gte": stored >= arg,
                "$lt": stored < arg,
                "$lte": stored <= arg,
            }[op]
        )
    if op in ("$like", "$ilike"):
        if not isinstance(stored, str):
            return False
        flags = re.IGNORECASE if op == "$ilike" else 0
        return re.fullmatch(_like_regex(arg), stored, flags | re.DOTALL) is not None
    raise ValueError(f"unknown filter operator {op!r}")


def _equal(metadata: Mapping[str, Any], key: str, value: Any) -> bool:
    if key not in metadata:
        return False
    stored = metadata[key]
    if isinstance(stored, bool) or isinstance(value, bool):
        return isinstance(stored, bool) and isinstance(value, bool) and stored is value
    return _kind(stored) == _kind(value) and stored == value


def _kind(value: object) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int | float):
        return "number"
    if isinstance(value, str):
        return "string"
    return type(value).__name__


def _like_regex(pattern: str) -> str:
    """A SQL ``LIKE`` pattern as a regular expression: ``%`` any run, ``_`` one character."""
    out = []
    escaped = False
    for char in pattern:
        if escaped:
            out.append(re.escape(char))
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == "%":
            out.append(".*")
        elif char == "_":
            out.append(".")
        else:
            out.append(re.escape(char))
    return "".join(out)
