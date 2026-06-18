'''
Copyright (c) 2025, SKAI Worldwide Co., Ltd.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

"""
Shared translation of LlamaIndex ``MetadataFilters`` into parameterized
AgensGraph Cypher, used by both the property graph store and the vector store.

Every value is bound as a query parameter (psycopg ``Jsonb``); nothing is
interpolated into the query text, so the translation is injection-safe. All
14 ``FilterOperator`` values are supported, along with nested filter groups
and the ``AND`` / ``OR`` / ``NOT`` conditions.
"""

import re
from typing import Any, Dict, List, Tuple, Union

from psycopg import sql
from psycopg.types.json import Jsonb

from llama_index.core.vector_stores.types import (
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)

# Binary operators that translate to ``<prop> <op> <param>``.
_BINARY_OPERATORS: Dict[FilterOperator, str] = {
    FilterOperator.EQ: "=",
    FilterOperator.NE: "<>",
    FilterOperator.GT: ">",
    FilterOperator.GTE: ">=",
    FilterOperator.LT: "<",
    FilterOperator.LTE: "<=",
    FilterOperator.IN: "IN",
    # CONTAINS / TEXT_MATCH are case-sensitive substring matches on strings.
    FilterOperator.CONTAINS: "CONTAINS",
    FilterOperator.TEXT_MATCH: "CONTAINS",
}


class _ParamAllocator:
    """Hands out unique parameter names so multiple filters never collide."""

    def __init__(self, prefix: str = "mf") -> None:
        self._prefix = prefix
        self._n = 0

    def alloc(self) -> str:
        name = f"{self._prefix}_{self._n}"
        self._n += 1
        return name


def _prop(alias: str, key: str) -> sql.Composed:
    """Render a safe ``alias."key"`` property reference."""
    if not alias.isidentifier():
        raise ValueError(f"Invalid node alias: {alias!r}")
    return sql.SQL("{}.{}").format(sql.SQL(alias), sql.Identifier(key))


def _single_filter(
    f: MetadataFilter, alias: str, alloc: _ParamAllocator, params: Dict[str, Any]
) -> sql.Composed:
    prop = _prop(alias, f.key)
    op = f.operator

    if op == FilterOperator.IS_EMPTY:
        # A field counts as empty when it is missing/null, an empty list, or an
        # empty string. Two params cover the empty-list / empty-string cases.
        p_list = alloc.alloc()
        p_str = alloc.alloc()
        params[p_list] = Jsonb([])
        params[p_str] = Jsonb("")
        return sql.SQL("({prop} IS NULL OR {prop} = %({pl})s OR {prop} = %({ps})s)").format(
            prop=prop,
            pl=sql.SQL(p_list),
            ps=sql.SQL(p_str),
        )

    if op == FilterOperator.NIN:
        name = alloc.alloc()
        params[name] = Jsonb(f.value)
        # Cypher has no ``NOT IN``; negate the membership test instead.
        return sql.SQL("NOT ({prop} IN %({p})s)").format(prop=prop, p=sql.SQL(name))

    if op in (FilterOperator.ANY, FilterOperator.ALL):
        name = alloc.alloc()
        params[name] = Jsonb(f.value)
        fn = "any" if op == FilterOperator.ANY else "all"
        # Match when any/all of the filter values are present in the list prop.
        return sql.SQL("{fn}(x IN %({p})s WHERE x IN {prop})").format(
            fn=sql.SQL(fn), p=sql.SQL(name), prop=prop
        )

    if op == FilterOperator.TEXT_MATCH_INSENSITIVE:
        name = alloc.alloc()
        # Case-insensitive substring via a regex with the (?i) flag.
        params[name] = Jsonb("(?i).*" + re.escape(str(f.value)) + ".*")
        return sql.SQL("{prop} =~ %({p})s").format(prop=prop, p=sql.SQL(name))

    binary = _BINARY_OPERATORS.get(op)
    if binary is None:
        raise ValueError(f"Unsupported metadata filter operator: {op}")
    name = alloc.alloc()
    params[name] = Jsonb(f.value)
    return sql.SQL("{prop} {op} %({p})s").format(
        prop=prop, op=sql.SQL(binary), p=sql.SQL(name)
    )


def _build(
    filters: MetadataFilters,
    alias: str,
    alloc: _ParamAllocator,
    params: Dict[str, Any],
) -> sql.Composed:
    parts: List[sql.Composed] = []
    for f in filters.filters:
        if isinstance(f, MetadataFilters):
            parts.append(
                sql.SQL("(") + _build(f, alias, alloc, params) + sql.SQL(")")
            )
        else:
            parts.append(_single_filter(f, alias, alloc, params))

    if not parts:
        return sql.SQL("true")

    condition = filters.condition or FilterCondition.AND
    if condition == FilterCondition.NOT:
        joined = sql.SQL(" AND ").join(parts)
        return sql.SQL("NOT (") + joined + sql.SQL(")")

    joiner = sql.SQL(" OR ") if condition == FilterCondition.OR else sql.SQL(" AND ")
    return joiner.join(parts)


def metadata_filters_to_cypher(
    filters: MetadataFilters,
    alias: str = "n",
    param_prefix: str = "mf",
) -> Tuple[sql.Composed, Dict[str, Any]]:
    """
    Translate ``MetadataFilters`` into a parameterized Cypher WHERE snippet.

    Args:
        filters: the LlamaIndex metadata filters (may be nested).
        alias: the node variable the properties belong to (e.g. ``"n"``).
        param_prefix: prefix for the generated parameter names.

    Returns:
        A ``(snippet, params)`` tuple. ``snippet`` is a ``psycopg.sql.Composed``
        boolean expression and ``params`` maps the generated parameter names to
        their ``Jsonb`` values. ``snippet`` is ``true`` when there are no
        filters.
    """
    alloc = _ParamAllocator(param_prefix)
    params: Dict[str, Any] = {}
    snippet = _build(filters, alias, alloc, params)
    return snippet, params


__all__ = ["metadata_filters_to_cypher"]
