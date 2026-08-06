"""Plan inspection, index advice and database health for a graph.

The advice is drawn from the plan a query actually gets, the labels' catalogs, and a set
of shapes that are known not to reach an index. It is never verified by building the index
first: a hypothetical index cannot stand in for a property index here, because the
expression a property index carries — ``properties.'key'::text`` — is not writable in a
plain ``CREATE INDEX``, and an index written with the jsonb operators instead is not the
expression a Cypher filter matches. Recommendations therefore say what they are: reasoned
from the plan, not measured.

Every check reports what it could not do rather than failing the tool, since the optional
extensions it would like are not part of AgensGraph.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from psycopg import sql

# Extensions each check needs. None of these ship with AgensGraph.
OPTIONAL_EXTENSIONS = ("pg_stat_statements", "pgstattuple", "pg_buffercache")

# Plan fragments that mean a predicate never reached an index.
_STARTS_WITH = re.compile(r"string_starts_with\(([^,]+),")
_JSONB_CONTAINS = re.compile(r"@>\s*(?:properties|\()")
_PROPERTY_REF = re.compile(r"properties\.'([^']+)'")


def extensions_query() -> str:
    """Which optional extensions are installed, so a caller knows what degraded.

    The names are inlined rather than bound: a list parameter is wrapped as JSONB for
    Cypher's sake, which is not what a SQL array comparison wants. They are a constant of
    this module, not caller input.
    """
    names = sql.SQL(", ").join(sql.Literal(name) for name in OPTIONAL_EXTENSIONS)
    return sql.SQL(
        "SELECT a.name AS name, "
        "       (SELECT count(*) > 0 FROM pg_extension e WHERE e.extname = a.name) "
        "           AS installed "
        "FROM pg_available_extensions a "
        "WHERE a.name IN ({names})"
    ).format(names=names).as_string()


def label_stats_query() -> str:
    """Row counts per label, from the catalog rather than by counting."""
    return """
        SELECT l.labname AS label,
               l.labkind AS kind,
               c.reltuples::bigint AS approx_rows,
               c.relname AS relname
        FROM pg_catalog.ag_label l
        JOIN pg_catalog.ag_graph g ON g.oid = l.graphid
        JOIN pg_catalog.pg_class c ON c.oid = l.relid
        WHERE g.graphname = %(graph)s
          AND l.labname NOT IN ('ag_vertex', 'ag_edge')
        ORDER BY c.reltuples DESC
    """


def existing_indexes_query() -> str:
    """Property indexes already defined, so advice never repeats one.

    ``ag_get_propindexdef`` renders a property index the way it was written, which is what
    a recommendation has to be compared against.
    """
    return """
        SELECT l.labname AS label,
               i.relname AS index_name,
               pg_catalog.ag_get_propindexdef(i.oid) AS definition,
               s.idx_scan AS scans
        FROM pg_catalog.ag_label l
        JOIN pg_catalog.ag_graph g ON g.oid = l.graphid
        JOIN pg_catalog.pg_index x ON x.indrelid = l.relid
        JOIN pg_catalog.pg_class i ON i.oid = x.indexrelid
        LEFT JOIN pg_catalog.pg_stat_user_indexes s ON s.indexrelid = i.oid
        WHERE g.graphname = %(graph)s
          AND NOT x.indisprimary
        ORDER BY l.labname, i.relname
    """


def health_queries() -> Dict[str, str]:
    """Checks that need only core catalogs."""
    return {
        "cache_hit_ratio": """
            SELECT CASE WHEN sum(heap_blks_hit) + sum(heap_blks_read) = 0 THEN NULL
                        ELSE round(sum(heap_blks_hit)::numeric
                             / (sum(heap_blks_hit) + sum(heap_blks_read)), 4)
                   END AS ratio
            FROM pg_statio_user_tables
        """,
        "unused_indexes": """
            SELECT s.relname AS relation, s.indexrelname AS index_name, s.idx_scan AS scans
            FROM pg_stat_user_indexes s
            JOIN pg_index x ON x.indexrelid = s.indexrelid
            WHERE s.idx_scan = 0 AND NOT x.indisprimary AND NOT x.indisunique
            ORDER BY s.relname, s.indexrelname
            LIMIT 50
        """,
        "vacuum_age": """
            SELECT relname AS relation, n_dead_tup AS dead_tuples,
                   greatest(last_autovacuum, last_vacuum) AS last_vacuum
            FROM pg_stat_user_tables
            WHERE n_dead_tup > 1000
            ORDER BY n_dead_tup DESC
            LIMIT 20
        """,
        "connections": """
            SELECT count(*) AS in_use,
                   current_setting('max_connections')::int AS max_connections
            FROM pg_stat_activity
        """,
        "graphmeta": """
            SELECT current_setting('auto_gather_graphmeta', true) AS auto_gather_graphmeta
        """,
    }


def top_cypher_queries_query() -> str:
    """Slowest Cypher statements in the workload.

    Cypher parse nodes carry jumble support, so pg_stat_statements normalises a Cypher
    statement the same way it does SQL and its literals show as parameters.
    """
    return """
        SELECT query, calls, round(total_exec_time::numeric, 2) AS total_ms,
               round(mean_exec_time::numeric, 2) AS mean_ms, rows
        FROM pg_stat_statements
        WHERE query ~* '^\\s*(MATCH|MERGE|CREATE\\s*\\(|OPTIONAL\\s+MATCH|UNWIND)'
        ORDER BY total_exec_time DESC
        LIMIT %(limit)s
    """


def explain_statement(cypher: str, analyze: bool = False) -> sql.Composed:
    """``EXPLAIN`` over a Cypher statement, as JSON.

    ``EXPLAIN`` accepts Cypher directly. Without ``ANALYZE`` it only plans, so a read is
    not executed; with it the statement runs, which is why the caller has to permit it.
    """
    options = "ANALYZE, BUFFERS, FORMAT JSON" if analyze else "COSTS ON, FORMAT JSON"
    return sql.SQL("EXPLAIN ({opts}) {stmt}").format(
        opts=sql.SQL(options), stmt=sql.SQL(cypher)  # noqa: S608 - caller-supplied Cypher
    )


def _walk(node: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Every node of a plan tree, parents before children."""
    out = [node]
    for child in node.get("Plans", []) or []:
        out.extend(_walk(child))
    return out


def _predicates(node: Dict[str, Any]) -> str:
    return " ".join(
        str(node.get(key, ""))
        for key in ("Filter", "Index Cond", "Recheck Cond", "Hash Cond", "Join Filter")
    )


def analyze_plan(
    plan_json: Any,
    label_rows: Dict[str, int],
    indexed: Dict[str, List[str]],
    min_rows: int = 1000,
) -> List[Dict[str, Any]]:
    """Turn a plan into findings, each naming what to do about it.

    Three shapes are looked for. A sequential scan over a label big enough to matter, with
    a property filter that an index could answer. A ``STARTS WITH``, which no index can
    serve at all. And a jsonb containment test, which is what a bound ``IN`` list becomes.
    """
    root = plan_json[0]["Plan"] if isinstance(plan_json, list) else plan_json["Plan"]
    findings: List[Dict[str, Any]] = []

    for node in _walk(root):
        relation = node.get("Relation Name")
        predicates = _predicates(node)
        properties = _PROPERTY_REF.findall(predicates)

        if node.get("Node Type") == "Seq Scan" and relation:
            rows = label_rows.get(relation, 0)
            if rows >= min_rows and properties:
                already = indexed.get(relation, [])
                missing = [p for p in dict.fromkeys(properties) if p not in already]
                if missing:
                    findings.append(
                        {
                            "kind": "missing_index",
                            "label": relation,
                            "properties": missing,
                            "approx_rows": rows,
                            "detail": (
                                f"{relation} is read end to end while filtering on "
                                f"{', '.join(missing)}."
                            ),
                            "suggestion": (
                                f'CREATE PROPERTY INDEX ON "{relation}" '
                                f'({", ".join(missing)});'
                            ),
                        }
                    )

        if _STARTS_WITH.search(predicates):
            findings.append(
                {
                    "kind": "starts_with_not_indexable",
                    "label": relation,
                    "detail": (
                        "STARTS WITH compiles to string_starts_with, which no btree can "
                        "serve, so this reads the whole label however selective the "
                        "prefix is."
                    ),
                    "suggestion": (
                        "Express the prefix as a range instead: "
                        "n.prop >= 'p' AND n.prop < 'q', where 'q' is 'p' with its last "
                        "character advanced by one. Tracked as AGV2-514."
                    ),
                }
            )

        if _JSONB_CONTAINS.search(predicates):
            findings.append(
                {
                    "kind": "bound_in_list_not_indexable",
                    "label": relation,
                    "detail": (
                        "A jsonb containment test in the filter is what an IN list bound "
                        "as a parameter becomes; only the other predicates reach an index."
                    ),
                    "suggestion": (
                        "Expand the list into an OR of equalities, one term per value, so "
                        "each reaches the index. Tracked as AGV2-515."
                    ),
                }
            )

    return findings


def format_findings(findings: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Wrap findings with the caveat that they are reasoned, not measured."""
    return {
        "findings": findings,
        "verified": False,
        "note": (
            "Reasoned from the query plan and the label catalogs. AgensGraph property "
            "indexes cannot be simulated before being built — a hypothetical index has "
            "to be given as a plain CREATE INDEX, and the expression a property index "
            "carries is not writable that way — so no recommendation here has been "
            "costed against a real plan. Build one and compare EXPLAIN before and after."
        ),
    }


def indexed_properties(rows: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """Which properties each label already has an index on.

    Read out of the rendered definition, which names the properties the way they were
    written when the index was created.
    """
    out: Dict[str, List[str]] = {}
    for row in rows:
        definition = row.get("definition") or ""
        names = _PROPERTY_REF.findall(definition)
        if not names:
            # A shorthand definition names the properties directly.
            inner = definition[definition.find("(") + 1 : definition.rfind(")")]
            names = [p.strip().strip('"') for p in inner.split(",") if p.strip()]
        out.setdefault(row["label"], []).extend(names)
    return out


def as_json(payload: Any) -> str:
    return json.dumps(payload, default=str)


def relname_to_label(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    """Row counts keyed by the relation name a plan reports."""
    return {row["relname"]: int(row["approx_rows"] or 0) for row in rows}


def missing_extension_note(name: str) -> Dict[str, Any]:
    return {
        "available": False,
        "note": (
            f"{name} is not installed. AgensGraph does not ship it; build it against this "
            f"server's pg_config and run CREATE EXTENSION {name} to enable this check."
        ),
    }


__all__ = [
    "OPTIONAL_EXTENSIONS",
    "analyze_plan",
    "as_json",
    "existing_indexes_query",
    "explain_statement",
    "extensions_query",
    "format_findings",
    "health_queries",
    "indexed_properties",
    "label_stats_query",
    "missing_extension_note",
    "relname_to_label",
    "top_cypher_queries_query",
]
