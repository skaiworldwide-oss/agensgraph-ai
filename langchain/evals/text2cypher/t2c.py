"""The text2cypher eval's shared pieces: dataset, fixture graphs, result canon.

The dataset never stores results — gold queries run against the fixture graphs
at scoring time, so a re-ingested graph or a new graphid layout cannot silently
stale the answers. What IS stored is the trap: each entry may carry the query a
model writes out of habit, with the outcome that makes the trap real (it errors,
or it runs and disagrees with gold). The validation test executes every gold and
every habit, so the dataset stays true against the engine it claims to describe.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any, Dict, List, Optional

import agensgraph

DATASET_PATH = pathlib.Path(__file__).with_name("dataset.jsonl")

FIXTURE_GRAPHS = ("t2c_movies", "t2c_traps")


def load_dataset(path: pathlib.Path = DATASET_PATH) -> List[Dict[str, Any]]:
    entries = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            entries.append(json.loads(line))
    return entries


def canonical(rows: List[Dict[str, Any]], ordered: bool = False) -> List[List[str]]:
    """Rows in a form where equal answers compare equal.

    Column names differ between equivalent queries (``RETURN n.name`` vs
    ``AS name``), so a row is its VALUES: each value dumped as sorted-key JSON,
    the dumps sorted within the row. Rows themselves are sorted unless the
    question fixed their order. Graph elements serialize through the driver
    (vertex -> id/label/properties), so two queries returning the same vertices
    agree; gold queries avoid returning raw ids where the comparison should
    survive a re-ingest.
    """
    plain = json.loads(agensgraph.to_json(rows).decode())
    canon_rows = []
    for row in plain:
        canon_rows.append(sorted(json.dumps(v, sort_keys=True) for v in row.values()))
    if not ordered:
        canon_rows.sort()
    return canon_rows


def results_match(
    got: List[Dict[str, Any]], want: List[Dict[str, Any]], ordered: bool = False
) -> bool:
    return canonical(got, ordered) == canonical(want, ordered)


def plan_has_index_path(conf: Dict[str, Any], graph: str, cypher: str) -> bool:
    """Whether any index can serve this statement at all.

    With ``enable_seqscan = off`` the planner still picks a sequential scan when
    nothing else exists, and marks it ``Disabled: true`` -- its own statement
    that no index path exists. That line, not which plan wins by cost, is what
    separates "the index cannot serve this predicate" from "the table is small".
    """
    import psycopg

    connect = {k: v for k, v in conf.items() if v is not None}
    with psycopg.connect(**connect) as conn:
        cur = conn.cursor()
        cur.execute(f"SET graph_path = {graph}")
        cur.execute("SET enable_seqscan = off")
        cur.execute("EXPLAIN (COSTS OFF) " + cypher)
        plan = "\n".join(row[0] for row in cur.fetchall())
    return "Disabled: true" not in plan


def requires_met(entry: Dict[str, Any], capabilities: Any) -> Optional[str]:
    """None when the server can run this entry, else the unmet requirement."""
    for requirement in entry.get("requires", ()):
        if requirement == "gql_clauses":
            if not capabilities.has_gql_clauses():
                return requirement
        elif requirement == "element_ordering":
            if not capabilities.has_element_ordering():
                return requirement
        elif requirement == "boolean_condition":
            # From 2.18 a condition takes only a boolean; 2.17 reads a value as truthy.
            if capabilities.version < (2, 18):
                return requirement
        else:
            return requirement
    return None


# ---- fixture graphs -------------------------------------------------------

MOVIES = [
    ("The Salt Meridian", 2019),
    ("Glasshouse Protocol", 2021),
    ("Winter Arithmetic", 2017),
    ("The Lantern Divers", 2020),
    ("Quiet Cartography", 2023),
    ("Marrow and Vine", 2018),
    ("The Eighth Ferry", 2022),
    ("Static Bloom", 2024),
]

DIRECTED = {
    "The Salt Meridian": "Ines Varga",
    "Winter Arithmetic": "Ines Varga",
    "Glasshouse Protocol": "Teodor Malik",
    "Quiet Cartography": "Teodor Malik",
    "The Lantern Divers": "Sana Whitfield",
    "Marrow and Vine": "Sana Whitfield",
    "The Eighth Ferry": "Ruth Okonjo",
    "Static Bloom": "Ruth Okonjo",
}

ACTED = {
    "The Salt Meridian": ["Greta Held", "Yusuf Adeyemi"],
    "Glasshouse Protocol": ["Milo Andersson", "Clara Voss"],
    "Winter Arithmetic": ["Clara Voss", "Daniel Rhee"],
    "The Lantern Divers": ["Yusuf Adeyemi", "Priya Nair"],
    "Quiet Cartography": ["Greta Held", "Daniel Rhee"],
    "Marrow and Vine": ["Priya Nair", "Milo Andersson"],
    "The Eighth Ferry": ["Clara Voss", "Priya Nair"],
    "Static Bloom": ["Milo Andersson", "Greta Held"],
}

GENRES = {
    "The Salt Meridian": ["mystery", "fantasy"],
    "Glasshouse Protocol": ["thriller"],
    "Winter Arithmetic": ["drama"],
    "The Lantern Divers": ["mystery", "adventure"],
    "Quiet Cartography": ["thriller", "mystery"],
    "Marrow and Vine": ["drama", "mystery"],
    "The Eighth Ferry": ["fantasy", "thriller"],
    "Static Bloom": ["romance", "fantasy"],
}


def build_movies(graph: Any) -> None:
    """The demo catalog without embeddings: mixed-case Movie, folded the rest."""
    for ddl in (
        'CREATE VLABEL IF NOT EXISTS "Movie"',
        "CREATE VLABEL IF NOT EXISTS person",
        "CREATE VLABEL IF NOT EXISTS genre",
        "CREATE ELABEL IF NOT EXISTS directed",
        "CREATE ELABEL IF NOT EXISTS acted_in",
        "CREATE ELABEL IF NOT EXISTS in_genre",
    ):
        graph.query(ddl)
    graph.query("MATCH (n) DETACH DELETE n")
    for title, year in MOVIES:
        graph.query(
            'CREATE (:"Movie" {title: %(t)s, year: %(y)s})',
            params={"t": title, "y": year},
        )
    people = sorted(set(DIRECTED.values()) | {a for c in ACTED.values() for a in c})
    for name in people:
        graph.query("CREATE (:person {name: %(n)s})", params={"n": name})
    for name in sorted({g for gs in GENRES.values() for g in gs}):
        graph.query("CREATE (:genre {name: %(n)s})", params={"n": name})
    for title, director in DIRECTED.items():
        graph.query(
            'MATCH (p:person {name: %(p)s}), (m:"Movie" {title: %(m)s}) '
            "CREATE (p)-[:directed]->(m)",
            params={"p": director, "m": title},
        )
    for title, cast in ACTED.items():
        for actor in cast:
            graph.query(
                'MATCH (p:person {name: %(p)s}), (m:"Movie" {title: %(m)s}) '
                "CREATE (p)-[:acted_in]->(m)",
                params={"p": actor, "m": title},
            )
    for title, genres in GENRES.items():
        for genre in genres:
            graph.query(
                'MATCH (m:"Movie" {title: %(m)s}), (g:genre {name: %(g)s}) '
                "CREATE (m)-[:in_genre]->(g)",
                params={"m": title, "g": genre},
            )
    # The indexes the sargability entries hold their plans against.
    for ddl in (
        'CREATE PROPERTY INDEX IF NOT EXISTS t2cm_title ON "Movie" (title)',
        'CREATE PROPERTY INDEX IF NOT EXISTS t2cm_year ON "Movie" (year)',
        "CREATE PROPERTY INDEX IF NOT EXISTS t2cm_person_name ON person (name)",
        "CREATE PROPERTY INDEX IF NOT EXISTS t2cm_genre_name ON genre (name)",
    ):
        graph.query(ddl)


def build_traps(graph: Any) -> None:
    """Shapes the dialect traps need, none of which a tidy dataset would have.

    A quoted upper-case relationship type next to folded labels, a stringly
    number, a list holding a JSON null, a quoted mixed-case property key, an
    inheritance chain, and a weighted road network with a longer direct route.
    """
    for ddl in (
        'CREATE VLABEL IF NOT EXISTS "Person"',
        'CREATE VLABEL IF NOT EXISTS student INHERITS ("Person")',
        "CREATE VLABEL IF NOT EXISTS city",
        'CREATE ELABEL IF NOT EXISTS "KNOWS"',
        "CREATE ELABEL IF NOT EXISTS lives_in",
        "CREATE ELABEL IF NOT EXISTS road",
    ):
        graph.query(ddl)
    graph.query("MATCH (n) DETACH DELETE n")
    graph.query(
        "CREATE (:\"Person\" {name: 'Ada', age: 36, scores: [1, null, 3], "
        "'nickName': 'The Countess', code: '007'})"
    )
    graph.query("CREATE (:\"Person\" {name: 'Bob', age: '42', code: '42'})")
    graph.query("CREATE (:\"Person\" {name: 'Cleo', age: 28, code: '7'})")
    graph.query("CREATE (:student {name: 'Dan', age: 20})")
    for name in ("Seoul", "Berlin", "Busan", "Tokyo"):
        graph.query("CREATE (:city {name: %(n)s})", params={"n": name})
    for a, b, since in (("Ada", "Bob", 2019), ("Bob", "Cleo", 2021)):
        graph.query(
            'MATCH (x:"Person" {name: %(a)s}), (y:"Person" {name: %(b)s}) '
            'CREATE (x)-[:"KNOWS" {since: %(s)s}]->(y)',
            params={"a": a, "b": b, "s": since},
        )
    for person, town in (
        ("Ada", "Seoul"),
        ("Bob", "Berlin"),
        ("Cleo", "Seoul"),
        ("Dan", "Seoul"),
    ):
        graph.query(
            'MATCH (p:"Person" {name: %(p)s}), (c:city {name: %(c)s}) '
            "CREATE (p)-[:lives_in]->(c)",
            params={"p": person, "c": town},
        )
    for a, b, km in (
        ("Seoul", "Tokyo", 12),
        ("Seoul", "Busan", 2),
        ("Busan", "Tokyo", 3),
        ("Berlin", "Seoul", 80),
    ):
        graph.query(
            "MATCH (x:city {name: %(a)s}), (y:city {name: %(b)s}) "
            "CREATE (x)-[:road {km: %(k)s}]->(y)",
            params={"a": a, "b": b, "k": km},
        )
    # A label's children do not inherit its indexes, and a MATCH on the parent
    # scans the children too -- one unindexed child and the whole read loses
    # its index path. So every "Person" index exists on student as well.
    for label, prefix in (('"Person"', "t2ct_person"), ("student", "t2ct_student")):
        for suffix, keys in (
            ("name", "(name)"),
            ("age", "(age)"),
            ("code", "(code)"),
            ("scores", "(scores)"),
            ("lower", "((tolower(name)))"),
        ):
            graph.query(
                f"CREATE PROPERTY INDEX IF NOT EXISTS {prefix}_{suffix} "
                f"ON {label} {keys}"
            )
    graph.query("CREATE PROPERTY INDEX IF NOT EXISTS t2ct_city_name ON city (name)")


BUILDERS = {"t2c_movies": build_movies, "t2c_traps": build_traps}
