"""Build the Meridian Pictures catalog: eight films, their people, their genres.

Every film, person and plot here is invented, which is the point of the demo:
an LLM cannot answer questions about this catalog from what it already knows,
so whatever the answers contain came out of the retrievers.

Movies are the vector store's own nodes (label ``Movie``, plot text embedded);
directors, actors and genres hang off them as ordinary graph structure. The
film's title is its ``__id__``, so the edge wiring reads as what it says.

    cd langchain
    .venv/bin/python examples/demos/05_graph_retrievers/ingest.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from _common import agens, console, models
from langchain_agensgraph import AgensgraphVector

GRAPH = "movie_retrievers"

MOVIES = [
    (
        "The Salt Meridian",
        2019,
        "A cartographer discovers that every map she has ever drawn of the salt "
        "flats is slowly rewriting the terrain itself.",
    ),
    (
        "Glasshouse Protocol",
        2021,
        "An insomniac negotiator is locked inside a transparent embassy during a "
        "seventy-two hour siege.",
    ),
    (
        "Winter Arithmetic",
        2017,
        "Two rival mathematicians snowed into a mountain observatory race to "
        "finish a proof their late mentor left in fragments.",
    ),
    (
        "The Lantern Divers",
        2020,
        "Deep-sea salvage divers find a sunken lighthouse whose lamp is somehow "
        "still burning.",
    ),
    (
        "Quiet Cartography",
        2023,
        "A retired spy maps the silences in old wiretap recordings and hears a "
        "confession no one ever spoke aloud.",
    ),
    (
        "Marrow and Vine",
        2018,
        "A vineyard forensic botanist traces a decades-old disappearance through "
        "the rings of a single grapevine.",
    ),
    (
        "The Eighth Ferry",
        2022,
        "A night-shift ferry pilot realizes her seven scheduled crossings keep "
        "producing an eighth she cannot remember making.",
    ),
    (
        "Static Bloom",
        2024,
        "A radio astronomer falls for the voice inside an interference pattern "
        "that predicts tomorrow's weather exactly.",
    ),
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


def main() -> None:
    console.section("Meridian Pictures — catalog ingest")
    embeddings = models.get_embeddings()

    with console.timer("embed + store 8 plots"):
        store = AgensgraphVector.from_texts(
            texts=[plot for _, _, plot in MOVIES],
            metadatas=[{"title": t, "year": y} for t, y, _ in MOVIES],
            ids=[title for title, _, _ in MOVIES],
            embedding=embeddings,
            graph_name=GRAPH,
            node_label="Movie",
            text_node_property="plot",
            pre_delete_collection=True,
            engine=agens.get_engine(),
        )

    with console.timer("people, genres and their edges"):
        for ddl in (
            "CREATE VLABEL IF NOT EXISTS person",
            "CREATE VLABEL IF NOT EXISTS genre",
            "CREATE ELABEL IF NOT EXISTS directed",
            "CREATE ELABEL IF NOT EXISTS acted_in",
            "CREATE ELABEL IF NOT EXISTS in_genre",
        ):
            store.query(ddl)
        people = sorted(set(DIRECTED.values()) | {a for c in ACTED.values() for a in c})
        for name in people:
            store.query("MERGE (:person {name: %(name)s})", params={"name": name})
        for name in sorted({g for gs in GENRES.values() for g in gs}):
            store.query("MERGE (:genre {name: %(name)s})", params={"name": name})
        for title, director in DIRECTED.items():
            store.query(
                'MATCH (p:person {name: %(p)s}), (m:"Movie") '
                "WHERE m.__id__ = %(m)s CREATE (p)-[:directed]->(m)",
                params={"p": director, "m": title},
            )
        for title, cast in ACTED.items():
            for actor in cast:
                store.query(
                    'MATCH (p:person {name: %(p)s}), (m:"Movie") '
                    "WHERE m.__id__ = %(m)s CREATE (p)-[:acted_in]->(m)",
                    params={"p": actor, "m": title},
                )
        for title, genres in GENRES.items():
            for genre in genres:
                store.query(
                    'MATCH (m:"Movie"), (g:genre {name: %(g)s}) '
                    "WHERE m.__id__ = %(m)s CREATE (m)-[:in_genre]->(g)",
                    params={"g": genre, "m": title},
                )

    counts = store.query(
        "MATCH (n) WITH count(n) AS vertices MATCH ()-[e]->() "
        "RETURN vertices, count(e) AS edges"
    )[0]
    console.kv("graph", GRAPH)
    console.kv("vertices", counts["vertices"])
    console.kv("edges", counts["edges"])
    store.close()
    agens.close()


if __name__ == "__main__":
    main()
