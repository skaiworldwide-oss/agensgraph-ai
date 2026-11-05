"""The same three retrievers, on the real thing: 138k vertices of arXiv.

Where the movie catalog proves the mechanics on data the model cannot know,
this leg runs the identical code against the ``arxiv`` graph demo 01 builds --
50,000 papers, 88,000 authors, 275,000 edges -- because an abstract never names
its authors: the vector search alone cannot answer "who wrote it", and the
graph context can.

Run AFTER 01_arxiv_graphrag/prepare.py has populated the ``arxiv`` graph:

    cd langchain
    .venv/bin/python examples/demos/05_graph_retrievers/arxiv.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from _common import agens, config, console, models
from langchain_agensgraph import AgensgraphVector
from langchain_agensgraph.retrievers import (
    AgensGraphContextRetriever,
    AgensVectorRetriever,
    render_graph_context,
)
from retrieve import ask, text2cypher

GRAPH = "arxiv"
# Enough papers for the graph to have the hubs this leg is about; the fixture has 50.
MIN_PAPERS = 50 if config.fake() else 1000


def main() -> None:
    config.require_openai_key()
    config.require_untrusted_url()
    graph = agens.make_graph(GRAPH)
    try:
        papers = graph.query('MATCH (p:"Paper") RETURN count(*) AS n')[0]["n"]
        if papers < MIN_PAPERS:
            raise SystemExit(
                "The arxiv graph is empty. Run "
                "examples/demos/01_arxiv_graphrag/prepare.py first."
            )

        llm = models.get_llm()
        store = AgensgraphVector.from_existing_index(
            models.get_embeddings(),
            engine=agens.get_engine(),
            graph_name=GRAPH,
            label="Paper",
        )
        vector = AgensVectorRetriever(vectorstore=store, k=3)
        # One hop from a paper reaches its authors, categories and year; an
        # abstract never says any of those.
        context = AgensGraphContextRetriever(
            vectorstore=store,
            k=3,
            hops=1,
            document_formatter=render_graph_context,
        )

        question = (
            "Who are the authors of the paper about the fate of dwarf galaxies "
            "in galaxy clusters?"
        )
        console.section(question)
        ask(llm, "vector only", vector, question)
        ask(llm, "vector + graph context (1 hop)", context, question)

        question = "Which author has written the most papers?"
        console.section(question)
        agens.grant(GRAPH)
        # retry_on_empty earns its keep here: the model tends to write AUTHORED_BY
        # against its schema direction, which runs clean and matches nothing.
        text2cypher(graph, llm, question, k=3, max_retries=2, retry_on_empty=True)
    finally:
        agens.close()


if __name__ == "__main__":
    main()
