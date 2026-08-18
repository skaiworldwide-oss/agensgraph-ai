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
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from _common import agens, console, models
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from langchain_agensgraph import AgensgraphVector
from langchain_agensgraph.retrievers import (
    AgensGraphContextRetriever,
    AgensText2CypherRetriever,
    AgensVectorRetriever,
    render_graph_context,
)

GRAPH = "arxiv"

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer from the context alone. If the context does not contain "
            "the answer, say so plainly.",
        ),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ]
)


def answer(llm, question: str, docs: list[Document]) -> str:
    context = "\n\n---\n\n".join(d.page_content for d in docs)
    return (ANSWER_PROMPT | llm).invoke(
        {"context": context, "question": question}
    ).content


def ask(llm, name: str, retriever, question: str) -> None:
    started = time.perf_counter()
    docs = retriever.invoke(question)
    retrieval_ms = (time.perf_counter() - started) * 1000
    console.sub(f"{name}  (retrieval {retrieval_ms:.0f} ms)")
    print("  " + answer(llm, question, docs).replace("\n", "\n  "))


def main() -> None:
    engine = agens.get_engine()
    with engine.connection(GRAPH) as conn:
        papers = conn.execute_query('MATCH (p:"Paper") RETURN count(*) AS n')
        if not papers.records or papers.records[0][0] < 1000:
            raise SystemExit(
                "The arxiv graph is empty. Run "
                "examples/demos/01_arxiv_graphrag/prepare.py first."
            )

    llm = models.get_llm()
    store = AgensgraphVector.from_existing_index(
        embedding=models.get_embeddings(),
        engine=engine,
        graph_name=GRAPH,
        node_label="Paper",
        text_node_property="abstract",
        embedding_node_property="embedding",
        index_name="paper_vec",
    )
    vector = AgensVectorRetriever(store=store, k=3)
    # One hop from a paper reaches its authors, categories and year; an
    # abstract never says any of those.
    context = AgensGraphContextRetriever(
        store=store,
        k=3,
        expand_by_hops=1,
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
    graph = agens.make_graph(GRAPH)
    # The demo cluster's role is a superuser, which the read-only boundary
    # refuses by default; a real deployment gives the model a plain role.
    # retry_on_empty earns its keep here: the model tends to write AUTHORED_BY
    # against its schema direction, which runs clean and matches nothing.
    text2cypher = AgensText2CypherRetriever(
        graph=graph,
        llm=llm,
        k=3,
        max_retries=2,
        retry_on_empty=True,
        allow_server_programs=True,
    )
    started = time.perf_counter()
    docs = text2cypher.invoke(question)
    retrieval_ms = (time.perf_counter() - started) * 1000
    console.sub(f"text2cypher  (generation + retrieval {retrieval_ms:.0f} ms)")
    console.kv("cypher", docs[0].metadata["cypher"] if docs else "(no rows)")
    for doc in docs:
        console.kv("row", doc.page_content)

    store.close()
    graph.close()
    agens.close()


if __name__ == "__main__":
    main()
