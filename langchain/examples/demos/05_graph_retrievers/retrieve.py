"""The same questions through three retrievers, side by side.

Every fact in the catalog is fictional, so the model answering has nothing to
lean on but what each retriever hands it -- when the vector-only answer comes
up empty-handed and the graph-context answer names the director, the difference
is the retrieval, not the model.

Run AFTER ingest.py:

    cd langchain
    .venv/bin/python examples/demos/05_graph_retrievers/retrieve.py
"""

from __future__ import annotations

import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from _common import agens, console, models
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from langchain_agensgraph.retrievers import (
    AgensGraphContextRetriever,
    AgensText2CypherRetriever,
    AgensVectorRetriever,
    render_graph_context,
)

GRAPH = "movie_retrievers"

QUESTIONS = [
    "Who directed the movie about divers finding a lighthouse that still burns "
    "underwater?",
    "Which other film was made by the director of the movie about mapping "
    "silences in wiretap recordings?",
]

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
    chain = ANSWER_PROMPT | llm
    return chain.invoke({"context": context, "question": question}).content


def ask(llm, name: str, retriever, question: str) -> None:
    started = time.perf_counter()
    docs = retriever.invoke(question)
    retrieval_ms = (time.perf_counter() - started) * 1000
    console.sub(f"{name}  (retrieval {retrieval_ms:.0f} ms)")
    print("  " + answer(llm, question, docs).replace("\n", "\n  "))


def main() -> None:
    llm = models.get_llm()
    store = agens.make_vector(
        models.get_embeddings(),
        graph_name=GRAPH,
        node_label="Movie",
        text_node_property="plot",
    )
    vector = AgensVectorRetriever(store=store, k=2)
    context = AgensGraphContextRetriever(
        store=store,
        k=2,
        expand_by_hops=2,
        document_formatter=render_graph_context,
    )

    for question in QUESTIONS:
        console.section(question)
        ask(llm, "vector only", vector, question)
        ask(llm, "vector + graph context (2 hops)", context, question)

    # Aggregates are a query, not a similarity -- the third retriever writes it.
    question = "How many films did Ines Varga direct?"
    console.section(question)
    graph = agens.make_graph(GRAPH)
    # The demo cluster's role is a superuser, which the read-only boundary
    # refuses by default; a real deployment gives the model a plain role.
    text2cypher = AgensText2CypherRetriever(
        graph=graph, llm=llm, allow_server_programs=True
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
