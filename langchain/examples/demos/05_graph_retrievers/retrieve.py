"""The same questions through three retrievers, side by side.

Every fact in the catalog is fictional, so the model answering has nothing to
lean on but what each retriever hands it -- when the vector-only answer comes
up empty-handed and the graph-context answer names the director, the difference
is the retrieval, not the model.

The third retriever runs Cypher the model writes, as the role AGENSGRAPH_UNTRUSTED_URL names,
which the demo grants read access to the graph first.

Run AFTER ingest.py:

    cd langchain
    .venv/bin/python examples/demos/05_graph_retrievers/retrieve.py
"""

from __future__ import annotations

import pathlib
import sys
import time
from typing import List, Tuple

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from _common import agens, config, console, models
from langchain_agensgraph import AgensGraph, AgensgraphVector
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

COUNT_QUESTION = "How many films did Ines Varga direct?"

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


def retrievers(store: AgensgraphVector) -> Tuple[AgensVectorRetriever, AgensGraphContextRetriever]:
    """The vector retriever, and the same search with each hit's two-hop neighbourhood."""
    vector = AgensVectorRetriever(vectorstore=store, k=2)
    context = AgensGraphContextRetriever(
        vectorstore=store,
        k=2,
        hops=2,
        document_formatter=render_graph_context,
    )
    return vector, context


def answer(llm, question: str, docs: List[Document]) -> str:
    context = "\n\n---\n\n".join(d.page_content for d in docs)
    chain = ANSWER_PROMPT | llm
    return chain.invoke({"context": context, "question": question}).text


def ask(llm, name: str, retriever, question: str) -> List[Document]:
    started = time.perf_counter()
    docs = retriever.invoke(question)
    retrieval_ms = (time.perf_counter() - started) * 1000
    console.sub(f"{name}  (retrieval {retrieval_ms:.0f} ms)")
    print("  " + answer(llm, question, docs).replace("\n", "\n  "))
    return docs


def text2cypher(graph: AgensGraph, llm, question: str, **kwargs) -> List[Document]:
    """The rows the model's Cypher returns, one document each, the statement in metadata."""
    retriever = AgensText2CypherRetriever(
        graph=graph, llm=llm, allow_dangerous_requests=True, **kwargs
    )
    started = time.perf_counter()
    docs = retriever.invoke(question)
    retrieval_ms = (time.perf_counter() - started) * 1000
    console.sub(f"text2cypher  (generation + retrieval {retrieval_ms:.0f} ms)")
    console.kv("cypher", docs[0].metadata["cypher"] if docs else "(no rows)")
    for doc in docs:
        console.kv("row", doc.page_content)
    return docs


def main() -> None:
    config.require_openai_key()
    config.require_untrusted_url()
    llm = models.get_llm()
    store = agens.make_vector(
        models.get_embeddings(), graph_name=GRAPH, label="Movie", text_property="plot"
    )
    try:
        vector, context = retrievers(store)
        for question in QUESTIONS:
            console.section(question)
            ask(llm, "vector only", vector, question)
            ask(llm, "vector + graph context (2 hops)", context, question)

        # Aggregates are a query, not a similarity -- the third retriever writes it.
        console.section(COUNT_QUESTION)
        agens.grant(GRAPH)
        text2cypher(agens.make_graph(GRAPH), llm, COUNT_QUESTION)
    finally:
        agens.close()


if __name__ == "__main__":
    main()
