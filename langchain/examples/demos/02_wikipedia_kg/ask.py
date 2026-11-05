"""Wikipedia knowledge graph — ask a question in plain language.

`AgensCypherQAChain` turns the question into AgensGraph Cypher against the schema of the graph
built by build_kg.py, runs it as the untrusted role, and answers from the rows:

    {"query": question}
        -> schema + question -> model             -> cypher
        -> quote names, turn edges round          -> a statement for this server
        -> query_untrusted, as llm_ro, read-only  -> rows (a refusal goes back to the model)
        -> hand the rows to the model
        -> {"query", "result", "intermediate_steps"}

The chain carries the AgensGraph dialect rules, so the demo does not have to: labels are quoted
(an unquoted one folds to lower case and matches nothing), and which clauses the model is told to
avoid depends on the server's capabilities. The statement runs as the role
AGENSGRAPH_UNTRUSTED_URL names, which the demo grants read access to the graph first
(``langchain_agensgraph.roles.render_grants``).

    cd langchain
    .venv/bin/python examples/demos/02_wikipedia_kg/ask.py
    .venv/bin/python examples/demos/02_wikipedia_kg/ask.py "your question"
"""

from __future__ import annotations

import pathlib
import sys
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from _common import agens, config, console
from _common.models import get_llm
from langchain_agensgraph import AgensCypherQAChain, AgensGraph
from langchain_agensgraph.errors import AgensQueryError, UnsafeCypherError

GRAPH = "wikipedia_kg"

DEFAULT_QUESTIONS = [
    "What types of entities are in the graph, and how many of each?",
    "Which 5 Wikipedia articles mention the most distinct entities?",
    "Show 8 example relationships as source, relationship, target.",
]


def make_chain(graph: AgensGraph, llm) -> AgensCypherQAChain:
    return AgensCypherQAChain.from_llm(
        llm,
        graph=graph,
        top_k=25,
        return_intermediate_steps=True,  # so the demo can show the generated query
        # The chain runs model-written Cypher as the engine's untrusted role; this says so.
        allow_dangerous_requests=True,
    )


def ask(chain: AgensCypherQAChain, question: str) -> Tuple[str, List[Dict[str, Any]], str]:
    """The statement that ran, its rows, and the answer."""
    out = chain.invoke({"query": question})
    # Earlier attempts, if any, come first; the last two steps are the statement and its rows.
    steps = out["intermediate_steps"]
    return steps[-2]["query"], steps[-1]["context"], out["result"]


def main() -> None:
    config.require_openai_key()
    config.require_untrusted_url()
    questions = [sys.argv[1]] if len(sys.argv) > 1 else DEFAULT_QUESTIONS

    # enhanced_schema=True samples example property values, which the chain puts in the
    # prompt so the model writes queries against values that actually occur.
    graph = agens.make_graph(GRAPH, enhanced_schema=True)
    try:
        agens.grant(GRAPH)
        console.section("graph schema (fed to the model)")
        print(graph.get_schema.strip()[:1200])

        chain = make_chain(graph, get_llm())
        for question in questions:
            console.section(f"Q: {question}")
            try:
                cypher, rows, answer = ask(chain, question)
            except (UnsafeCypherError, AgensQueryError) as refused:
                # A statement the reading refused, or one the server would not run, after the
                # model's corrections ran out. Anything else is not about the question.
                print(f"(refused: {refused})")
                continue
            print("Cypher:\n  " + cypher.replace("\n", "\n  "))
            print(f"\nRows: {len(rows)}")
            print("\nAnswer:\n" + answer)
    finally:
        agens.close()


if __name__ == "__main__":
    main()
