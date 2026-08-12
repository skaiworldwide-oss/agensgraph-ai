"""Wikipedia knowledge graph — ask a question in plain language.

`AgensCypherQAChain` turns the question into AgensGraph Cypher against the schema of the
graph built by build_kg.py, runs it read-only, and answers from the rows:

    {"query": question}
        -> schema + question -> model      -> cypher
        -> repair, refuse writes, EXPLAIN  -> a query that will run
        -> run it, hand the rows to the model
        -> {"query", "result", "intermediate_steps"}

The chain carries the AgensGraph dialect rules, so the demo does not have to: labels are
quoted (an unquoted one folds to lower case and matches nothing), and the query is checked
with EXPLAIN before it runs. Which clauses the model is told to avoid depends on the
server -- `CALL { }`, `COUNT { }` and `EXISTS { }` are ruled out only where the connected
server lacks them, which the chain asks its capabilities rather than assuming.

    cd langchain
    .venv/bin/python examples/demos/02_wikipedia_kg/ask.py
    .venv/bin/python examples/demos/02_wikipedia_kg/ask.py "your question"
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from langchain_agensgraph import AgensCypherQAChain

from _common import agens, config, console
from _common.models import get_llm

GRAPH = "wikipedia_kg"

DEFAULT_QUESTIONS = [
    "What types of entities are in the graph, and how many of each?",
    "Which 5 Wikipedia articles mention the most distinct entities?",
    "Show 8 example relationships as source, relationship, target.",
]


def main() -> None:
    config.require_openai_key()
    questions = [sys.argv[1]] if len(sys.argv) > 1 else DEFAULT_QUESTIONS

    # enhanced_schema=True samples example property values, which the chain puts in the
    # prompt so the model writes queries against values that actually occur.
    graph = agens.make_graph(GRAPH, create=False, enhanced_schema=True)
    try:
        console.section("graph schema (fed to the model)")
        print(graph.get_schema.strip()[:1200])

        chain = AgensCypherQAChain.from_llm(
            get_llm(),
            graph=graph,
            top_k=25,
            return_intermediate_steps=True,  # so the demo can show the generated query
            # What refuses a generated write is a read-only transaction, and one is not a
            # boundary for a role that may run a command on the server's host -- which the
            # role these demos connect as can, being the one that created the databases.
            # The refusal is accepted here so the demo runs; an application serving
            # questions from the public should connect as a role that cannot, and leave
            # this alone.
            allow_server_programs=True,
        )

        for question in questions:
            console.section(f"Q: {question}")
            try:
                out = chain.invoke({"query": question})
            except ValueError as refused:
                # A generated write, or Cypher that EXPLAIN says will not run. Anything
                # else -- a server that is not there, a role the boundary cannot be opened
                # for -- is not about the question and is left to surface.
                print(f"(refused: {refused})")
                continue
            cypher, rows = out["intermediate_steps"]
            print("Cypher:\n  " + cypher["query"].replace("\n", "\n  "))
            print(f"\nRows: {len(rows['context'])}")
            print("\nAnswer:\n" + out["result"])
    finally:
        agens.close()


if __name__ == "__main__":
    main()
