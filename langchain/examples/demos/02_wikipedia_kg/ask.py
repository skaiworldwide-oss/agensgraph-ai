"""Wikipedia knowledge graph — ask (Text2Cypher with LangChain LCEL).

A natural-language question is turned into AgensGraph Cypher by the LLM, run
against the graph built by build_kg.py, and the rows are used to ground a final
answer. The whole thing is one LangChain Expression Language (LCEL) pipeline:

    {schema, question}
        | (cypher prompt | llm | parse | clean)        -> cypher
        | run cypher (read-only, timed)                -> results
        | answer prompt | llm | parse                  -> answer

    cd langchain
    .venv/bin/python examples/demos/02_wikipedia_kg/ask.py
    .venv/bin/python examples/demos/02_wikipedia_kg/ask.py "your question"
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from _common import agens, config, console
from _common.models import get_llm

GRAPH = "wikipedia_kg"

CYPHER_SYSTEM = """You translate questions into a SINGLE read-only AgensGraph Cypher query.

AgensGraph Cypher rules (important):
- Quote every label with double quotes: (n:"Person").
- Relationship types are MANY and entity-specific. PREFER an untyped relationship
  (a)-[r]->(b) and read its kind with type(r); do NOT guess a relationship type.
  Only write -[:"TYPE"]-> if that exact TYPE appears in the schema's relationship list.
- Match properties inline {{name: 'X'}} or in WHERE (n.name = 'X'); access as n.name.
- Use ONLY labels / properties that appear in the schema.
- Read-only ONLY: MATCH / OPTIONAL MATCH / WHERE / RETURN / WITH / ORDER BY / LIMIT.
  Never CREATE / MERGE / SET / DELETE / REMOVE / DROP.
- Always add a LIMIT (<= 50). Return ONLY the query — no markdown, no explanation.

Examples:
  Q: How many entities of each type are there?
  MATCH (n) RETURN label(n) AS type, count(*) AS n ORDER BY n DESC LIMIT 50

  Q: Which people are connected to the most other entities?
  MATCH (p:"Person")-[r]->(e) RETURN p.id AS person, count(e) AS connections
  ORDER BY connections DESC LIMIT 5"""

ANSWER_SYSTEM = (
    "Answer the user's question using ONLY the Cypher query results. "
    "If the results are empty, say you couldn't find it in the graph. Be concise."
)

DEFAULT_QUESTIONS = [
    "What types of entities are in the graph, and how many of each?",
    "Which 5 Wikipedia articles mention the most distinct entities?",
    "Show 8 example relationships as source, relationship, target.",
]

_WRITE = re.compile(r"\b(CREATE|MERGE|SET|DELETE|REMOVE|DROP|DETACH)\b", re.IGNORECASE)


def _clean_cypher(text: str) -> str:
    """Strip markdown fences / prose the LLM may add; keep the query."""
    text = re.sub(r"```(?:cypher)?", "", text).strip()
    return text.rstrip(";").strip()


def build_chain(graph, llm):
    cypher_prompt = ChatPromptTemplate.from_messages(
        [("system", CYPHER_SYSTEM), ("human", "Schema:\n{schema}\n\nQuestion: {question}")]
    )
    answer_prompt = ChatPromptTemplate.from_messages(
        [("system", ANSWER_SYSTEM),
         ("human", "Question: {question}\n\nCypher:\n{cypher}\n\nResults:\n{results}")]
    )

    cypher_chain = cypher_prompt | llm | StrOutputParser() | RunnableLambda(_clean_cypher)

    def run_cypher(d: dict) -> str:
        cypher = d["cypher"]
        if _WRITE.search(cypher):
            return "(refused: query is not read-only)"
        try:
            rows = graph.query(cypher, timeout=15)
            return json.dumps(rows[:25], default=str)
        except Exception as e:  # surface the error to the answer step
            return f"(query error: {e})"

    # The full LCEL pipeline: generate cypher -> run it -> answer from results.
    return (
        RunnablePassthrough.assign(cypher=cypher_chain)
        | RunnablePassthrough.assign(results=RunnableLambda(run_cypher))
        | RunnablePassthrough.assign(
            answer=(answer_prompt | llm | StrOutputParser())
        )
    )


def main() -> None:
    config.require_openai_key()
    questions = [sys.argv[1]] if len(sys.argv) > 1 else DEFAULT_QUESTIONS

    # enhanced_schema=True gives the LLM example property values for better Cypher.
    graph = agens.make_graph(GRAPH, create=False, enhanced_schema=True)
    try:
        schema = graph.get_schema
        console.section("graph schema (fed to the LLM)")
        print(schema.strip()[:1200])

        chain = build_chain(graph, get_llm())
        for q in questions:
            console.section(f"Q: {q}")
            out = chain.invoke({"schema": schema, "question": q})
            print("Cypher:\n  " + out["cypher"].replace("\n", "\n  "))
            print("\nAnswer:\n" + out["answer"])
    finally:
        agens.close()


if __name__ == "__main__":
    main()
