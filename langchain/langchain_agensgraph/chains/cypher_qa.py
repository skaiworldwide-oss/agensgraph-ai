"""Answer a natural-language question by generating and running Cypher.

Two entry points over the same pipeline:

* :class:`AgensCypherQAChain` — ``from_llm(...)`` then ``invoke({"query": ...})``,
  the shape a graph question-answering chain conventionally has.
* :func:`create_cypher_tool` — the same thing as a tool an agent can call.

The pipeline reads the graph's schema, asks the model for a query, repairs and checks it,
runs it read-only, and asks the model to answer from the rows it got back.

Two things about AgensGraph shape the prompt more than anything else. Unquoted identifiers
fold to lower case, so ``(n:Person)`` matches nothing and every label has to be written
``(n:"Person")``. And AgensGraph is not Neo4j: constructs a model reaches for out of habit
— pattern expressions, ``COUNT { }``, ``EXISTS { }``, ``apoc.*`` — are not available, so
the prompt rules them out by name rather than leaving the model to discover it.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from langchain_agensgraph.graphs.agensgraph import AgensGraph

CYPHER_SYSTEM = """\
Generate one read-only AgensGraph (openCypher) query answering the user's question.

AgensGraph dialect — these differ from Neo4j and matter:
- Quote every label and relationship type: (n:"Person"), (a)-[r:"WORKS_AT"]->(b).
  Unquoted identifiers fold to lower case and match nothing.
- Quote mixed-case property names the same way: n."firstName". All-lowercase names
  need no quotes: n.name.
- No pattern expressions inside expressions: no size((n)--()), no [(n)-->(m) | m].
- No COUNT { ... }, no EXISTS { ... }, no CALL { ... } subqueries, no apoc.*.
  To count a node's relationships, MATCH them and use count(*).
- Use count(*) rather than count(n) over a node variable: count(n) materializes every
  property of every node, including any embeddings, and is far slower.
- Prefer an untyped relationship (a)-[r]->(b) and read type(r) unless the exact
  relationship type appears in the schema.

Hard rules:
- Read-only: MATCH / OPTIONAL MATCH / WHERE / WITH / RETURN / ORDER BY / SKIP / LIMIT.
  Never CREATE / MERGE / SET / DELETE / REMOVE / DROP / DETACH / LOAD.
- Use only the labels, relationship types and properties shown in the schema.
- Always end with a LIMIT of at most {top_k}.
- Return only the query. No prose, no markdown fences, no trailing semicolon.

Examples:
  Q: How many of each kind of node are there?
  MATCH (n) RETURN label(n) AS label, count(*) AS n ORDER BY n DESC LIMIT {top_k}

  Q: Who does Alice work with?
  MATCH (a:"Person")-[r]->(b:"Person") WHERE a.name = 'Alice'
  RETURN b.name AS name, type(r) AS relationship LIMIT {top_k}\
"""

QA_SYSTEM = """\
Answer the question using only the query results provided. If they are empty, say the
graph does not contain that information. Do not invent detail the results do not show.
Be concise.\
"""

# Clauses that write. Matched against the query with comments and strings removed, so a
# label or a string containing the word cannot look like one.
_WRITE = re.compile(
    r"\b(CREATE|MERGE|SET|DELETE|REMOVE|DROP|DETACH|LOAD)\b", re.IGNORECASE
)
_COMMENTS_AND_STRINGS = re.compile(
    r"//[^\n]*|/\*.*?\*/|'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\"", re.DOTALL
)
_FENCE = re.compile(r"```(?:cypher|sql)?", re.IGNORECASE)

# Identifiers the model may have left unquoted. Any identifier that is not already all
# lower case needs quoting, since AgensGraph folds an unquoted one: firstName becomes
# firstname and matches nothing. ":" after ":" is a type cast, not a label.
_LABEL = re.compile(r'(?<!:):(?!")([A-Za-z_][A-Za-z0-9_]*)')
_PROP_KEY = re.compile(r'([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)')
_PROP_ACCESS = re.compile(r'\.(?!")([A-Za-z_][A-Za-z0-9_]*)\b')
_STRING = re.compile(r"'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\"", re.DOTALL)


def strip_fences(text: str) -> str:
    """Take the query out of whatever the model wrapped it in."""
    return _FENCE.sub("", text).strip().rstrip(";").strip()


def is_write_query(cypher: str) -> bool:
    """Does the query contain a write clause, ignoring comments and strings?"""
    return _WRITE.search(_COMMENTS_AND_STRINGS.sub(" ", cypher)) is not None


def case_sensitive_names(structured_schema: Dict[str, Any]) -> set:
    """Identifiers in the graph whose spelling is not all lower case.

    Folding runs both ways: a label or key written unquoted is *stored* folded, so
    ``n.firstName`` is right for data created as ``{"firstName": ...}`` and wrong for data
    created as ``{firstName: ...}``, which is stored as ``firstname``. Only the schema
    knows which exists, so it decides what may be quoted.
    """
    names: set = set()

    def add(value: Any) -> None:
        if isinstance(value, str) and value != value.lower():
            names.add(value)

    for section in ("node_props", "rel_props"):
        for label, props in (structured_schema.get(section) or {}).items():
            add(label)
            for prop in props or []:
                add(prop.get("property") if isinstance(prop, dict) else prop)
    for rel in structured_schema.get("relationships") or []:
        if isinstance(rel, dict):
            for key in ("type", "relationship_type", "start", "end", "label"):
                add(rel.get(key))
    return names


def quote_identifiers(cypher: str, known: Optional[set] = None) -> str:
    """Quote the identifiers the model left bare that the graph stores case-sensitively.

    An identifier is quoted only when the graph actually holds that exact spelling —
    quoting one the graph stores folded would turn a working query into one that matches
    nothing, which is the failure this is meant to prevent. Text inside string literals is
    left alone. With no ``known`` set, nothing is quoted.
    """
    allowed = known or set()

    def needs_quoting(name: str) -> bool:
        return name in allowed

    def repair(fragment: str) -> str:
        fragment = _LABEL.sub(
            lambda m: f':"{m.group(1)}"' if needs_quoting(m.group(1)) else m.group(0),
            fragment,
        )
        fragment = _PROP_KEY.sub(
            lambda m: (
                f'{m.group(1)}"{m.group(2)}"{m.group(3)}'
                if needs_quoting(m.group(2))
                else m.group(0)
            ),
            fragment,
        )
        return _PROP_ACCESS.sub(
            lambda m: f'."{m.group(1)}"' if needs_quoting(m.group(1)) else m.group(0),
            fragment,
        )

    out, last = [], 0
    for match in _STRING.finditer(cypher):
        out.append(repair(cypher[last : match.start()]))
        out.append(match.group(0))
        last = match.end()
    out.append(repair(cypher[last:]))
    return "".join(out)


def render_cypher_system(template: str, top_k: int) -> str:
    """Fill in the row limit and escape the braces that are part of the prose.

    The dialect rules name constructs like ``COUNT { ... }``, whose braces a prompt
    template would otherwise read as variables.
    """
    filled = template.replace("{top_k}", str(top_k))
    return filled.replace("{", "{{").replace("}", "}}")


class _Question(BaseModel):
    question: str = Field(description="A question to answer from the graph.")


class AgensCypherQAChain(Runnable[Dict[str, Any], Dict[str, Any]]):
    """Question answering over an AgensGraph graph by generated Cypher.

    Build with :meth:`from_llm` and call ``invoke({"query": "..."})``. The result carries
    ``query`` and ``result``; ask for ``return_intermediate_steps`` to also get the
    generated Cypher and the rows it returned.

    Generated Cypher is not trusted: writes are refused, the query is checked with
    ``EXPLAIN`` before it runs, and execution carries a timeout.
    """

    def __init__(
        self,
        *,
        graph: AgensGraph,
        cypher_llm: BaseLanguageModel,
        qa_llm: BaseLanguageModel,
        top_k: int = 10,
        timeout: Optional[float] = 30.0,
        validate_cypher: bool = True,
        allow_dangerous_requests: bool = False,
        cypher_prompt: Optional[ChatPromptTemplate] = None,
        qa_prompt: Optional[ChatPromptTemplate] = None,
        return_intermediate_steps: bool = False,
    ) -> None:
        self.graph = graph
        self.cypher_llm = cypher_llm
        self.qa_llm = qa_llm
        self.top_k = top_k
        self.timeout = timeout
        self.validate_cypher = validate_cypher
        self.allow_dangerous_requests = allow_dangerous_requests
        self.return_intermediate_steps = return_intermediate_steps
        self.cypher_prompt = cypher_prompt or ChatPromptTemplate.from_messages(
            [
                ("system", render_cypher_system(CYPHER_SYSTEM, top_k)),
                ("human", "Schema:\n{schema}\n\nQuestion: {question}"),
            ]
        )
        self.qa_prompt = qa_prompt or ChatPromptTemplate.from_messages(
            [
                ("system", QA_SYSTEM),
                ("human", "Question: {question}\n\nResults:\n{results}"),
            ]
        )

    @classmethod
    def from_llm(
        cls,
        llm: Optional[BaseLanguageModel] = None,
        *,
        graph: AgensGraph,
        cypher_llm: Optional[BaseLanguageModel] = None,
        qa_llm: Optional[BaseLanguageModel] = None,
        **kwargs: Any,
    ) -> "AgensCypherQAChain":
        """Build from one model, or from a separate model for each step."""
        cypher_llm = cypher_llm or llm
        qa_llm = qa_llm or llm
        if cypher_llm is None or qa_llm is None:
            raise ValueError(
                "Provide `llm`, or both `cypher_llm` and `qa_llm`."
            )
        return cls(graph=graph, cypher_llm=cypher_llm, qa_llm=qa_llm, **kwargs)

    # ---- pipeline steps ----

    def _schema(self) -> str:
        """The schema for the prompt, cached per the graph's ``schema_cache_ttl``."""
        return self.graph.get_schema

    def _known_names(self) -> set:
        """Which identifiers the graph stores case-sensitively."""
        return case_sensitive_names(self.graph.get_structured_schema)

    def generate_cypher(self, question: str) -> str:
        chain = self.cypher_prompt | self.cypher_llm | StrOutputParser()
        raw = chain.invoke({"schema": self._schema(), "question": question})
        return quote_identifiers(strip_fences(raw), self._known_names())

    def check(self, cypher: str) -> None:
        """Refuse a write, then let the planner reject anything malformed.

        ``EXPLAIN`` plans without executing, so an unrunnable query is caught here rather
        than part-way through running.
        """
        if not self.allow_dangerous_requests and is_write_query(cypher):
            raise ValueError(
                "Refusing to run a generated query that writes. Pass "
                "allow_dangerous_requests=True only if the model is trusted to modify "
                "this graph."
            )
        if self.validate_cypher:
            try:
                self.graph.query(f"EXPLAIN (COSTS OFF) {cypher}", timeout=self.timeout)
            except Exception as exc:
                raise ValueError(f"Generated Cypher is not runnable: {exc}") from exc

    def run_cypher(self, cypher: str) -> List[Dict[str, Any]]:
        return self.graph.query(cypher, timeout=self.timeout)[: self.top_k]

    def answer(self, question: str, rows: List[Dict[str, Any]]) -> str:
        chain = self.qa_prompt | self.qa_llm | StrOutputParser()
        return chain.invoke({"question": question, "results": str(rows)})

    # ---- Runnable ----

    def invoke(
        self,
        input: Dict[str, Any],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        question = input["query"] if "query" in input else input["question"]
        cypher = self.generate_cypher(question)
        self.check(cypher)
        rows = self.run_cypher(cypher)
        out: Dict[str, Any] = {
            "query": question,
            "result": self.answer(question, rows),
        }
        if self.return_intermediate_steps:
            out["intermediate_steps"] = [{"query": cypher}, {"context": rows}]
        return out

    async def ainvoke(
        self,
        input: Dict[str, Any],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        question = input["query"] if "query" in input else input["question"]
        chain = self.cypher_prompt | self.cypher_llm | StrOutputParser()
        raw = await chain.ainvoke({"schema": self._schema(), "question": question})
        cypher = quote_identifiers(strip_fences(raw), self._known_names())
        if not self.allow_dangerous_requests and is_write_query(cypher):
            raise ValueError(
                "Refusing to run a generated query that writes. Pass "
                "allow_dangerous_requests=True only if the model is trusted to modify "
                "this graph."
            )
        if self.validate_cypher:
            try:
                await self.graph.aquery(
                    f"EXPLAIN (COSTS OFF) {cypher}", timeout=self.timeout
                )
            except Exception as exc:
                raise ValueError(f"Generated Cypher is not runnable: {exc}") from exc
        rows = (await self.graph.aquery(cypher, timeout=self.timeout))[: self.top_k]
        qa = self.qa_prompt | self.qa_llm | StrOutputParser()
        out: Dict[str, Any] = {
            "query": question,
            "result": await qa.ainvoke({"question": question, "results": str(rows)}),
        }
        if self.return_intermediate_steps:
            out["intermediate_steps"] = [{"query": cypher}, {"context": rows}]
        return out


def create_cypher_tool(
    graph: AgensGraph,
    llm: BaseLanguageModel,
    *,
    name: str = "query_graph",
    description: Optional[str] = None,
    answer: bool = True,
    **kwargs: Any,
) -> StructuredTool:
    """A tool that answers a question from the graph, for use with an agent.

    With ``answer=False`` the tool returns the rows instead of prose, leaving the agent
    to interpret them — useful when it is combining the graph with other sources.
    """
    chain = AgensCypherQAChain.from_llm(llm, graph=graph, **kwargs)

    def _run(question: str) -> Any:
        if answer:
            return chain.invoke({"query": question})["result"]
        cypher = chain.generate_cypher(question)
        chain.check(cypher)
        return chain.run_cypher(cypher)

    async def _arun(question: str) -> Any:
        if answer:
            return (await chain.ainvoke({"query": question}))["result"]
        return _run(question)

    return StructuredTool.from_function(
        func=_run,
        coroutine=_arun,
        name=name,
        description=description
        or (
            "Answer a question about the data in the graph. Give the question in plain "
            "language; it is turned into a read-only graph query."
        ),
        args_schema=_Question,
    )


__all__: List[str] = [
    "AgensCypherQAChain",
    "CYPHER_SYSTEM",
    "QA_SYSTEM",
    "create_cypher_tool",
    "case_sensitive_names",
    "is_write_query",
    "quote_identifiers",
    "render_cypher_system",
    "strip_fences",
]
