"""Answer a natural-language question by generating and running Cypher.

Two entry points over the same pipeline:

* :class:`AgensCypherQAChain` — ``from_llm(...)`` then ``invoke({"query": ...})``,
  the shape a graph question-answering chain conventionally has.
* :func:`create_cypher_tool` — the same thing as a tool an agent can call.

The pipeline reads the graph's schema, asks the model for a query, repairs and checks
it, runs it read-only, and asks the model to answer from the rows it got back.

Two things about AgensGraph shape the prompt more than anything else. Unquoted
identifiers fold to lower case, so ``(n:Person)`` matches nothing and every label has to
be written ``(n:"Person")``. And AgensGraph is not Neo4j: constructs a model reaches for
out of habit — pattern expressions, ``COUNT { }``, ``EXISTS { }``, ``apoc.*`` — are not
available, so the prompt rules them out by name rather than leaving the model to
discover it.
"""

from __future__ import annotations

import contextvars
import re
import time
from contextlib import asynccontextmanager, contextmanager
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import agensgraph
import psycopg
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from langchain_agensgraph.graphs.agensgraph import AgensGraph

_DEADLINE: contextvars.ContextVar[Mapping[int, float]] = contextvars.ContextVar(
    "agens_qa_deadline", default={}
)
"""When each chain's question in hand runs out of time.

Per question rather than per chain, because one chain answers many at once: a chain is
built once and handed to an agent, or shared across the conversations of a served
application, and a deadline held on the chain would be overwritten by whichever question
started last.

Keyed by chain so that two chains in one request do not read each other's, and a context
variable so that it is per thread and per task at once -- a task inherits a copy of the
context, so an awaited question carries its own.
"""

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

Indexes — the schema lists them; write predicates they can serve:
- Compare the raw property. A cast (n.x::int4), arithmetic (n.x + 1) or coalesce()
  around an indexed property forces a full label read; jsonb already compares
  numbers numerically, so n.age > 30 is right when age holds numbers.
- Match the stored type exactly: a number property compares to 30, not '30' — the
  wrong type returns zero rows without an error.
- Prefix match as a half-open range: n.name >= 'Al' AND n.name < 'Am'.
  STARTS WITH, CONTAINS, ENDS WITH and =~ never use an index.
- Case-insensitive equality: tolower(n.prop) = 'x', when the schema lists an index
  ON ((tolower(prop))).
- Avoid <>: enumerate the values you DO want with IN [...].
- ORDER BY an indexed property with LIMIT is served by the index, and
  id(n) = 'N.M' is the fastest lookup of all.

Hard rules:
- Read-only: MATCH / OPTIONAL MATCH / WHERE / WITH / RETURN / ORDER BY / SKIP / LIMIT.
  Never CREATE / MERGE / SET / DELETE / REMOVE / DROP / DETACH / LOAD.
- Use only the labels, relationship types and properties shown in the schema.
- Always end with a LIMIT of at most {top_k}.
- Return only the query. No prose, no markdown fences, no trailing semicolon.

Examples:
  Q: How many of each kind of node are there? MATCH (n) RETURN label(n) AS label,
  count(*) AS n ORDER BY n DESC LIMIT {top_k}

  Q: Who does Alice work with? MATCH (a:"Person")-[r]->(b:"Person") WHERE a.name =
  'Alice' RETURN b.name AS name, type(r) AS relationship LIMIT {top_k}\
"""

QA_SYSTEM = """\
Answer the question using only the query results provided. If they are empty, say the
graph does not contain that information. Do not invent detail the results do not show.
Be concise.\
"""

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


def is_write_query(query: str) -> bool:
    """Does the query hold a write clause?

    The driver reads it, blanking what the server's lexer does not read as syntax --
    strings, quoted identifiers, dollar-quoted bodies, ``--`` and ``/* */`` comments --
    so a property named ``set`` is a name and a string saying ``delete`` is a string. On
    this server ``//`` does not begin a comment, so nothing hides behind one.

    A first opinion, not the boundary; see :meth:`AgensCypherQAChain.check`.
    """
    try:
        agensgraph.cypher.check_can_wrap(query)
    except ValueError:
        return True
    return False


def _not_runnable(exc: Exception) -> Exception:
    """Say what went wrong with a statement, or say what actually went wrong.

    Only the server refusing the statement means the model wrote something unrunnable. A
    pool with no connection to give, a server that is not there, a role the boundary
    cannot be opened for -- none of those are about the Cypher, and reporting them as a
    query problem sends whoever reads it to look at the query.
    """
    cause = exc
    while cause.__cause__ is not None and cause.__cause__ is not cause:
        cause = cause.__cause__  # type: ignore[assignment]
    if isinstance(cause, (psycopg.ProgrammingError, psycopg.DataError)):
        return ValueError(f"Generated Cypher is not runnable: {exc}")
    return exc


def case_sensitive_names(structured_schema: Dict[str, Any]) -> set:
    """Identifiers in the graph whose spelling is not all lower case.

    Folding runs both ways: a label or key written unquoted is *stored* folded, so
    ``n.firstName`` is right for data created as ``{"firstName": ...}`` and wrong for
    data created as ``{firstName: ...}``, which is stored as ``firstname``. Only the
    schema knows which exists, so it decides what may be quoted.
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
    nothing, which is the failure this is meant to prevent. Text inside string literals
    is left alone. With no ``known`` set, nothing is quoted.
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


GQL_RULE = (
    "- No COUNT { ... }, no EXISTS { ... }, no CALL { ... } subqueries, no apoc.*.\n"
)
"""The line that forbids the GQL clause set, dropped on a server that has it."""


def _example_text(text: str) -> str:
    """A few-shot text made safe for a prompt template: braces doubled."""
    return text.replace("{", "{{").replace("}", "}}")


DIALECT_EXAMPLES: Tuple[Tuple[str, str], ...] = (
    (
        "How many Product nodes are there?",
        'MATCH (p:"Product") RETURN count(*) AS n LIMIT 1',
    ),
    (
        "Who works at Acme?",
        'MATCH (p:"Person")-[:"WORKS_AT"]->(c:"Company") '
        "WHERE c.name = 'Acme' RETURN p.name AS name LIMIT 10",
    ),
    (
        "Which person is named exactly 'Ada'?",
        "MATCH (p:\"Person\") WHERE p.name = 'Ada' RETURN p.name AS name LIMIT 10",
    ),
    (
        "Which titles contain the word graph?",
        'MATCH (b:"Book") WHERE b.title CONTAINS \'graph\' '
        "RETURN b.title AS title LIMIT 10",
    ),
    (
        "Who is older than 30?",
        'MATCH (p:"Person") WHERE p.age > 30 '
        "RETURN p.name AS name, p.age AS age ORDER BY age DESC LIMIT 10",
    ),
    (
        "Return the vertex whose graph id is 3.1.",
        "MATCH (n) WHERE id(n) = 3.1 RETURN n LIMIT 1",
    ),
    (
        "Which people are friends, in either direction?",
        'MATCH (a:"Person")-[:"FRIENDS_WITH"]-(b:"Person") '
        "RETURN a.name AS a, b.name AS b LIMIT 10",
    ),
    (
        "Which authors wrote more than five papers?",
        'MATCH (a:"Author")<-[:"AUTHORED_BY"]-(p:"Paper") '
        "WITH a, count(*) AS papers WHERE papers > 5 "
        "RETURN a.name AS name, papers ORDER BY papers DESC LIMIT 10",
    ),
    (
        "Which customers placed at least one order?",
        'MATCH (c:"Customer") WHERE EXISTS((c)-[:"ORDERED"]->()) '
        "RETURN c.name AS name LIMIT 10",
    ),
    (
        "Return the first three characters of every name.",
        'MATCH (p:"Person") RETURN substring(p.name, 0, 3) AS prefix LIMIT 10',
    ),
    (
        "How many distinct genres does each movie have?",
        'MATCH (m:"Movie")-[:"IN_GENRE"]->(g:"Genre") '
        "RETURN m.title AS title, count(DISTINCT g.name) AS genres "
        "ORDER BY genres DESC LIMIT 10",
    ),
    (
        "Double every item's price and keep the ones over 100.",
        'MATCH (i:"Item") LET doubled = i.price * 2 FILTER doubled > 100 '
        "RETURN i.name AS name, doubled LIMIT 10",
    ),
    (
        "Which products have a name starting with Pro?",
        "MATCH (p:\"Product\") WHERE p.name >= 'Pro' AND p.name < 'Prp' "
        "RETURN p.name AS name LIMIT 10",
    ),
    (
        "Which people are named Ada or Bob?",
        "MATCH (p:\"Person\") WHERE p.name IN ['Ada', 'Bob'] "
        "RETURN p.name AS name LIMIT 10",
    ),
    (
        "Find the person named ada, whatever the letter case.",
        "MATCH (p:\"Person\") WHERE tolower(p.name) = 'ada' "
        "RETURN p.name AS name LIMIT 10",
    ),
    (
        "The five cheapest items with their prices.",
        'MATCH (i:"Item") RETURN i.name AS name, i.price AS price '
        "ORDER BY price LIMIT 5",
    ),
    (
        "Which codes are exactly three digits? Use a pattern.",
        "MATCH (c:\"Code\") WHERE c.value =~ '^[0-9]{3}$' "
        "RETURN c.value AS value LIMIT 10",
    ),
)
"""Question-and-query pairs that teach the dialect's sharpest edges.

Each pair exists because a model reaching for its habits gets that case wrong
here — wrong rows, or right rows off a full label read. Unquoted labels fold to
lower case; ``count()`` takes an argument or ``*``; a graph id is
``labid.locid``; ``substring`` starts at zero. And the index-serving spellings:
compare the raw property (a cast or arithmetic around it reads the whole
label), write a prefix match as a half-open range, use ``tolower(prop) =``
against its expression index for case-insensitive lookups, enumerate values
with ``IN``, and let ``ORDER BY prop LIMIT k`` ride the index. A regex is for
questions that genuinely need a pattern; ``CONTAINS`` is correct dialect for
infix matching and has no index-served form at all. The LET/FILTER pair needs
a 2.18 server.

Pass to :class:`AgensCypherQAChain` or ``AgensText2CypherRetriever`` as
``examples=DIALECT_EXAMPLES``.
"""


def render_cypher_system(
    template: str, top_k: int, graph: Optional[AgensGraph] = None
) -> str:
    """Fill in the row limit and escape the braces that are part of the prose.

    The dialect rules name constructs like ``COUNT { ... }``, whose braces a prompt
    template would otherwise read as variables.

    The rule forbidding the GQL clause set is dropped where the server understands it.
    Telling a model a construct is unavailable when it is available costs a worse query
    for no reason, and the version arrives with the connection, so asking is free.
    """
    filled = template.replace("{top_k}", str(top_k))
    if graph is not None:
        try:
            if graph.capabilities.has_gql_clauses():
                filled = filled.replace(GQL_RULE, "")
        except Exception:
            # A server that will not say keeps the conservative wording.
            pass
    return filled.replace("{", "{{").replace("}", "}}")


class _Question(BaseModel):
    question: str = Field(description="A question to answer from the graph.")


class AgensCypherQAChain(Runnable[Dict[str, Any], Dict[str, Any]]):
    """Question answering over an AgensGraph graph by generated Cypher.

    Build with :meth:`from_llm` and call ``invoke({"query": "..."})``. The result
    carries ``query`` and ``result``; ask for ``return_intermediate_steps`` to also get
    the generated Cypher and the rows it returned.

    Generated Cypher is not trusted: writes are refused, the query is checked with
    ``EXPLAIN`` before it runs, and execution carries a timeout.

    A read-only transaction is what refuses the write, and it is not one for a role that
    may run a command on the server's host through ``COPY ... TO PROGRAM`` -- a superuser,
    or a member of ``pg_execute_server_program``. Connecting as such a role is refused
    rather than quietly given a boundary that does not hold; ``allow_server_programs=True``
    accepts it, and is for a deployment that has decided the role is acceptable.
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
        allow_server_programs: bool = False,
        cypher_prompt: Optional[ChatPromptTemplate] = None,
        qa_prompt: Optional[ChatPromptTemplate] = None,
        examples: Optional[Sequence[Tuple[str, str]]] = None,
        return_intermediate_steps: bool = False,
    ) -> None:
        if cypher_prompt is not None and examples:
            raise ValueError(
                "Pass examples or a custom cypher_prompt, not both; a custom "
                "prompt carries its own examples as messages."
            )
        self.graph = graph
        self.cypher_llm = cypher_llm
        self.qa_llm = qa_llm
        self.top_k = top_k
        self.timeout = timeout
        self.validate_cypher = validate_cypher
        self.allow_dangerous_requests = allow_dangerous_requests
        self.allow_server_programs = allow_server_programs
        self.return_intermediate_steps = return_intermediate_steps
        self.examples = tuple(examples or ())
        # Few-shot pairs become real conversation turns, not prompt text: the
        # model sees each question answered with the query alone, which is the
        # shape it is being asked to produce. The texts are escaped because
        # from_messages reads every string as a template, and a Cypher map
        # literal's braces would otherwise be taken for variables.
        shots: List[Any] = []
        for shown_question, shown_cypher in self.examples:
            shots.append(("human", _example_text(shown_question)))
            shots.append(("ai", _example_text(shown_cypher)))
        self.cypher_prompt = cypher_prompt or ChatPromptTemplate.from_messages(
            [
                ("system", render_cypher_system(CYPHER_SYSTEM, top_k, graph=graph)),
                *shots,
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

    @contextmanager
    def _budget(self) -> Iterator[None]:
        """One time budget for a question, spent across every statement it takes.

        `EXPLAIN` and the statement itself are two round trips, and giving each the whole
        timeout means a caller who asked for thirty seconds can wait sixty. A timeout that
        starts again at each round trip is not one anybody can hold this to.
        """
        if self.timeout is None:
            yield
            return
        held = _DEADLINE.get()
        token = _DEADLINE.set(
            {**held, id(self): time.monotonic() + self.timeout}
        )
        try:
            yield
        finally:
            _DEADLINE.reset(token)

    @property
    def _remaining(self) -> Optional[float]:
        """What is left of the budget, or the whole timeout outside one."""
        deadline = _DEADLINE.get().get(id(self))
        if deadline is None:
            return self.timeout
        left = deadline - time.monotonic()
        # Not zero or below: the server reads that as no timeout at all, which is the
        # opposite of what a spent budget means.
        return max(left, 0.001)

    def check(self, cypher: str) -> None:
        """Refuse a write, then let the planner reject anything malformed.

        The reading of the text is a fast first opinion and **not** the boundary. It
        cannot be one: this is PostgreSQL underneath, so ``INSERT``, ``TRUNCATE`` and
        ``COPY ... TO PROGRAM`` are all available and none of them is Cypher, so none of
        them matches. What actually stops a write is :meth:`run_cypher` running the
        statement inside a transaction the server will not let write.

        ``EXPLAIN`` plans without executing, so an unrunnable query is caught here
        rather than part-way through running.
        """
        if not self.allow_dangerous_requests and is_write_query(cypher):
            raise ValueError(
                "Refusing to run a generated query that writes. Pass "
                "allow_dangerous_requests=True only if the model is trusted to modify "
                "this graph."
            )
        if self.validate_cypher:
            with self._boundary():
                try:
                    self.graph.query(
                        f"EXPLAIN (COSTS OFF) {cypher}", timeout=self._remaining
                    )
                except Exception as exc:
                    raise _not_runnable(exc) from exc

    @contextmanager
    def _boundary(self) -> Iterator[None]:
        """The transaction a generated statement runs inside.

        Both the ``EXPLAIN`` and the statement itself run in one, because planning is not
        the only thing an ``EXPLAIN`` can cause: a statement carrying no parameters goes
        to the server over the simple query protocol, which runs every statement in it,
        and ``strip_fences`` removes only a trailing semicolon.

        With ``allow_dangerous_requests`` the model is trusted to modify the graph, so
        there is nothing to enforce and the statement runs ordinarily. Opening one anyway
        would refuse the write the caller just permitted.
        """
        if self.allow_dangerous_requests:
            yield
            return
        with self.graph.read_only(allow_server_programs=self.allow_server_programs):
            yield

    @asynccontextmanager
    async def _aboundary(self) -> AsyncIterator[None]:
        """Async sibling of :meth:`_boundary`."""
        if self.allow_dangerous_requests:
            yield
            return
        async with self.graph.aread_only(
            allow_server_programs=self.allow_server_programs
        ):
            yield

    def run_cypher(self, cypher: str) -> List[Dict[str, Any]]:
        """Run the generated statement, inside a boundary the server enforces.

        A read-only transaction refuses a write however the statement is spelled, so the
        chain holds no opinion about what writing looks like -- which is the only way to
        be right about it, because every reading of the text misses.

        Every row comes back and ``top_k`` is taken from them here. Cutting at the server
        would mean appending a limit to a statement somebody else wrote, which changes what
        it means whenever it already has one or ends in anything that a limit cannot follow.
        What bounds this instead is the timeout, which is why one is applied: a statement
        that would return more than a chain can hold does not get the time to.
        """
        with self._boundary():
            return self.graph.query(cypher, timeout=self._remaining)[: self.top_k]

    @staticmethod
    def _for_prompt(rows: List[Dict[str, Any]]) -> str:
        """The rows as JSON, for the model to read.

        A vertex arrives as its label, its identity and its properties.
        """
        return agensgraph.to_json(rows).decode()

    def answer(self, question: str, rows: List[Dict[str, Any]]) -> str:
        chain = self.qa_prompt | self.qa_llm | StrOutputParser()
        return chain.invoke({"question": question, "results": self._for_prompt(rows)})

    # ---- Runnable ----

    def invoke(
        self,
        input: Dict[str, Any],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        question = input["query"] if "query" in input else input["question"]
        cypher = self.generate_cypher(question)
        with self._budget():
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
        """The same pipeline as :meth:`invoke`, awaited."""
        question = input["query"] if "query" in input else input["question"]
        chain = self.cypher_prompt | self.cypher_llm | StrOutputParser()
        raw = await chain.ainvoke(
            {"schema": self._schema(), "question": question}, config=config
        )
        cypher = quote_identifiers(strip_fences(raw), self._known_names())
        with self._budget():
            await self.acheck(cypher)
            rows = await self.arun_cypher(cypher)
        qa = self.qa_prompt | self.qa_llm | StrOutputParser()
        out: Dict[str, Any] = {
            "query": question,
            "result": await qa.ainvoke(
                {"question": question, "results": self._for_prompt(rows)}, config=config
            ),
        }
        if self.return_intermediate_steps:
            out["intermediate_steps"] = [{"query": cypher}, {"context": rows}]
        return out

    async def acheck(self, cypher: str) -> None:
        """Async sibling of :meth:`check`."""
        if not self.allow_dangerous_requests and is_write_query(cypher):
            raise ValueError(
                "Refusing to run a generated query that writes. Pass "
                "allow_dangerous_requests=True only if the model is trusted to modify "
                "this graph."
            )
        if self.validate_cypher:
            async with self._aboundary():
                try:
                    await self.graph.aquery(
                        f"EXPLAIN (COSTS OFF) {cypher}", timeout=self._remaining
                    )
                except Exception as exc:
                    raise _not_runnable(exc) from exc

    async def arun_cypher(self, cypher: str) -> List[Dict[str, Any]]:
        """Async sibling of :meth:`run_cypher`."""
        async with self._aboundary():
            rows = await self.graph.aquery(cypher, timeout=self._remaining)
            return rows[: self.top_k]


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
        cypher = chain.generate_cypher(question)
        await chain.acheck(cypher)
        return await chain.arun_cypher(cypher)

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
    "DIALECT_EXAMPLES",
    "QA_SYSTEM",
    "create_cypher_tool",
    "case_sensitive_names",
    "is_write_query",
    "quote_identifiers",
    "render_cypher_system",
    "strip_fences",
]
