"""Answer a question from the graph with Cypher a model writes.

:class:`AgensCypherQAChain` asks a model for one Cypher statement, runs it with
:meth:`~langchain_agensgraph.AgensGraph.query_untrusted`, and asks a model to answer from the rows;
:func:`create_cypher_tool` is the same as a tool an agent can call.

Before the statement is sent it is taken out of any fences, the names the graph stores with
capitals are quoted, an edge written against its schema direction is turned round
(``validate_cypher``), and :func:`~langchain_agensgraph.chains.prepass.prepare` makes the
rewrites the server needs. ``query_untrusted`` then reads it again and runs it as a role made for
model-written Cypher, in a transaction that is read-only and rolled back, reading at most
``top_k`` rows. A refusal at any step, or a server error about the statement, goes back to the
model with the statement it wrote, up to ``max_retries`` times.

The prompt is composed from the server's capabilities: the rules for every release, then those
only the releases in use need.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal, cast

from agensgraph import Capabilities, to_json
from agensgraph.untrusted import ALLOWED_FUNCTIONS
from langchain_core.callbacks import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    BasePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.runnables import Runnable, RunnableConfig, RunnableSerializable
from langchain_core.runnables.config import patch_config
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator
from typing_extensions import Self

from ..errors import AgensQueryError, UnsafeCypherError
from ..graphs.agensgraph import AgensGraph, _rendered
from .prepass import (
    case_sensitive_names,
    correct_directions,
    prepare,
    quote_identifiers,
    strip_fences,
)

_ALL_RELEASES = """\
AgensGraph dialect; these differ from Neo4j and matter:
- Quote every label and relationship type: (n:"Person"), (a)-[r:"WORKS_AT"]->(b). Unquoted names \
fold to lower case and match nothing. No backticks.
- Quote mixed-case property names the same way: n."firstName". All-lowercase names need no \
quotes: n.name.
- Strings take single quotes only: n.name = 'Ada' ("Ada" is a name, not a string). Escape a quote \
by doubling it: 'O''Brien'.
- Always write relationships with brackets: (a)-[]->(b), (a)<-[]-(b), (a)-[]-(b). Never -->, <-- \
or --.
- One label per node and one type per relationship: no (n:"A":"B"), (n:"A"|"B") or \
[:"R1"|"R2"]; write WHERE label(n) IN ['A', 'B'] or WHERE type(r) IN ['R1', 'R2']. In WHERE use \
label(n) = 'A', not n:"A".
- A pattern used as a condition goes inside exists(): WHERE exists((a)-[:"R"]->()), WHERE NOT \
exists(...). No pattern comprehensions, no map projections (write {name: n.name}), no reduce(), \
FOREACH or elementId() (use id(n)).
- MATCH cannot follow OPTIONAL MATCH directly: put the MATCH first or a WITH between them.
- Name every variable-length relationship and bound it, at most 5 steps: \
(a)-[r:"KNOWS"*1..3]->(b). Inside shortestpath() and allshortestpaths() leave it unnamed: \
MATCH p = shortestpath((a)-[:"KNOWS"*1..5]->(b)).
- Convert with casts, not toInteger() or toFloat(): n.x::int8, n.x::float8, n.x::text, \
n.x::bool, n.x::date. A decimal must go to float8 (int8 refuses 9.5). No CAST(x AS t).
- left(), right() and reverse() on a string need ::text: left(n.name::text, 2).
- Do not name a path variable path: MATCH p = ...
- Use count(*) rather than count(n) over a node variable: count(n) reads every property of every \
node, embeddings included, and is far slower.
- Prefer an untyped relationship (a)-[r]->(b) and read type(r) unless the exact relationship type \
appears in the schema."""

_BEFORE_GQL = """\
- No GQL: ORDER BY, SKIP and LIMIT only after WITH or RETURN; no LET, FILTER, NEXT, FINISH, \
OFFSET (use SKIP), COUNT { }, EXISTS { }, CALL { }, XOR, IS [NOT] UNKNOWN, nullif(), count(ALL x) \
or current_date.
- An aggregate other than count(*) must be a whole RETURN or WITH item. Aggregate in a WITH, \
compute in the next clause: WITH avg(n.age) AS a RETURN round(a) AS a. Never round(avg(..)), \
sum(..) / count(*), max(..) - min(..), collect(..)[0..3], length(collect(..)), \
head(collect(..)), CASE WHEN avg(..) > ...
- min() and max() over a property must cast it: min(n.age::int4), max(n.score::float8), \
min(n.name::text).
- length() for lists and strings, never size(list); n.prop IS NOT NULL, never exists(n.prop); \
last(nodes(p)) or last(r) for the last node or relationship (nodes(p)[-1] is null).
- Cast only to int4, int8, float8, text, bool, date, timestamptz (int, integer, bigint, float, \
numeric, boolean and varchar are syntax errors).
- id(n) IN [...] fails: write id(n) = '3.1' OR id(n) = '3.2'. UNWIND a list literal, not \
range(...).
- [x IN list WHERE x IS NOT NULL] does not remove nulls."""

_TEXT_CAST_SORT = """\
- Do not ORDER BY, DISTINCT or group on a ::text cast inside WITH; use the property itself."""

_PROJECTED_ORDER = """\
- ORDER BY may name only what the RETURN or WITH it follows outputs, by alias: RETURN n.name AS \
name ORDER BY name. Alias aggregates and sort by the alias: count(*) AS n ... ORDER BY n. To sort \
by something you do not return, sort in a WITH first: WITH n ORDER BY n.age DESC LIMIT 10 RETURN \
n.name AS name. To sort nodes, sort by id(n) in a WITH."""

_COLLECT_NULLS = """\
- collect() gives null over no rows and keeps nulls: filter WHERE x.p IS NOT NULL before \
collecting; after OPTIONAL MATCH write WITH k, collect(x.p) AS xs, count(x) AS c RETURN k, CASE \
WHEN c = 0 THEN [] ELSE xs END; where no row may match, WITH collect(..) AS xs RETURN \
coalesce(xs, [])."""

_MISSING_LABELS = """\
- A label that is not in the schema is an error. Do not repeat a label on a variable bound \
earlier: MATCH (a:"P") ... MATCH (a)-[]->(b), not MATCH (a:"P")-[]->(b). Do not reuse a \
relationship variable in a second MATCH. Do not name a variable _."""

_RELEASE_RULES: tuple[tuple[str, str], ...] = (
    ("gql_clauses", _BEFORE_GQL),
    ("text_cast_sort_in_with", _TEXT_CAST_SORT),
    ("order_by_unprojected", _PROJECTED_ORDER),
    ("collect_drops_nulls", _COLLECT_NULLS),
    ("missing_label_read_empty", _MISSING_LABELS),
)
"""Each rule, and the capability whose absence makes it needed."""

_INDEXES = """\
Indexes (the schema lists them); write conditions they can serve:
- Compare the raw property. A cast (n.x::int4), arithmetic (n.x + 1) or coalesce() around an \
indexed property reads the whole label; numbers compare as numbers, so n.age > 30 is right when \
age holds numbers.
- Match the stored type exactly: a number property compares to 30, not '30'; the wrong type \
returns no rows and no error.
- Prefix match as a half-open range: n.name >= 'Al' AND n.name < 'Am'. STARTS WITH, CONTAINS, \
ENDS WITH and =~ never use an index.
- Case-insensitive equality: tolower(n.prop) = 'x', when the schema lists an index ON \
((tolower(prop))).
- Avoid <>: list the values you want with IN [...].
- An indexed property sorted under a LIMIT is read from the index: RETURN n.prop AS prop ORDER BY \
prop LIMIT 10. id(n) = '3.1' is the fastest lookup of all."""

_READS = "- Read only: never CREATE, MERGE, SET, DELETE, REMOVE or DETACH."
_WRITES = (
    "- Write with CREATE, MERGE, SET, DELETE or REMOVE only when the question asks for a change; "
    "otherwise read."
)

_RULES = """\
Rules for the statement:
{access}
- One statement, starting with {starts}. No SQL (SELECT, FROM, JOIN, UNION), no CALL, no \
comments, no parameters.
- Only these functions: {functions}.
- Use only the labels, relationship types and properties the schema shows.
- End with a LIMIT of at most {top_k}.
- Return only the statement: no prose, no markdown fences, no semicolon.

Examples:
  Q: How many of each kind of node are there?
  MATCH (n) RETURN label(n) AS kind, count(*) AS cnt ORDER BY cnt DESC LIMIT {top_k}
  Q: Who does Alice work with?
  MATCH (a:"Person")-[r]->(b:"Person") WHERE a.name = 'Alice' RETURN b.name AS name, type(r) AS \
relationship LIMIT {top_k}"""

_NOT_OFFERED = frozenset({"tointeger", "tofloat"})
_SQL_FUNCTIONS = frozenset({"range", "split"})


def render_cypher_system(
    capabilities: Capabilities, *, top_k: int = 10, allow_writes: bool = False
) -> str:
    """The system prompt for writing Cypher for a server with these capabilities."""
    rules = [_ALL_RELEASES]
    rules += [text for name, text in _RELEASE_RULES if not capabilities.has(name)]
    functions = ALLOWED_FUNCTIONS - _NOT_OFFERED
    if not capabilities.has("cypher_sql_functions"):
        functions -= _SQL_FUNCTIONS
    starts = "MATCH, OPTIONAL MATCH, UNWIND or RETURN"
    if allow_writes:
        starts = "MATCH, OPTIONAL MATCH, UNWIND, RETURN, CREATE or MERGE"
    header = "Write one Cypher statement for AgensGraph that answers the user's question" + (
        ", or makes the change it asks for." if allow_writes else "."
    )
    return "\n\n".join(
        [
            header,
            "\n".join(rules),
            _INDEXES,
            _RULES.format(
                access=_WRITES if allow_writes else _READS,
                starts=starts,
                functions=", ".join(sorted(functions)),
                top_k=top_k,
            ),
        ]
    )


QA_SYSTEM = """\
Answer the question using only the query results provided. If they are empty, say the graph does \
not contain that information. Do not invent detail the results do not show. Be concise."""

FUNCTION_RESPONSE_SYSTEM = """\
Answer the question using only what the tool returned. If it returned nothing, say the graph does \
not contain that information. Do not invent detail it does not show. Be concise."""

CYPHER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "{system}"),
        MessagesPlaceholder("examples", optional=True),
        ("human", "Schema:\n{schema}\n\nQuestion: {question}"),
    ]
)
"""The prompt for writing the statement: ``system`` is :func:`render_cypher_system`'s text,
``examples`` the few-shot turns, then ``schema`` and ``question``."""

QA_PROMPT = ChatPromptTemplate.from_messages(
    [("system", QA_SYSTEM), ("human", "Question: {question}\n\nResults:\n{context}")]
)
"""The prompt for answering: ``question``, and ``context``, the rows as JSON."""

_CORRECTION = (
    "Running that statement failed: {error}\n\nWrite a corrected statement that answers the same "
    "question. Return only the statement."
)
_EMPTY_FEEDBACK = (
    "That statement ran without error and returned no rows. Check each relationship's direction "
    "and each label's spelling against the schema; the schema's directions are authoritative."
)
_DANGER = (
    "AgensCypherQAChain runs Cypher a model wrote. It runs it as the engine's untrusted role "
    "(untrusted_conninfo), in a transaction that is read-only and rolled back, after reading it "
    "for anything that is not plainly one Cypher read; that role decides what the model can read. "
    "Pass allow_dangerous_requests=True to acknowledge this."
)

DIALECT_EXAMPLES: tuple[tuple[str, str], ...] = (
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
        "MATCH (b:\"Book\") WHERE b.title CONTAINS 'graph' RETURN b.title AS title LIMIT 10",
    ),
    (
        "Who is older than 30?",
        'MATCH (p:"Person") WHERE p.age > 30 '
        "RETURN p.name AS name, p.age AS age ORDER BY age DESC LIMIT 10",
    ),
    (
        "Return the vertex whose graph id is 3.1.",
        "MATCH (n) WHERE id(n) = '3.1' RETURN n LIMIT 1",
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
        'MATCH (c:"Customer") WHERE exists((c)-[:"ORDERED"]->()) RETURN c.name AS name LIMIT 10',
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
        "MATCH (p:\"Person\") WHERE p.name IN ['Ada', 'Bob'] RETURN p.name AS name LIMIT 10",
    ),
    (
        "Find the person named ada, whatever the letter case.",
        "MATCH (p:\"Person\") WHERE tolower(p.name) = 'ada' RETURN p.name AS name LIMIT 10",
    ),
    (
        "The five cheapest items with their prices.",
        'MATCH (i:"Item") RETURN i.name AS name, i.price AS price ORDER BY price LIMIT 5',
    ),
    (
        "Which codes are exactly three digits? Use a pattern.",
        "MATCH (c:\"Code\") WHERE c.value =~ '^[0-9]{3}$' RETURN c.value AS value LIMIT 10",
    ),
)
"""Question-and-statement pairs that teach the dialect's sharpest edges.

Each pair is a case a model reaching for its habits gets wrong here, with wrong rows or with right
rows off a full label read: unquoted labels fold to lower case; ``count()`` takes an argument or
``*``; a graph id is ``'labid.locid'``; ``substring`` starts at zero; a condition on a pattern goes
in ``exists()``. And the spellings an index serves: compare the raw property, write a prefix match
as a half-open range, use ``tolower(prop) =`` against its expression index, list values with
``IN``, and let ``ORDER BY prop LIMIT k`` ride the index.

Pass as ``examples=DIALECT_EXAMPLES``; a pair the server cannot run is left out of the prompt.
"""

_EXAMPLE_NEEDS = {DIALECT_EXAMPLES[11][1]: "gql_clauses"}


def _statement_fault(exc: BaseException) -> bool:
    """Whether a failure is the statement's, so the model may be asked to correct it.

    The pre-pass and the driver's reading refuse with :class:`UnsafeCypherError`; of the server's
    errors, those about the statement: syntax, names, types, data, limits, a write refused, the
    untrusted role's budgets (``statement_timeout``, and ``temp_file_limit`` for a statement that
    sorts or hashes too much), and AgensGraph's internal errors, most of which a statement causes.
    """
    if isinstance(exc, UnsafeCypherError):
        return True
    if isinstance(exc, AgensQueryError):
        state = exc.sqlstate or ""
        return state[:2] in ("0A", "21", "22", "42", "54") or state in (
            "25006",
            "53400",
            "57014",
            "XX000",
        )
    return False


def _text(output: Any) -> str:
    return _PARSER.invoke(output)


_PARSER = StrOutputParser()


class CypherRun(BaseModel):
    """What answering took: the statement that ran, its rows, and each attempt before it."""

    cypher: str
    rows: list[dict[str, Any]]
    attempts: list[dict[str, str]] = Field(default_factory=list)


class AgensCypherQAChain(RunnableSerializable[dict[str, Any] | str, dict[str, Any]]):
    """Question answering over an AgensGraph graph with Cypher a model writes.

    Build it with :meth:`from_llm` and call ``invoke({"query": "..."})``. The result holds
    ``query`` and ``result``, the answer or with ``return_direct`` the rows, and with
    ``return_intermediate_steps`` the ``intermediate_steps``: each failed attempt, the statement
    that ran and the rows it returned.

    The graph's engine needs ``untrusted_conninfo``, a role made for model-written Cypher (see
    :func:`~langchain_agensgraph.roles.render_grants`), and ``writer_conninfo`` for
    ``allow_writes``. ``top_k`` is how many rows are read; ``timeout`` the seconds each statement
    may take. ``schema_refresh="ttl"`` reads the schema again when the graph's cache has expired,
    ``"frozen"`` once. ``include_types`` and ``exclude_types`` limit the labels the prompt shows.
    ``use_function_response`` hands the rows to the answering model as a tool's response.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    graph: AgensGraph
    cypher_llm: Runnable[Any, Any]
    qa_llm: Runnable[Any, Any]
    # Unparametrized: pydantic validates against the class named, and BasePromptTemplate[Any] is
    # a subclass no prompt is an instance of.
    cypher_prompt: BasePromptTemplate | None = None  # type: ignore[type-arg]
    qa_prompt: BasePromptTemplate | None = None  # type: ignore[type-arg]
    examples: tuple[tuple[str, str], ...] = ()
    input_key: str = "query"
    output_key: str = "result"
    top_k: int = Field(default=10, ge=1)
    timeout: float = Field(default=10.0, gt=0)
    max_retries: int = Field(default=2, ge=0)
    allow_dangerous_requests: bool = False
    allow_writes: bool = False
    unsafe_shared_role: bool = False
    validate_cypher: bool = True
    return_direct: bool = False
    return_intermediate_steps: bool = False
    include_types: tuple[str, ...] = ()
    exclude_types: tuple[str, ...] = ()
    schema_refresh: Literal["ttl", "frozen"] = "ttl"
    use_function_response: bool = False

    _frozen: tuple[str, dict[str, Any]] | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _acknowledged(self) -> Self:
        if not self.allow_dangerous_requests:
            raise ValueError(_DANGER)
        if self.include_types and self.exclude_types:
            raise ValueError("pass include_types or exclude_types, not both")
        if self.examples and self.cypher_prompt is not None:
            raise ValueError(
                "pass examples or a cypher_prompt, not both; a prompt of your own carries its "
                "own examples"
            )
        return self

    @classmethod
    def from_llm(
        cls,
        llm: Runnable[Any, Any] | None = None,
        *,
        graph: AgensGraph,
        cypher_llm: Runnable[Any, Any] | None = None,
        qa_llm: Runnable[Any, Any] | None = None,
        cypher_prompt: BasePromptTemplate[Any] | None = None,
        qa_prompt: BasePromptTemplate[Any] | None = None,
        cypher_llm_kwargs: Mapping[str, Any] | None = None,
        qa_llm_kwargs: Mapping[str, Any] | None = None,
        examples: Sequence[tuple[str, str]] | None = None,
        include_types: Sequence[str] | None = None,
        exclude_types: Sequence[str] | None = None,
        schema: Literal["ttl", "frozen"] = "ttl",
        **kwargs: Any,
    ) -> AgensCypherQAChain:
        """Build from one model, or a model for each step.

        ``cypher_llm_kwargs`` and ``qa_llm_kwargs`` may hold a ``prompt``; anything else in them
        is bound to that step's model.
        """
        cypher_kwargs = dict(cypher_llm_kwargs or {})
        qa_kwargs = dict(qa_llm_kwargs or {})
        writer = cypher_llm or llm
        answerer = qa_llm or llm
        if writer is None or answerer is None:
            raise ValueError("pass llm, or both cypher_llm and qa_llm")
        cypher_prompt = cypher_kwargs.pop("prompt", cypher_prompt)
        qa_prompt = qa_kwargs.pop("prompt", qa_prompt)
        if cypher_kwargs:
            writer = writer.bind(**cypher_kwargs)
        if qa_kwargs:
            answerer = answerer.bind(**qa_kwargs)
        return cls(
            graph=graph,
            cypher_llm=writer,
            qa_llm=answerer,
            cypher_prompt=cypher_prompt,
            qa_prompt=qa_prompt,
            examples=tuple(examples or ()),
            include_types=tuple(include_types or ()),
            exclude_types=tuple(exclude_types or ()),
            schema_refresh=schema,
            **kwargs,
        )

    # -- Runnable --------------------------------------------------------------------------------

    def invoke(
        self, input: dict[str, Any] | str, config: RunnableConfig | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        return self._call_with_config(self._call, input, config, run_type="chain", **kwargs)

    async def ainvoke(
        self, input: dict[str, Any] | str, config: RunnableConfig | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        return await self._acall_with_config(self._acall, input, config, run_type="chain", **kwargs)

    def _question(self, input: dict[str, Any] | str) -> str:
        if isinstance(input, str):
            return input
        if self.input_key in input:
            return str(input[self.input_key])
        if "question" in input:
            return str(input["question"])
        raise KeyError(f"the input needs {self.input_key!r}")

    def _call(
        self,
        input: dict[str, Any] | str,
        run_manager: CallbackManagerForChainRun,
        config: RunnableConfig,
    ) -> dict[str, Any]:
        question = self._question(input)
        child = patch_config(config, callbacks=run_manager.get_child())
        run = self.run_cypher(question, config=child)
        if self.return_direct:
            return self._output(question, run.rows, run)
        return self._output(question, self._answer(question, run.rows, child), run)

    async def _acall(
        self,
        input: dict[str, Any] | str,
        run_manager: AsyncCallbackManagerForChainRun,
        config: RunnableConfig,
    ) -> dict[str, Any]:
        question = self._question(input)
        child = patch_config(config, callbacks=run_manager.get_child())
        run = await self.arun_cypher(question, config=child)
        if self.return_direct:
            return self._output(question, run.rows, run)
        return self._output(question, await self._aanswer(question, run.rows, child), run)

    def _output(self, question: str, result: Any, run: CypherRun) -> dict[str, Any]:
        out: dict[str, Any] = {"query": question, self.output_key: result}
        if self.return_intermediate_steps:
            out["intermediate_steps"] = [
                *run.attempts,
                {"query": run.cypher},
                {"context": run.rows},
            ]
        return out

    # -- writing and running the statement -------------------------------------------------------

    def run_cypher(
        self, question: str, *, config: RunnableConfig | None = None, retry_on_empty: bool = False
    ) -> CypherRun:
        """Have the model write a statement for the question and run it, correcting it as the
        refusals and errors it meets come back; ``retry_on_empty`` treats no rows as one."""
        caps = self.graph.capabilities
        schema, structured = self._schema()
        messages = self._messages(question, caps, schema, config)
        attempts: list[dict[str, str]] = []
        while True:
            written = _text(self.cypher_llm.invoke(messages, config=config))
            cypher = written
            outcome: list[dict[str, Any]] | Exception
            try:
                cypher = self._prepared(written, caps, structured)
                outcome = self.graph.query_untrusted(
                    cypher,
                    max_rows=self.top_k,
                    timeout=self.timeout,
                    allow_writes=self.allow_writes,
                    unsafe_shared_role=self.unsafe_shared_role,
                )
            except (UnsafeCypherError, AgensQueryError) as exc:
                outcome = exc
            settled = self._settled(cypher, outcome, attempts, messages, retry_on_empty)
            if isinstance(settled, CypherRun):
                return settled
            messages = settled

    async def arun_cypher(
        self, question: str, *, config: RunnableConfig | None = None, retry_on_empty: bool = False
    ) -> CypherRun:
        caps = await self.graph.acapabilities()
        schema, structured = await self._aschema()
        messages = self._messages(question, caps, schema, config)
        attempts: list[dict[str, str]] = []
        while True:
            written = _text(await self.cypher_llm.ainvoke(messages, config=config))
            cypher = written
            outcome: list[dict[str, Any]] | Exception
            try:
                cypher = self._prepared(written, caps, structured)
                outcome = await self.graph.aquery_untrusted(
                    cypher,
                    max_rows=self.top_k,
                    timeout=self.timeout,
                    allow_writes=self.allow_writes,
                    unsafe_shared_role=self.unsafe_shared_role,
                )
            except (UnsafeCypherError, AgensQueryError) as exc:
                outcome = exc
            settled = self._settled(cypher, outcome, attempts, messages, retry_on_empty)
            if isinstance(settled, CypherRun):
                return settled
            messages = settled

    def _settled(
        self,
        cypher: str,
        outcome: list[dict[str, Any]] | Exception,
        attempts: list[dict[str, str]],
        messages: list[BaseMessage],
        retry_on_empty: bool,
    ) -> CypherRun | list[BaseMessage]:
        """The run, when an attempt settles it; otherwise the conversation asking for another,
        after the attempt is recorded. A failure that is not the statement's is raised."""
        if isinstance(outcome, Exception):
            if not _statement_fault(outcome) or len(attempts) >= self.max_retries:
                raise outcome
            error = str(outcome)[:500]
            feedback = _CORRECTION.format(error=error)
        elif outcome or not retry_on_empty or len(attempts) >= self.max_retries:
            return CypherRun(cypher=cypher, rows=outcome, attempts=attempts)
        else:
            error, feedback = "the statement returned no rows", _EMPTY_FEEDBACK
        attempts.append({"query": cypher, "error": error})
        return [*messages, AIMessage(content=cypher), HumanMessage(content=feedback)]

    def _messages(
        self, question: str, caps: Capabilities, schema: str, config: RunnableConfig | None
    ) -> list[BaseMessage]:
        shots: list[BaseMessage] = []
        for shown, cypher in self.examples:
            needs = _EXAMPLE_NEEDS.get(cypher)
            if needs is None or caps.has(needs):
                shots += [HumanMessage(content=shown), AIMessage(content=cypher)]
        system = render_cypher_system(caps, top_k=self.top_k, allow_writes=self.allow_writes)
        prompt = (self.cypher_prompt or CYPHER_PROMPT).invoke(
            {
                "system": system,
                "examples": shots,
                "schema": schema,
                "question": question,
                "top_k": self.top_k,
            },
            config=config,
        )
        return prompt.to_messages()

    def _prepared(self, written: str, caps: Capabilities, structured: dict[str, Any]) -> str:
        cypher = quote_identifiers(strip_fences(written), case_sensitive_names(structured))
        if self.validate_cypher:
            cypher = correct_directions(cypher, structured.get("relationships") or ())
        return prepare(cypher, caps)

    # -- the schema ------------------------------------------------------------------------------

    def _schema(self) -> tuple[str, dict[str, Any]]:
        if self._frozen is not None:
            return self._frozen
        self.graph.refresh_schema()
        return self._kept(self.graph.schema, self.graph.structured_schema)

    async def _aschema(self) -> tuple[str, dict[str, Any]]:
        if self._frozen is not None:
            return self._frozen
        await self.graph.arefresh_schema()
        return self._kept(self.graph.schema, self.graph.structured_schema)

    def _kept(self, text: str, structured: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        if self.include_types or self.exclude_types:
            structured = _limited(structured, self.include_types, self.exclude_types)
            text = _rendered(structured)
        if self.schema_refresh == "frozen":
            self._frozen = (text, structured)
        return text, structured

    # -- answering -------------------------------------------------------------------------------

    def _answer_messages(self, question: str, rows: list[dict[str, Any]]) -> Any:
        context = to_json(rows).decode()
        if self.use_function_response:
            return [
                SystemMessage(content=FUNCTION_RESPONSE_SYSTEM),
                HumanMessage(content=question),
                AIMessage(
                    content="",
                    tool_calls=[{"name": "query_graph", "args": {}, "id": "query_graph"}],
                ),
                ToolMessage(content=context, tool_call_id="query_graph"),
            ]
        return (self.qa_prompt or QA_PROMPT).invoke({"question": question, "context": context})

    def _answer(self, question: str, rows: list[dict[str, Any]], config: RunnableConfig) -> str:
        return _text(self.qa_llm.invoke(self._answer_messages(question, rows), config=config))

    async def _aanswer(
        self, question: str, rows: list[dict[str, Any]], config: RunnableConfig
    ) -> str:
        messages = self._answer_messages(question, rows)
        return _text(await self.qa_llm.ainvoke(messages, config=config))


def _limited(
    structured: dict[str, Any], include: Sequence[str], exclude: Sequence[str]
) -> dict[str, Any]:
    """The schema with only the labels and types a prompt is to see."""

    def kept(name: str) -> bool:
        return name in include if include else name not in exclude

    relationships = [
        rel
        for rel in structured.get("relationships", [])
        if kept(rel["start"]) and kept(rel["type"]) and kept(rel["end"])
    ]
    return {
        **structured,
        "node_props": {k: v for k, v in structured.get("node_props", {}).items() if kept(k)},
        "rel_props": {k: v for k, v in structured.get("rel_props", {}).items() if kept(k)},
        "relationships": relationships,
        "counts": {k: v for k, v in structured.get("counts", {}).items() if kept(k)},
        "indexes": [i for i in structured.get("indexes", []) if kept(i["label"])],
    }


class _Question(BaseModel):
    question: str = Field(description="A question to answer from the graph, in plain language.")


def create_cypher_tool(
    graph: AgensGraph,
    llm: Runnable[Any, Any],
    *,
    name: str = "query_graph",
    description: str | None = None,
    answer: bool = True,
    allow_dangerous_requests: bool = False,
    allow_writes: bool = False,
    **kwargs: Any,
) -> BaseTool:
    """A tool that answers a question from the graph, for an agent.

    With ``answer=False`` it returns the rows as JSON instead of an answer, for an agent that
    combines the graph with other sources. Other arguments go to
    :meth:`AgensCypherQAChain.from_llm`.
    """
    chain = AgensCypherQAChain.from_llm(
        llm,
        graph=graph,
        allow_dangerous_requests=allow_dangerous_requests,
        allow_writes=allow_writes,
        **kwargs,
    )

    def run(question: str, config: RunnableConfig) -> str:
        if answer:
            return cast(str, chain.invoke({"query": question}, config=config)[chain.output_key])
        return to_json(chain.run_cypher(question, config=config).rows).decode()

    async def arun(question: str, config: RunnableConfig) -> str:
        if answer:
            out = await chain.ainvoke({"query": question}, config=config)
            return cast(str, out[chain.output_key])
        return to_json((await chain.arun_cypher(question, config=config)).rows).decode()

    return StructuredTool.from_function(
        func=run,
        coroutine=arun,
        name=name,
        description=description
        or (
            "Answer a question about the data in the graph. Give the question in plain language; "
            "it is turned into a graph query."
        ),
        args_schema=_Question,
    )


__all__ = [
    "CYPHER_PROMPT",
    "DIALECT_EXAMPLES",
    "FUNCTION_RESPONSE_SYSTEM",
    "QA_PROMPT",
    "QA_SYSTEM",
    "AgensCypherQAChain",
    "CypherRun",
    "create_cypher_tool",
    "render_cypher_system",
]
