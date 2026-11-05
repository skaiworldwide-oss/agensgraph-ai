"""The QA chain, its tool and the text2cypher retriever, with a scripted model and graph."""

from __future__ import annotations

import json
from typing import Any

import pytest
from agensgraph import Capabilities
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import BaseTool
from langchain_core.tracers.context import collect_runs
from langchain_tests.unit_tests import ToolsUnitTests

from langchain_agensgraph.chains import (
    DIALECT_EXAMPLES,
    AgensCypherQAChain,
    create_cypher_tool,
    render_cypher_system,
)
from langchain_agensgraph.errors import AgensQueryError, UnsafeCypherError
from langchain_agensgraph.graphs.agensgraph import AgensGraph, _readable, _rendered
from langchain_agensgraph.retrievers import AgensText2CypherRetriever

SCHEMA: dict[str, Any] = {
    "node_props": {
        "Person": [{"property": "name", "type": "STRING"}],
        "Company": [{"property": "name", "type": "STRING"}],
        "Secret": [{"property": "code", "type": "STRING"}],
    },
    "rel_props": {"WORKS_AT": []},
    "relationships": [{"start": "Person", "type": "WORKS_AT", "end": "Company"}],
    "counts": {"Person": 2, "Company": 1, "Secret": 1, "WORKS_AT": 2},
    "indexes": [],
}


class Graph(AgensGraph):
    """A graph that answers from a script: each call to ``query_untrusted`` takes the next
    outcome, rows or an error, after the same reading the real one does."""

    def __init__(self, release: str = "2.18.6.1", *outcomes: Any) -> None:
        super().__init__("host=nowhere dbname=none", graph_name="g")
        self.release = Capabilities(release)
        self.outcomes = list(outcomes)
        self.sent: list[tuple[str, dict[str, Any]]] = []
        self.refreshes = 0
        self.structured_schema = SCHEMA
        self.schema = _rendered(SCHEMA)

    @property
    def capabilities(self) -> Capabilities:
        return self.release

    async def acapabilities(self) -> Capabilities:
        return self.release

    def refresh_schema(self, *, force: bool = False) -> None:
        self.refreshes += 1

    async def arefresh_schema(self, *, force: bool = False) -> None:
        self.refreshes += 1

    def query_untrusted(self, text: str, **kwargs: Any) -> list[dict[str, Any]]:
        _readable(text, kwargs["max_rows"], kwargs["allow_writes"])
        self.sent.append((text, kwargs))
        outcome = self.outcomes.pop(0) if self.outcomes else []
        if isinstance(outcome, Exception):
            raise outcome
        return list(outcome)

    async def aquery_untrusted(self, text: str, **kwargs: Any) -> list[dict[str, Any]]:
        return self.query_untrusted(text, **kwargs)


class Model:
    """A model that replies from a script and keeps what it was shown."""

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.seen: list[list[BaseMessage]] = []
        self.kwargs: list[dict[str, Any]] = []

    def __call__(self, prompt: Any, **kwargs: Any) -> AIMessage:
        self.seen.append(prompt.to_messages() if hasattr(prompt, "to_messages") else list(prompt))
        self.kwargs.append(kwargs)
        return AIMessage(content=self.replies.pop(0))

    @property
    def runnable(self) -> RunnableLambda[Any, AIMessage]:
        return RunnableLambda(self)


@pytest.fixture
def graph() -> Any:
    made: list[Graph] = []

    def make(*outcomes: Any, release: str = "2.18.6.1") -> Graph:
        made.append(Graph(release, *outcomes))
        return made[-1]

    yield make
    for each in made:
        each.close()


def qa(graph: Graph, writer: Model, answerer: Model | None = None, **kwargs: Any) -> Any:
    return AgensCypherQAChain.from_llm(
        graph=graph,
        cypher_llm=writer.runnable,
        qa_llm=(answerer or Model(*["An answer."] * 4)).runnable,
        allow_dangerous_requests=True,
        **kwargs,
    )


def scripted(*replies: str) -> RunnableLambda[Any, AIMessage]:
    return Model(*replies).runnable


def test_the_chain_asks_for_the_acknowledgement(graph: Any) -> None:
    with pytest.raises(ValueError, match="allow_dangerous_requests=True"):
        AgensCypherQAChain.from_llm(scripted(), graph=graph())


def test_a_question_is_answered_from_the_rows_of_the_statement_it_became(graph: Any) -> None:
    g = graph([{"name": "Ada"}])
    writer = Model("```cypher\nMATCH (p:Person) RETURN p.name AS name LIMIT 5\n```")
    answerer = Model("Ada.")
    out = qa(g, writer, answerer, top_k=5).invoke({"query": "Who is there?"})
    assert out == {"query": "Who is there?", "result": "Ada."}
    (text, kwargs) = g.sent[0]
    assert text == 'MATCH (p:"Person") RETURN p.name AS name LIMIT 5'
    assert kwargs == {
        "max_rows": 5,
        "timeout": 10.0,
        "allow_writes": False,
        "unsafe_shared_role": False,
    }
    shown = answerer.seen[0][-1].content
    assert "Who is there?" in shown and '"name":"Ada"' in shown.replace(" ", "")


@pytest.mark.parametrize(
    ("refused", "reason"),
    [
        ("MATCH (p:Person) RETURN toFloat(p.age) AS a LIMIT 5", "cast instead"),
        ("MATCH (p:Person) DETACH DELETE p", "a write"),
        ("MATCH (p:Person) RETURN p -- every one", "comment"),
    ],
)
def test_a_refused_statement_goes_back_to_the_model(graph: Any, refused: str, reason: str) -> None:
    g = graph([{"a": 1.0}])
    writer = Model(refused, "MATCH (p:Person) RETURN p.age::float8 AS a LIMIT 5")
    out = qa(g, writer, Model("one"), return_intermediate_steps=True).invoke("Ages?")
    assert [text for text, _ in g.sent] == ['MATCH (p:"Person") RETURN p.age::float8 AS a LIMIT 5']
    failed, ran, context = out["intermediate_steps"]
    assert reason in failed["error"] and ran == {"query": g.sent[0][0]}
    assert context == {"context": [{"a": 1.0}]}
    second = writer.seen[1]
    assert second[-2].content == failed["query"] and reason in second[-1].content


def test_the_server_rejecting_the_statement_goes_back_to_the_model(graph: Any) -> None:
    g = graph(AgensQueryError('syntax error at or near "RETURN"', sqlstate="42601"), [{"n": 2}])
    writer = Model("MATCH (n) RETUR n", "MATCH (n) RETURN count(*) AS n")
    out = qa(g, writer, return_direct=True).invoke("How many?")
    assert out["result"] == [{"n": 2}] and len(g.sent) == 2
    assert "syntax error" in writer.seen[1][-1].content


def test_a_statement_over_the_roles_temp_file_budget_goes_back_to_the_model(graph: Any) -> None:
    budget = AgensQueryError(
        "temporary file size exceeds temp_file_limit (262144kB)", sqlstate="53400"
    )
    g = graph(budget, [{"n": 84}])
    writer = Model(
        "MATCH (p)-->(a) WITH a, count(DISTINCT p) AS n RETURN n",
        "MATCH (p)-->(a) WITH a, count(*) AS n RETURN n",
    )
    out = qa(g, writer, return_direct=True).invoke("Most papers?")
    assert out["result"] == [{"n": 84}] and len(g.sent) == 2
    assert "temp_file_limit" in writer.seen[1][-1].content


def test_a_failure_that_is_not_the_statement_is_raised_at_once(graph: Any) -> None:
    g = graph(AgensQueryError("terminating connection", sqlstate="57P01"))
    writer = Model("MATCH (n) RETURN n LIMIT 1", "never asked")
    with pytest.raises(AgensQueryError, match="terminating"):
        qa(g, writer).invoke("Anything?")
    assert len(writer.seen) == 1


def test_attempts_stop_after_max_retries(graph: Any) -> None:
    writer = Model("RETURN toFloat(1) AS x", "RETURN toInteger(1) AS x")
    with pytest.raises(UnsafeCypherError, match="toInteger"):
        qa(graph(), writer, max_retries=1).invoke("x?")
    assert len(writer.seen) == 2


def test_no_rows_is_an_answer_unless_retry_on_empty(graph: Any) -> None:
    g = graph([], [{"name": "Ada"}])
    writer = Model(
        'MATCH (c:"Company")-[:"KNOWS"]->(p:"Person") RETURN p.name AS name',
        'MATCH (p:"Person") RETURN p.name AS name',
    )
    run = qa(g, writer).run_cypher("Who?", retry_on_empty=True)
    assert run.rows == [{"name": "Ada"}] and "no rows" in run.attempts[0]["error"]


def test_an_edge_against_the_schema_is_turned_unless_validation_is_off(graph: Any) -> None:
    written = 'MATCH (c:"Company")-[:"WORKS_AT"]->(p:"Person") RETURN p.name AS name'
    g = graph()
    qa(g, Model(written), return_direct=True).invoke("Who works where?")
    assert g.sent[0][0] == 'MATCH (c:"Company")<-[:"WORKS_AT"]-(p:"Person") RETURN p.name AS name'
    g = graph()
    qa(g, Model(written), return_direct=True, validate_cypher=False).invoke("Who works where?")
    assert g.sent[0][0] == written


@pytest.mark.parametrize(("release", "count"), [("2.17.0", 16), ("2.18.4.0", 17)])
def test_examples_the_server_cannot_run_are_left_out(graph: Any, release: str, count: int) -> None:
    writer = Model("RETURN 1 AS one")
    qa(graph(release=release), writer, examples=DIALECT_EXAMPLES).invoke("q")
    shown = writer.seen[0]
    taught = [m.content for m in shown if m.type == "ai"]
    assert len(taught) == count and shown[0].type == "system" and shown[-1].type == "human"
    assert ("LET doubled" in " ".join(map(str, taught))) == (count == 17)
    assert taught[0] == DIALECT_EXAMPLES[0][1] and shown[1].content == DIALECT_EXAMPLES[0][0]


def test_examples_and_a_prompt_of_ones_own_refuse_each_other(graph: Any) -> None:
    prompt = ChatPromptTemplate.from_messages([("human", "{schema} {question}")])
    with pytest.raises(ValueError, match="not both"):
        qa(graph(), Model(), examples=[("q", "c")], cypher_prompt=prompt)


def test_cypher_llm_kwargs_carry_a_prompt_and_bind_the_rest(graph: Any) -> None:
    prompt = ChatPromptTemplate.from_messages([("human", "S={schema} Q={question}")])
    writer = Model("RETURN 1 AS one")
    qa(graph(), writer, cypher_llm_kwargs={"prompt": prompt, "stop": ["\n\n"]}).invoke("q")
    assert writer.seen[0][0].content.startswith("S=Node properties")
    assert writer.kwargs[0] == {"stop": ["\n\n"]}


@pytest.mark.parametrize(
    ("limit", "shown", "hidden"),
    [
        ({"include_types": ["Person", "Company", "WORKS_AT"]}, "WORKS_AT", "Secret"),
        ({"exclude_types": ["Secret"]}, "Company", "Secret"),
        ({"include_types": ["Person"]}, "Person", "WORKS_AT"),
    ],
)
def test_the_schema_shown_can_be_limited(
    graph: Any, limit: dict[str, Any], shown: str, hidden: str
) -> None:
    writer = Model("RETURN 1 AS one")
    qa(graph(), writer, **limit).invoke("q")
    schema = writer.seen[0][-1].content
    assert shown in schema and hidden not in schema


@pytest.mark.parametrize(("mode", "reads"), [("ttl", 2), ("frozen", 1)])
def test_a_frozen_schema_is_read_once(graph: Any, mode: str, reads: int) -> None:
    g = graph()
    chain = qa(g, Model("RETURN 1 AS one", "RETURN 1 AS one"), return_direct=True, schema=mode)
    chain.invoke("q")
    chain.invoke("q")
    assert g.refreshes == reads


def test_the_rows_can_reach_the_answer_as_a_tool_response(graph: Any) -> None:
    answerer = Model("Ada.")
    qa(
        graph([{"name": "Ada"}]),
        Model("MATCH (p:Person) RETURN p.name AS name"),
        answerer,
        use_function_response=True,
    ).invoke("Who?")
    _system, question, call, response = answerer.seen[0]
    assert question.content == "Who?" and isinstance(response, ToolMessage)
    assert json.loads(str(response.content)) == [{"name": "Ada"}]
    assert isinstance(call, AIMessage) and call.tool_calls[0]["id"] == response.tool_call_id


async def test_awaiting_gives_what_invoking_gives(graph: Any) -> None:
    g = graph(AgensQueryError("bad", sqlstate="42703"), [{"name": "Ada"}])
    writer = Model(
        "MATCH (p:Person) RETURN p.nme AS name", "MATCH (p:Person) RETURN p.name AS name"
    )
    out = await qa(g, writer, Model("Ada."), return_intermediate_steps=True).ainvoke("Who?")
    assert out["result"] == "Ada." and len(out["intermediate_steps"]) == 3
    assert [text for text, _ in g.sent][-1] == 'MATCH (p:"Person") RETURN p.name AS name'


def test_the_models_run_inside_the_chain_run(graph: Any) -> None:
    with collect_runs() as runs:
        qa(graph(), Model("RETURN 1 AS one"), Model("one")).invoke("q")
    (root,) = runs.traced_runs
    assert root.name == "AgensCypherQAChain" and len(root.child_runs) >= 3


def test_batch_answers_each_question(graph: Any) -> None:
    g = graph()
    chain = qa(g, Model(*["RETURN 1 AS one"] * 3), return_direct=True)
    outs = chain.batch(["a", "b", "c"], config={"max_concurrency": 2})
    assert sorted(out["query"] for out in outs) == ["a", "b", "c"] and len(g.sent) == 3


@pytest.mark.parametrize(
    "release", ["2.14.1", "2.15.0", "2.16.0", "2.17.0", "2.18.4.0", "2.18.6.1"]
)
def test_the_prompt_carries_the_rules_this_release_needs(release: str) -> None:
    text = render_cypher_system(Capabilities(release), top_k=7)
    assert "Quote every label" in text and "LIMIT of at most 7" in text
    assert "Read only" in text and "tointeger" not in text and "tofloat" not in text
    assert ("No GQL" in text) == (release < "2.18")
    assert ("on a ::text cast inside WITH" in text) == (release < "2.18.6")
    assert ("ORDER BY may name only" in text) == (release < "2.17")
    assert ("collect() gives null" in text) == (release < "2.15")
    assert ("A label that is not in the schema is an error" in text) == (release < "2.15")
    assert (", split" in text) == (release >= "2.15")


def test_a_prompt_for_writes_says_so() -> None:
    text = render_cypher_system(Capabilities("2.17.0"), allow_writes=True)
    assert "Read only" not in text and "CREATE or MERGE" in text


def test_the_tool_answers_or_hands_back_the_rows(graph: Any) -> None:
    rows = [{"name": "Ada"}]
    tool = create_cypher_tool(
        graph(rows), scripted("RETURN 'Ada' AS name", "Ada."), allow_dangerous_requests=True
    )
    assert tool.invoke({"question": "Who?"}) == "Ada."
    tool = create_cypher_tool(
        graph(rows), scripted("RETURN 'Ada' AS name"), answer=False, allow_dangerous_requests=True
    )
    assert json.loads(tool.invoke({"question": "Who?"})) == rows


def test_the_retriever_makes_a_document_of_each_row(graph: Any) -> None:
    g = graph([{"name": "Ada"}, {"name": "Bob"}])
    retriever = AgensText2CypherRetriever(
        graph=g,
        llm=scripted("MATCH (p:Person) RETURN p.name AS name"),
        allow_dangerous_requests=True,
    )
    documents = retriever.invoke("Who?", k=1)
    assert [json.loads(d.page_content) for d in documents] == [{"name": "Ada"}]
    assert documents[0].metadata == {"cypher": 'MATCH (p:"Person") RETURN p.name AS name'}
    assert g.sent[0][1]["max_rows"] == 10


class TestCypherToolStandard(ToolsUnitTests):
    @pytest.fixture(autouse=True)
    def _graph(self) -> Any:
        self.graph = Graph()
        yield
        self.graph.close()

    @property
    def tool_constructor(self) -> BaseTool:
        return create_cypher_tool(
            self.graph, scripted("RETURN 1 AS one", "one"), allow_dangerous_requests=True
        )

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"question": "How many people are there?"}
