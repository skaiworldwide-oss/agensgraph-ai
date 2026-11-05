"""A chat model for running the demos without an API key, answering the same way every time.

:class:`FakeChatModel` is a langchain_core ``BaseChatModel`` that answers from what it is asked:

- asked for a graph by ``LLMGraphTransformer`` (a forced call of its ``_Graph`` tool), it takes the
  capitalized phrases of the text as entities, gives each an allowed type chosen by a hash of its
  name, and relates the first to each of the others;
- asked for Cypher by ``AgensCypherQAChain``, it writes the statement :data:`CYPHER` holds for the
  question, or else one counting the vertices of each label;
- given tools by an agent, it calls the one the last message asks for, and once the tool has
  answered, repeats the start of what it returned;
- asked anything else, it repeats the start of the last message.

langchain_core's own fake chat models answer from a fixed list and cannot bind tools, which the
graph transformer and the agent both need.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool

DEFAULT_CYPHER = "MATCH (n) RETURN label(n) AS kind, count(*) AS cnt ORDER BY cnt DESC LIMIT 10"

# The statement written for each question the demos ask, keyed by the question in lower case.
CYPHER: dict[str, str] = {
    "what types of entities are in the graph, and how many of each?": DEFAULT_CYPHER,
    "which 5 wikipedia articles mention the most distinct entities?": (
        'MATCH (d:"Document")-[:"MENTIONS"]->(e) '
        "RETURN d.title AS title, count(*) AS entities ORDER BY entities DESC LIMIT 5"
    ),
    "show 8 example relationships as source, relationship, target.": (
        "MATCH (a)-[r]->(b) WHERE type(r) <> 'MENTIONS' "
        "RETURN a.id AS source, type(r) AS relationship, b.id AS target LIMIT 8"
    ),
    "how many films did ines varga direct?": (
        "MATCH (p:person)-[:directed]->(m:\"Movie\") WHERE p.name = 'Ines Varga' "
        "RETURN count(*) AS films LIMIT 1"
    ),
    "how many films did each director make? give the name and the count.": (
        'MATCH (p:person)-[:directed]->(m:"Movie") '
        "RETURN p.name AS director, count(*) AS films ORDER BY films DESC LIMIT 10"
    ),
    "which author has written the most papers?": (
        'MATCH (p:"Paper")-[:"AUTHORED_BY"]->(a:"Author") '
        "RETURN a.name AS author, count(*) AS papers ORDER BY papers DESC LIMIT 3"
    ),
}

CYPHER_SYSTEM = "Write one Cypher statement"
GRAPH_TOOL = "_Graph"
NODE_TYPES = "Extract only nodes of these types:"
RELATIONSHIP_TYPES = "Extract only relationships of these types:"
TRIPLES = "Extract only these relationships, in this direction"

_STOPWORDS = frozenset(
    "A An And As At But By During For From He Her His However In It Its Of On Or She The Their "
    "They This To When While With".split()
)
_PHRASE = re.compile(r"[A-Z][\w'’-]*(?:(?:\s+(?:of|de|von|van|the))?\s+[A-Z][\w'’-]*)*")
_TRIPLE = re.compile(r"\(([^,()]+), ([^,()]+), ([^,()]+)\)")


class FakeChatModel(BaseChatModel):
    """A chat model whose answer depends only on the messages and tools it is given."""

    cypher: dict[str, str] = CYPHER

    @property
    def _llm_type(self) -> str:
        return "demos-fake-chat"

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: Any = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        formatted = [convert_to_openai_tool(tool) for tool in tools]
        return self.bind(tools=formatted, tool_choice=tool_choice, **kwargs)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        tools = [tool["function"] for tool in kwargs.get("tools") or []]
        message = self._respond(messages, tools, kwargs.get("tool_choice"))
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _respond(
        self, messages: list[BaseMessage], tools: list[dict[str, Any]], tool_choice: Any
    ) -> AIMessage:
        system = "\n".join(_text(m) for m in messages if isinstance(m, SystemMessage))
        if tool_choice and any(tool["name"] == GRAPH_TOOL for tool in tools):
            args = extract(system, _text(messages[-1]))
            return AIMessage(content="", tool_calls=[_call(GRAPH_TOOL, args, len(messages))])
        if system.startswith(CYPHER_SYSTEM):
            question = _question(messages)
            return AIMessage(content=self.cypher.get(question.lower(), DEFAULT_CYPHER))
        if tools:
            return _agent_turn(messages, tools)
        return AIMessage(content="[fake answer] " + _squeezed(_text(messages[-1]))[:240])


def extract(system: str, text: str, limit: int = 6) -> dict[str, Any]:
    """The ``_Graph`` arguments for a text: its capitalized phrases, typed and related."""
    names = entities(text, limit)
    triples = _TRIPLE.findall(_line(system, TRIPLES).partition("): ")[2])
    if triples:
        start, relationship, end = (part.strip() for part in triples[0])
        typed = [(name, start if i == 0 else end) for i, name in enumerate(names)]
    else:
        kinds = _listed(system, NODE_TYPES) or ["Entity"]
        typed = [(name, kinds[_digest(name) % len(kinds)]) for name in names]
        allowed = _listed(system, RELATIONSHIP_TYPES)
        relationship = "RELATED_TO" if not allowed or "RELATED_TO" in allowed else allowed[0]
    nodes = [{"id": name, "type": kind, "properties": []} for name, kind in typed]
    relationships = [
        {
            "source_id": typed[0][0],
            "source_type": typed[0][1],
            "target_id": name,
            "target_type": kind,
            "type": relationship,
            "properties": [],
        }
        for name, kind in typed[1:]
    ]
    return {"nodes": nodes, "relationships": relationships}


def entities(text: str, limit: int = 6) -> list[str]:
    """The first ``limit`` distinct runs of capitalized words, leading stopwords dropped."""
    found: list[str] = []
    for match in _PHRASE.finditer(text):
        words = match.group(0).split()
        while words and words[0] in _STOPWORDS:
            words = words[1:]
        name = " ".join(words).strip(".,;:'’-")
        if len(name) < 3 or name in found:
            continue
        found.append(name)
        if len(found) == limit:
            break
    return found


def _agent_turn(messages: list[BaseMessage], tools: list[dict[str, Any]]) -> AIMessage:
    last = messages[-1]
    if isinstance(last, ToolMessage):
        returned = _squeezed(_text(last))[:240]
        return AIMessage(content=f"From {last.name or 'the tool'}: {returned}")
    text = _text(last)
    lowered = text.lower()
    names = [tool["name"] for tool in tools]

    def named(prefix: str) -> dict[str, Any] | None:
        return next((tool for tool in tools if tool["name"].startswith(prefix)), None)

    remember, recall = named("remember"), named("recall")
    search = next((tool for tool in tools if "search" in tool["name"]), None)
    if remember is not None and "remember" in lowered:
        chosen = remember
    elif recall is not None and ("know about me" in lowered or "recall" in lowered):
        chosen = recall
    elif search is not None and not re.search(r"(without|do not|don't) search", lowered):
        chosen = search
    else:
        humans = [m for m in messages if isinstance(m, HumanMessage)]
        if len(humans) > 1:
            first = _text(humans[0])
            return AIMessage(content=f"Your first message in this conversation was: {first}")
        return AIMessage(content=f"You said: {text}. Tools available: {', '.join(names)}.")
    properties = chosen.get("parameters", {}).get("properties", {})
    required = chosen.get("parameters", {}).get("required") or list(properties)
    args = {required[0]: text} if required else {}
    return AIMessage(content="", tool_calls=[_call(chosen["name"], args, len(messages))])


def _call(name: str, args: dict[str, Any], position: int) -> dict[str, Any]:
    digest = hashlib.sha256(json.dumps([name, args], sort_keys=True).encode()).hexdigest()
    return {"name": name, "args": args, "id": f"call_{position}_{digest[:12]}", "type": "tool_call"}


def _question(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage) and "Question: " in _text(message):
            return _text(message).rsplit("Question: ", 1)[1].strip()
    return ""


def _line(system: str, lead: str) -> str:
    return next((line for line in system.splitlines() if line.startswith(lead)), "")


def _listed(system: str, lead: str) -> list[str]:
    line = _line(system, lead)
    return [part.strip() for part in line[len(lead) :].split(",") if part.strip()]


def _digest(name: str) -> int:
    return int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)


def _text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return " ".join(
        part if isinstance(part, str) else str(part.get("text", "")) for part in content
    )


def _squeezed(text: str) -> str:
    return " ".join(text.split())
