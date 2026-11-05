"""A channel's value as items, per mode, and a delta channel's value rebuilt from its history."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pytest
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, message_to_dict
from langgraph.checkpoint.serde.types import _DeltaSnapshot

from langchain_agensgraph.checkpoint._items import checked, folded, items


def test_a_mode_must_be_one_the_saver_knows() -> None:
    assert checked({"messages": "messages", "todo": "append"}) == {
        "messages": "messages",
        "todo": "append",
    }
    with pytest.raises(ValueError, match="'reducer'"):
        checked({"x": "reducer"})


def test_messages_are_keyed_by_id_in_order() -> None:
    ada, bob = HumanMessage("hi", id="m1"), AIMessage("hello", id="m2")
    assert items("messages", "messages", [ada, bob]) == [
        ("m1", 0, message_to_dict(ada)),
        ("m2", 1, message_to_dict(bob)),
    ]
    (key, position, value) = items("messages", "messages", [("user", "no id")])[0]
    assert (key, position, value["type"]) == ("#0", 0, "human")


@pytest.mark.parametrize("single", ["hi", ("human", "hi"), HumanMessage("hi")])
def test_a_value_that_is_not_a_list_is_one_message_as_add_messages_reads_it(single: object) -> None:
    (only,) = items("messages", "messages", single)
    assert (only[2]["type"], only[2]["data"]["content"]) == ("human", "hi")
    assert [m.content for m in folded("messages", {"seed": single, "writes": []})] == ["hi"]


def test_a_snapshot_is_read_as_its_value() -> None:
    message = HumanMessage("hi", id="m1")
    assert items("m", "messages", _DeltaSnapshot([message])) == [
        ("m1", 0, message_to_dict(message))
    ]


@dataclass
class Note:
    text: str


def test_other_values_become_json_and_what_has_none_is_left_out(
    caplog: pytest.LogCaptureFixture,
) -> None:
    assert items("todo", "append", ["a", {"b": (1, 2)}, Note("c")]) == [
        ("0", 0, "a"),
        ("1", 1, {"b": [1, 2]}),
        ("2", 2, {"text": "c"}),
    ]
    assert items("todo", "append", "one") == [("0", 0, "one")]
    assert items("plan", "last_value", {"step": 2}) == [("", 0, {"step": 2})]
    with caplog.at_level(logging.WARNING):
        assert items("todo", "append", [1, object(), float("nan")]) == [("0", 0, 1)]
    assert "'todo'" in caplog.text and "no JSON form" in caplog.text


def test_text_jsonb_cannot_hold_is_cleaned_or_left_out(caplog: pytest.LogCaptureFixture) -> None:
    assert items("todo", "append", ["a\x00b", {"k\x00": "v\x00"}]) == [
        ("0", 0, "ab"),
        ("1", 1, {"k": "v"}),
    ]
    with caplog.at_level(logging.WARNING):
        assert items("todo", "append", ["fine", "a\ud800"]) == [("0", 0, "fine")]
    assert "not valid Unicode" in caplog.text
    ((key, _, value),) = items("m", "messages", [HumanMessage("hi", id="m\x001")])
    assert (key, value["data"]["id"]) == ("m1", "m1")
    assert items("m", "messages", [HumanMessage("hi", id="m\ud8001")]) == []


def test_a_delta_history_folds_as_its_mode_reads() -> None:
    first, second = HumanMessage("a", id="m1"), AIMessage("b", id="m2")
    history = {
        "seed": _DeltaSnapshot([first]),
        "writes": [("t1", "messages", [second]), ("t2", "messages", [RemoveMessage(id="m1")])],
    }
    assert [m.id for m in folded("messages", history)] == ["m2"]
    writes = {"writes": [("t1", "todo", ["x"]), ("t2", "todo", "y")]}
    assert folded("append", {"seed": ["w"], **writes}) == ["w", "x", "y"]
    assert folded("last_value", writes) == "y"
    assert folded("last_value", {"seed": 3, "writes": []}) == 3
    assert folded("messages", {"writes": []}) == []
