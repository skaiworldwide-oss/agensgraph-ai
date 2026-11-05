"""A channel's value as rows a Cypher statement can read: ``queryable_channels``.

A saver never sees a channel's type or reducer, so the caller declares how each channel is read:

=============== ======================================== ==========================================
mode            for                                      one item per
=============== ======================================== ==========================================
``messages``    a list kept by ``add_messages``          message, keyed by its id
``append``      a list kept by ``operator.add``          element, keyed by its position
``last_value``  any other value                          channel, the whole value
=============== ======================================== ==========================================

An item's value is plain JSON: a message as ``message_to_dict`` gives it, anything else as its JSON
form. A value that has none is left out with a warning. Strings and keys lose their null characters,
which ``jsonb`` and ``text`` cannot hold.
"""

from __future__ import annotations

import dataclasses
import logging
import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from langchain_core.load import dumpd
from langchain_core.load.serializable import Serializable
from langchain_core.messages import BaseMessage, convert_to_messages, message_to_dict
from langgraph.checkpoint.serde.types import _DeltaSnapshot
from pydantic import BaseModel

logger = logging.getLogger(__name__)

Mode = Literal["messages", "append", "last_value"]
MODES: tuple[Mode, ...] = ("messages", "append", "last_value")

Item = tuple[str, int, Any]
"""An item: its key, its position in the channel, and its value as JSON."""


def checked(channels: Mapping[str, str]) -> dict[str, Mode]:
    """The declared channels, each with a mode this module knows."""
    out: dict[str, Mode] = {}
    for channel, mode in channels.items():
        if mode not in MODES:
            raise ValueError(
                f"queryable channel {channel!r} has mode {mode!r}; a mode is one of {MODES}"
            )
        out[channel] = mode
    return out


def items(channel: str, mode: Mode, value: Any) -> list[Item]:
    """The items of a channel holding ``value``; the last of two with one key is kept."""
    if isinstance(value, _DeltaSnapshot):
        value = value.value
    found: dict[str, tuple[int, Any]] = {}
    if mode == "messages":
        for position, message in enumerate(_messages(value)):
            encoded = _json(channel, message_to_dict(message))
            if encoded is not _UNENCODABLE:
                found[_key(message.id) if message.id else f"#{position}"] = (position, encoded)
    elif mode == "append":
        for position, element in enumerate(_sequence(value)):
            encoded = _json(channel, element)
            if encoded is not _UNENCODABLE:
                found[str(position)] = (position, encoded)
    else:
        encoded = _json(channel, value)
        if encoded is not _UNENCODABLE:
            found[""] = (0, encoded)
    return [(key, position, encoded) for key, (position, encoded) in found.items()]


def folded(mode: Mode, history: Mapping[str, Any]) -> Any:
    """A delta channel's value rebuilt from its history: the seed, then each write in order."""
    seed = history.get("seed")
    state = seed.value if isinstance(seed, _DeltaSnapshot) else seed
    writes = [write[2] for write in history.get("writes", ())]
    if mode == "messages":
        from langgraph.graph.message import add_messages

        merged: Any = list(_messages(state)) if state is not None else []
        for write in writes:
            merged = add_messages(merged, write)
        return merged
    if mode == "append":
        out = list(_sequence(state)) if state is not None else []
        for write in writes:
            out += _sequence(write)
        return out
    return writes[-1] if writes else state


def _messages(value: Any) -> list[BaseMessage]:
    """A message channel's value as messages, read as ``add_messages`` reads one: a list holds
    messages, and anything else is one -- a message, a string, a ``(role, content)`` tuple."""
    if value is None:
        return []
    return convert_to_messages(value if isinstance(value, list) else [value])


def _sequence(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return list(value)
    return [value]


_UNENCODABLE = object()


def _json(channel: str, value: Any) -> Any:
    try:
        return _plain(value)
    except (TypeError, ValueError) as exc:
        logger.warning("queryable channel %r: an item was left out: %s", channel, exc)
        return _UNENCODABLE


def _key(key: str) -> str:
    """A message's id as an item's key: without null characters, as its value holds it. An id
    that is not valid Unicode never gets here, since the message's value holds it too and is left
    out."""
    return key.replace("\x00", "")


def _text(value: str) -> str:
    """A string without its null characters; one that is not valid Unicode has no JSON form."""
    if not value.isascii():
        try:
            value.encode()
        except UnicodeEncodeError:
            raise ValueError("a string that is not valid Unicode has no JSON form") from None
    return value.replace("\x00", "")


def _plain(value: Any) -> Any:
    """``value`` as JSON: numbers, strings, booleans, null, lists and maps with string keys."""
    if value is None or isinstance(value, bool | int):
        return value
    if isinstance(value, str):
        return _text(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{value} has no JSON form")
        return value
    if isinstance(value, Mapping):
        return {_text(str(key)): _plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set | frozenset):
        return [_plain(item) for item in value]
    if isinstance(value, BaseMessage):
        return _plain(message_to_dict(value))
    if isinstance(value, Serializable):
        return _plain(dumpd(value))
    if isinstance(value, BaseModel):
        return _plain(value.model_dump(mode="json"))
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _plain(dataclasses.asdict(value))
    raise TypeError(f"a {type(value).__name__} has no JSON form")
