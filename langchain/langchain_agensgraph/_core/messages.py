"""SQL over messages stored in LangChain's ``message_to_dict`` form."""

from __future__ import annotations


def text_of(message: str) -> str:
    """SQL for a stored message's text, ``message`` being SQL for its ``message_to_dict`` jsonb.

    It reads as LangChain's ``BaseMessage.text`` does: the content when it is a string, and when
    it is a list of blocks, the strings and ``text`` blocks in it joined. A model's reply often
    comes as such a list, which a Cypher cast to text refuses on 2.14.
    """
    content = f"({message})->'data'->'content'"
    return (
        f"CASE jsonb_typeof({content}) WHEN 'string' THEN {content} #>> '{{}}' "
        "WHEN 'array' THEN (SELECT coalesce(string_agg(CASE WHEN jsonb_typeof(b) = 'string' "
        "THEN b #>> '{}' WHEN b->>'type' = 'text' THEN b->>'text' END, '' ORDER BY o), '') "
        f"FROM jsonb_array_elements({content}) WITH ORDINALITY AS t(b, o)) ELSE '' END"
    )


def message_fields(message: str) -> str:
    """SQL for the ``jsonb_build_object`` arguments every read of a stored message ends with: its
    ``type``, ``text`` (:func:`text_of`), ``content`` and the whole ``message``."""
    return (
        f"'type', ({message})->>'type', 'text', {text_of(message)}, "
        f"'content', ({message})->'data'->'content', 'message', {message}"
    )
