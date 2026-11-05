"""Names the server keeps as given, and derived names that stay unique when shortened."""

from __future__ import annotations

import pytest

from langchain_agensgraph._core.names import (
    MAX_IDENTIFIER_BYTES,
    bounded,
    checked,
    companion_schema,
    companion_table,
    ident,
    literal,
    quoted,
    quoted_literal,
)


def test_identifiers_and_literals_are_quoted() -> None:
    assert ident('we"ird') == '"we""ird"'
    assert literal("it's") == "'it''s'"


def test_a_percent_is_doubled_for_a_statement_with_bound_parameters() -> None:
    assert ident("50%") == '"50%%"'
    assert literal("50%") == "'50%%'"
    assert quoted("50%") == '"50%"'
    assert quoted_literal("it's 50%") == "'it''s 50%'"


def test_a_name_that_fits_is_kept_whole() -> None:
    assert bounded("Chunk__embedding__vec") == "Chunk__embedding__vec"
    assert companion_schema("g") == "g_store"
    assert companion_table("Chunk", "embedding", "vec") == "Chunk__embedding__vec"


def test_a_long_name_is_shortened_to_a_prefix_and_a_hash_of_the_whole() -> None:
    long_a = "x" * 70 + "a"
    long_b = "x" * 70 + "b"
    assert len(bounded(long_a).encode()) <= MAX_IDENTIFIER_BYTES
    assert bounded(long_a) != bounded(long_b)
    assert bounded(long_a) == bounded(long_a)


def test_a_multibyte_name_is_cut_on_a_character_boundary() -> None:
    name = "그래프" * 30
    shortened = bounded(name)
    assert len(shortened.encode()) <= MAX_IDENTIFIER_BYTES
    shortened.encode().decode()


@pytest.mark.parametrize("bad", ["", "a\x00b", "y" * 64])
def test_a_chosen_name_the_server_would_not_read_as_given_is_refused(bad: str) -> None:
    with pytest.raises(ValueError):
        checked(bad, "graph name")


def test_a_chosen_name_that_fits_is_returned() -> None:
    assert checked("Person", "label") == "Person"
