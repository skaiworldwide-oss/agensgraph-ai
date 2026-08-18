"""What a relationship type may be, read from what the server will store."""

import unicodedata

import pytest
from agensgraph.introspect import MAX_IDENTIFIER
from mcp_agensgraph_memory.agensgraph_memory import canonical_name, canonical_relation_type


class TestATypeTheServerWillStore:
    """A quoted label takes far more than ASCII letters, and the gate said otherwise.

    Every one of these was measured being created and read back spelled the way it was written,
    so refusing them meant a deployment not writing in English could not name a relationship.
    """

    @pytest.mark.parametrize(
        ("written", "stored"),
        [
            ("WORKS_AT", "WORKS_AT"),
            ("works_at", "WORKS_AT"),
            ("Works_At", "WORKS_AT"),
            ("A B", "A B"),
            ("RELÉ", "RELÉ"),
            ("relé", "RELÉ"),
            ("관계", "관계"),
            ("X🙂Y", "X🙂Y"),
            ("A-B", "A-B"),
            ("1ST", "1ST"),
        ],
    )
    def test_it_is_taken_and_folded_to_one_spelling(self, written, stored):
        assert canonical_relation_type(written) == stored


class TestATypeTheServerWillNot:
    """Two limits, and the length one is the one that bites silently.

    Past 63 bytes the server truncates rather than refusing, so the next type sharing those
    bytes comes back as ``DuplicateTable`` naming a label the caller never wrote.
    """

    def test_a_name_at_the_limit_is_taken(self):
        assert canonical_relation_type("A" * MAX_IDENTIFIER) == "A" * MAX_IDENTIFIER

    @pytest.mark.parametrize(
        "written",
        [
            "A" * (MAX_IDENTIFIER + 1),
            "가" * 22,  # 66 bytes, though only 22 characters
        ],
    )
    def test_one_past_it_is_refused_rather_than_cut(self, written):
        with pytest.raises(ValueError, match="bytes"):
            canonical_relation_type(written)

    def test_length_is_measured_after_case_folding(self):
        """Upper casing can lengthen: 'ǰ' is two bytes and 'J̌' is three."""
        written = "ǰ" * 31 + "AA"  # 64 bytes as written, 95 once upper-cased
        assert len(written.encode("utf-8")) <= MAX_IDENTIFIER + 2
        with pytest.raises(ValueError, match="bytes"):
            canonical_relation_type(written)

    @pytest.mark.parametrize("written", ["A\x00B", "", None, 5])
    def test_what_cannot_be_an_identifier_at_all(self, written):
        with pytest.raises(ValueError):
            canonical_relation_type(written)


class TestOneSpellingOfAName:
    """The same visible name has more than one encoding.

    An accented letter is one code point or a letter and a combining mark, and macOS hands over
    the second when text is pasted. They are different strings, so they were different entities:
    two rows both reading ``Cafe`` with an accent, which the unique index cannot collapse because
    it is doing what it was asked, and looking one up left the other unreachable.
    """

    def test_two_encodings_of_a_name_become_one(self):
        composed = unicodedata.normalize("NFC", "Café")
        decomposed = unicodedata.normalize("NFD", "Café")
        assert composed != decomposed
        assert canonical_name(composed) == canonical_name(decomposed)

    def test_a_relationship_type_is_folded_the_same_way(self):
        assert canonical_relation_type(
            unicodedata.normalize("NFD", "RELÉ")
        ) == canonical_relation_type("RELÉ")

    def test_a_name_already_composed_is_unchanged(self):
        for name in ("John Smith", "SKAI Worldwide", "관계"):
            assert canonical_name(name) == name
