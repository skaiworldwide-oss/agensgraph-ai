"""Test AgensGraph functionality."""

import pytest

from langchain_agensgraph.vectorstores.agensgraph_vector import (
    AgensgraphVector,
    dict_to_yaml_str,
    remove_lucene_chars,
)
from langchain_agensgraph.vectorstores.utils import DistanceStrategy


def test_escaping_lucene() -> None:
    """Test escaping lucene characters"""
    assert remove_lucene_chars("Hello+World") == "Hello World"
    assert remove_lucene_chars("Hello World\\") == "Hello World"
    assert (
        remove_lucene_chars("It is the end of the world. Take shelter!")
        == "It is the end of the world. Take shelter"
    )
    assert (
        remove_lucene_chars("It is the end of the world. Take shelter&&")
        == "It is the end of the world. Take shelter"
    )
    assert (
        remove_lucene_chars("Bill&&Melinda Gates Foundation")
        == "Bill  Melinda Gates Foundation"
    )
    assert (
        remove_lucene_chars("It is the end of the world. Take shelter(&&)")
        == "It is the end of the world. Take shelter"
    )
    assert (
        remove_lucene_chars("It is the end of the world. Take shelter??")
        == "It is the end of the world. Take shelter"
    )
    assert (
        remove_lucene_chars("It is the end of the world. Take shelter^")
        == "It is the end of the world. Take shelter"
    )
    assert (
        remove_lucene_chars("It is the end of the world. Take shelter+")
        == "It is the end of the world. Take shelter"
    )
    assert (
        remove_lucene_chars("It is the end of the world. Take shelter-")
        == "It is the end of the world. Take shelter"
    )
    assert (
        remove_lucene_chars("It is the end of the world. Take shelter~")
        == "It is the end of the world. Take shelter"
    )


def test_converting_to_yaml() -> None:
    example_dict = {
        "name": "John Doe",
        "age": 30,
        "skills": ["Python", "Data Analysis", "Machine Learning"],
        "location": {"city": "Ljubljana", "country": "Slovenia"},
    }

    yaml_str = dict_to_yaml_str(example_dict)

    expected_output = (
        "name: John Doe\nage: 30\nskills:\n- Python\n- "
        "Data Analysis\n- Machine Learning\nlocation:\n  city: Ljubljana\n"
        "  country: Slovenia\n"
    )

    assert yaml_str == expected_output


class _FakeConnection:
    """Answers the two capability questions and records what was sent."""

    def __init__(self, version=(0, 8, 5)):
        self._version = version
        self.statements: list = []

    def has_vectors(self) -> bool:
        return self._version is not None

    def vector_version(self):
        return self._version

    def execute(self, statement, *args, **kwargs):
        self.statements.append(str(statement))
        return self


def _store(version, **attrs):
    """A store with only what the capability checks read, and no database."""
    store = AgensgraphVector.__new__(AgensgraphVector)
    store.connection = _FakeConnection(version)
    store._vector_type = attrs.get("vector_type", "vector")
    store._distance_strategy = attrs.get("distance_strategy", DistanceStrategy.COSINE)
    store._search_options = attrs.get("search_options", {})
    return store


class TestVectorSupportIsRead:
    """Whether vectors can be read is a catalog question, not a privilege."""

    def test_a_database_with_pgvector_is_accepted_without_sending_anything(self):
        store = _store((0, 8, 5))
        store.verify_vector_support()
        assert store.connection.statements == []

    def test_a_database_without_it_is_told_what_to_run(self):
        store = _store(None)
        with pytest.raises(ValueError, match="CREATE EXTENSION vector"):
            store.verify_vector_support()
        # Creating an extension is a privilege an application role usually lacks, and
        # trying it reports the refusal as pgvector not being on the server.
        assert store.connection.statements == []


class TestAFeatureNamesTheVersionItNeeds:
    """pgvector gates its own features, and a server without one says so obscurely."""

    @pytest.mark.parametrize(
        "version,attrs,needs",
        [
            ((0, 6, 0), {"vector_type": "halfvec"}, "0.7.0"),
            ((0, 6, 0), {"vector_type": "sparsevec"}, "0.7.0"),
            ((0, 6, 0), {"distance_strategy": DistanceStrategy.TAXICAB}, "0.7.0"),
            (
                (0, 7, 4),
                {"search_options": {"hnsw.iterative_scan": "relaxed_order"}},
                "0.8.0",
            ),
        ],
    )
    def test_too_old_is_refused_by_number(self, version, attrs, needs):
        store = _store(version, **attrs)
        with pytest.raises(ValueError, match=needs):
            store._verify_vector_version()

    @pytest.mark.parametrize(
        "attrs",
        [
            {"vector_type": "halfvec"},
            {"vector_type": "sparsevec"},
            {"distance_strategy": DistanceStrategy.TAXICAB},
            {"search_options": {"hnsw.iterative_scan": "relaxed_order"}},
        ],
    )
    def test_new_enough_is_accepted(self, attrs):
        _store((0, 8, 0), **attrs)._verify_vector_version()

    def test_the_ordinary_configuration_needs_nothing_recent(self):
        _store((0, 5, 0))._verify_vector_version()
