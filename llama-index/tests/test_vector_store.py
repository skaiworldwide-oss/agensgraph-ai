from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index_agensgraph.vector_stores.agensgraph import AgensgraphVectorStore


def test_class():
    names_of_base_classes = [b.__name__ for b in AgensgraphVectorStore.__mro__]
    assert BasePydanticVectorStore.__name__ in names_of_base_classes
    assert "client" not in AgensgraphVectorStore.__abstractmethods__
