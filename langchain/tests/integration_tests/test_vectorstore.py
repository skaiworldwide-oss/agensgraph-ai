"""Test AgensgraphVector functionality."""

import os
from hashlib import md5
from math import isclose
from typing import Any, Dict, List, cast

import pytest
from langchain_core.documents import Document
from psycopg import sql
from yaml import safe_load


def _no_id(docs):
    """Drop ``Document.id`` from results.

    These tests pre-date 0.2.0's ``Document.id`` population (which the
    LangChain conformance suite requires). Comparing on page_content +
    metadata only preserves the original test intent.
    """
    out = []
    for item in docs:
        if isinstance(item, tuple):
            d, s = item
            out.append((Document(page_content=d.page_content, metadata=d.metadata), s))
        else:
            out.append(
                Document(page_content=item.page_content, metadata=item.metadata)
            )
    return out

from langchain_agensgraph.graphs.agensgraph import AgensGraph
from langchain_agensgraph.vectorstores.agensgraph_vector import (
    AgensgraphVector,
    SearchType,
)
from langchain_agensgraph.vectorstores.utils import DistanceStrategy
from tests.integration_tests.fake_embeddings import (
    AngularTwoDimensionalEmbeddings,
    FakeEmbeddings,
)
from tests.integration_tests.fixtures.filtering_test_cases import (
    DOCUMENTS,
    TYPE_1_FILTERING_TEST_CASES,
    TYPE_2_FILTERING_TEST_CASES,
    TYPE_3_FILTERING_TEST_CASES,
    TYPE_4_FILTERING_TEST_CASES,
)

OS_TOKEN_COUNT = 1536

texts = ["foo", "bar", "baz", "It is the end of the world. Take shelter!"]

conf = {
    "dbname": os.getenv("AGENSGRAPH_DB"),
    "user": os.getenv("AGENSGRAPH_USER"),
    "password": os.getenv("AGENSGRAPH_PASSWORD"),
    "host": os.getenv("AGENSGRAPH_HOST", "localhost"),
    "port": int(os.getenv("AGENSGRAPH_PORT", 5432)),
}

url = os.environ.get("AGENSGRAPH_URL", f"postgresql://{conf['user']}:{conf['password']}@{conf['host']}:{conf['port']}/{conf['dbname']}")

def drop_vector_indexes(store: AgensgraphVector) -> None:
    """Cleanup all vector indexes"""
    for index in store._vector_indexes():
        store.query(f"""DROP PROPERTY INDEX "{index['name']}" CASCADE""")

    store.query("MATCH (n) DETACH DELETE n;")

def drop_fulltext_indexes(store: AgensgraphVector) -> None:
    """Cleanup all keyword indexes"""
    for index in store._text_indexes():
        store.query(f"""DROP PROPERTY INDEX "{index['name']}" CASCADE""")

    store.query("MATCH (n) DETACH DELETE n;")

class FakeEmbeddingsWithOsDimension(FakeEmbeddings):
    """Fake embeddings functionality for testing."""

    def embed_documents(self, embedding_texts: List[str]) -> List[List[float]]:
        """Return simple embeddings."""
        return [
            [float(1.0)] * (OS_TOKEN_COUNT - 1) + [float(i + 1)]
            for i in range(len(embedding_texts))
        ]

    def embed_query(self, text: str) -> List[float]:
        """Return simple embeddings."""
        return [float(1.0)] * (OS_TOKEN_COUNT - 1) + [float(texts.index(text) + 1)]


def test_agensgraph_vector() -> None:
    """Test end to end construction and search."""
    docsearch = AgensgraphVector.from_texts(
        texts=texts,
        embedding=FakeEmbeddingsWithOsDimension(),
        url=url,
        pre_delete_collection=True,
    )
    output = docsearch.similarity_search("foo", k=1)
    assert _no_id(output) == [Document(page_content="foo")]

    drop_vector_indexes(docsearch)


def test_agensgraph_vector_euclidean() -> None:
    """Test euclidean distance"""
    docsearch = AgensgraphVector.from_texts(
        texts=texts,
        embedding=FakeEmbeddingsWithOsDimension(),
        url=url,
        pre_delete_collection=True,
        distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
    )
    output = docsearch.similarity_search("foo", k=1)
    assert _no_id(output) == [Document(page_content="foo")]

    drop_vector_indexes(docsearch)


def test_agensgraph_vector_embeddings() -> None:
    """Test end to end construction with embeddings and search."""
    text_embeddings = FakeEmbeddingsWithOsDimension().embed_documents(texts)
    text_embedding_pairs = list(zip(texts, text_embeddings))
    docsearch = AgensgraphVector.from_embeddings(
        text_embeddings=text_embedding_pairs,
        embedding=FakeEmbeddingsWithOsDimension(),
        url=url,
        pre_delete_collection=True,
    )
    output = docsearch.similarity_search("foo", k=1)
    assert _no_id(output) == [Document(page_content="foo")]

    drop_vector_indexes(docsearch)


def test_agensgraph_vector_catch_wrong_index_name() -> None:
    """Test if index name is misspelled, but node label and property are correct."""
    text_embeddings = FakeEmbeddingsWithOsDimension().embed_documents(texts)
    text_embedding_pairs = list(zip(texts, text_embeddings))
    AgensgraphVector.from_embeddings(
        text_embeddings=text_embedding_pairs,
        embedding=FakeEmbeddingsWithOsDimension(),
        url=url,
        pre_delete_collection=True,
    )
    existing = AgensgraphVector.from_existing_index(
        embedding=FakeEmbeddingsWithOsDimension(),
        url=url,
        index_name="test",
    )
    output = existing.similarity_search("foo", k=1)
    assert _no_id(output) == [Document(page_content="foo")]

    drop_vector_indexes(existing)


def test_agensgraph_vector_catch_wrong_node_label() -> None:
    """Test if node label is misspelled, but index name is correct."""
    text_embeddings = FakeEmbeddingsWithOsDimension().embed_documents(texts)
    text_embedding_pairs = list(zip(texts, text_embeddings))
    AgensgraphVector.from_embeddings(
        text_embeddings=text_embedding_pairs,
        embedding=FakeEmbeddingsWithOsDimension(),
        url=url,
        pre_delete_collection=True,
    )
    existing = AgensgraphVector.from_existing_index(
        embedding=FakeEmbeddingsWithOsDimension(),
        url=url,
        index_name="vector",
        node_label="test",
    )
    output = existing.similarity_search("foo", k=1)
    assert _no_id(output) == [Document(page_content="foo")]

    drop_vector_indexes(existing)


def test_agensgraph_vector_with_metadatas() -> None:
    """Test end to end construction and search."""
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = AgensgraphVector.from_texts(
        texts=texts,
        embedding=FakeEmbeddingsWithOsDimension(),
        metadatas=metadatas,
        url=url,
        pre_delete_collection=True,
    )
    output = docsearch.similarity_search("foo", k=1)
    assert _no_id(output) == [Document(page_content="foo", metadata={"page": "0"})]

    drop_vector_indexes(docsearch)


def test_agensgraph_vector_with_metadatas_with_scores() -> None:
    """Test end to end construction and search."""
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = AgensgraphVector.from_texts(
        texts=texts,
        embedding=FakeEmbeddingsWithOsDimension(),
        metadatas=metadatas,
        url=url,
        pre_delete_collection=True,
    )
    output = [
        (doc, round(score, 1))
        for doc, score in docsearch.similarity_search_with_score("foo", k=1)
    ]
    assert _no_id(output) == [(Document(page_content="foo", metadata={"page": "0"}), 1.0)]

    drop_vector_indexes(docsearch)


def test_agensgraph_vector_relevance_score() -> None:
    """Test to make sure the relevance score is scaled to 0-1."""
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = AgensgraphVector.from_texts(
        texts=texts,
        embedding=FakeEmbeddingsWithOsDimension(),
        metadatas=metadatas,
        url=url,
        pre_delete_collection=True,
    )

    output = docsearch.similarity_search_with_relevance_scores("foo", k=3)
    expected_output = [
        (Document(page_content="foo", metadata={"page": "0"}), 1.0),
        (Document(page_content="bar", metadata={"page": "1"}), 0.9996752725007386),
        (Document(page_content="baz", metadata={"page": "2"}), 0.998704667587627),
    ]

    # Check if the length of the outputs matches
    assert len(output) == len(expected_output)

    # Check if each document and its relevance score is close to the expected value
    for (doc, score), (expected_doc, expected_score) in zip(output, expected_output):
        assert doc.page_content == expected_doc.page_content
        assert doc.metadata == expected_doc.metadata
        assert isclose(score, expected_score, rel_tol=1e-5)

    drop_vector_indexes(docsearch)


def test_agensgraph_vector_retriever_search_threshold() -> None:
    """Test using retriever for searching with threshold."""
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = AgensgraphVector.from_texts(
        texts=texts,
        embedding=FakeEmbeddingsWithOsDimension(),
        metadatas=metadatas,
        url=url,
        pre_delete_collection=True,
    )

    retriever = docsearch.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": 0.9999},
    )
    output = retriever.invoke("foo")
    assert _no_id(output) == [
        Document(page_content="foo", metadata={"page": "0"}),
    ]

    drop_vector_indexes(docsearch)


def test_custom_return_agensgraph_vector() -> None:
    """Test end to end construction and search."""
    docsearch = AgensgraphVector.from_texts(
        texts=["test"],
        embedding=FakeEmbeddingsWithOsDimension(),
        url=url,
        pre_delete_collection=True,
        retrieval_query="RETURN 'foo' AS text, score, {{test: 'test'}} AS metadata",
    )
    output = docsearch.similarity_search("foo", k=1)
    assert _no_id(output) == [Document(page_content="foo", metadata={"test": "test"})]

    drop_vector_indexes(docsearch)


def test_agensgraph_vector_prefer_indexname() -> None:
    """Test using when two indexes are found, prefer by index_name."""
    AgensgraphVector.from_texts(
        texts=["foo"],
        embedding=FakeEmbeddingsWithOsDimension(),
        url=url,
        pre_delete_collection=True,
    )

    AgensgraphVector.from_texts(
        texts=["bar"],
        embedding=FakeEmbeddingsWithOsDimension(),
        url=url,
        index_name="foo",
        node_label="Test",
        embedding_node_property="vector",
        text_node_property="info",
        pre_delete_collection=True,
    )

    existing_index = AgensgraphVector.from_existing_index(
        embedding=FakeEmbeddingsWithOsDimension(),
        url=url,
        index_name="foo",
        text_node_property="info",
    )

    output = existing_index.similarity_search("bar", k=1)
    assert _no_id(output) == [Document(page_content="bar", metadata={})]
    drop_vector_indexes(existing_index)

def test_agensgraph_vector_hybrid() -> None:
    """Test end to end construction with hybrid search."""
    text_embeddings = FakeEmbeddingsWithOsDimension().embed_documents(texts)
    text_embedding_pairs = list(zip(texts, text_embeddings))
    docsearch = AgensgraphVector.from_embeddings(
        text_embeddings=text_embedding_pairs,
        embedding=FakeEmbeddingsWithOsDimension(),
        url=url,
        pre_delete_collection=True,
        search_type=SearchType.HYBRID,
    )
    output = docsearch.similarity_search("foo", k=1)
    assert _no_id(output) == [Document(page_content="foo")]

    drop_vector_indexes(docsearch)


def test_agensgraph_vector_hybrid_deduplicate() -> None:
    """Test result deduplication with hybrid search."""
    text_embeddings = FakeEmbeddingsWithOsDimension().embed_documents(texts)
    text_embedding_pairs = list(zip(texts, text_embeddings))
    docsearch = AgensgraphVector.from_embeddings(
        text_embeddings=text_embedding_pairs,
        embedding=FakeEmbeddingsWithOsDimension(),
        url=url,
        pre_delete_collection=True,
        search_type=SearchType.HYBRID,
    )
    output = docsearch.similarity_search("foo", k=3)
    assert _no_id(output) == [
        Document(page_content="foo"),
        Document(page_content="bar"),
        Document(page_content="baz"),
    ]

    drop_vector_indexes(docsearch)


def test_agensgraph_vector_hybrid_retrieval_query() -> None:
    """Test custom retrieval_query with hybrid search."""
    text_embeddings = FakeEmbeddingsWithOsDimension().embed_documents(texts)
    text_embedding_pairs = list(zip(texts, text_embeddings))
    docsearch = AgensgraphVector.from_embeddings(
        text_embeddings=text_embedding_pairs,
        embedding=FakeEmbeddingsWithOsDimension(),
        url=url,
        pre_delete_collection=True,
        search_type=SearchType.HYBRID,
        retrieval_query="RETURN 'moo' AS text, score, {{test: 'test'}} AS metadata",
    )
    output = docsearch.similarity_search("foo", k=1)
    assert _no_id(output) == [Document(page_content="moo", metadata={"test": "test"})]

    drop_vector_indexes(docsearch)


def test_agensgraph_vector_hybrid_retrieval_query2() -> None:
    """Test custom retrieval_query with hybrid search."""
    text_embeddings = FakeEmbeddingsWithOsDimension().embed_documents(texts)
    text_embedding_pairs = list(zip(texts, text_embeddings))
    docsearch = AgensgraphVector.from_embeddings(
        text_embeddings=text_embedding_pairs,
        embedding=FakeEmbeddingsWithOsDimension(),
        url=url,
        pre_delete_collection=True,
        search_type=SearchType.HYBRID,
        retrieval_query="RETURN node.text AS text, score, {{test: 'test'}} AS metadata",
    )
    output = docsearch.similarity_search("foo", k=1)
    assert _no_id(output) == [Document(page_content="foo", metadata={"test": "test"})]

    drop_vector_indexes(docsearch)


def test_agensgraph_vector_missing_keyword() -> None:
    """Test hybrid search with missing keyword_index_search."""
    text_embeddings = FakeEmbeddingsWithOsDimension().embed_documents(texts)
    text_embedding_pairs = list(zip(texts, text_embeddings))
    docsearch = AgensgraphVector.from_embeddings(
        text_embeddings=text_embedding_pairs,
        embedding=FakeEmbeddingsWithOsDimension(),
        url=url,
        pre_delete_collection=True,
    )
    try:
        AgensgraphVector.from_existing_index(
            embedding=FakeEmbeddingsWithOsDimension(),
            url=url,
            index_name="vector",
            search_type=SearchType.HYBRID,
        )
    except ValueError as e:
        assert str(e) == (
            "keyword_index name has to be specified when using hybrid search option"
        )
    drop_vector_indexes(docsearch)


def test_agensgraph_vector_hybrid_from_existing() -> None:
    """Test hybrid search with missing keyword_index_search."""
    text_embeddings = FakeEmbeddingsWithOsDimension().embed_documents(texts)
    text_embedding_pairs = list(zip(texts, text_embeddings))
    AgensgraphVector.from_embeddings(
        text_embeddings=text_embedding_pairs,
        embedding=FakeEmbeddingsWithOsDimension(),
        url=url,
        pre_delete_collection=True,
        search_type=SearchType.HYBRID,
    )
    existing = AgensgraphVector.from_existing_index(
        embedding=FakeEmbeddingsWithOsDimension(),
        url=url,
        index_name="vector",
        keyword_index_name="keyword",
        search_type=SearchType.HYBRID,
    )

    output = existing.similarity_search("foo", k=1)
    assert _no_id(output) == [Document(page_content="foo")]

    drop_vector_indexes(existing)


def test_agensgraph_vector_from_existing_graph() -> None:
    """Test from_existing_graph with a single property."""
    graph = AgensgraphVector.from_texts(
        texts=["test"],
        embedding=FakeEmbeddingsWithOsDimension(),
        url=url,
        index_name="foo",
        node_label="Foo",
        embedding_node_property="vector",
        text_node_property="info",
        pre_delete_collection=True,
    )

    graph.query("MATCH (n) DETACH DELETE n")

    graph.query("""CREATE (:"Test" {name:'Foo'}),(:"Test" {name:'Bar'})""")

    existing = AgensgraphVector.from_existing_graph(
        embedding=FakeEmbeddingsWithOsDimension(),
        url=url,
        index_name="vector",
        node_label="Test",
        text_node_properties=["name"],
        embedding_node_property="embedding",
    )

    output = existing.similarity_search("foo", k=1)
    assert _no_id(output) == [Document(page_content="\nname: Foo")]

    drop_vector_indexes(existing)


def test_agensgraph_vector_from_existing_graph_hybrid() -> None:
    """Test from_existing_graph hybrid with a single property."""
    graph = AgensgraphVector.from_texts(
        texts=["test"],
        embedding=FakeEmbeddingsWithOsDimension(),
        url=url,
        index_name="foo",
        node_label="Foo",
        embedding_node_property="vector",
        text_node_property="info",
        pre_delete_collection=True,
    )

    graph.query("MATCH (n) DETACH DELETE n")

    graph.query("""CREATE (:"Test" {name:'foo'}),(:"Test" {name:'Bar'})""")

    existing = AgensgraphVector.from_existing_graph(
        embedding=FakeEmbeddingsWithOsDimension(),
        url=url,
        index_name="vector",
        node_label="Test",
        text_node_properties=["name"],
        embedding_node_property="embedding",
        search_type=SearchType.HYBRID,
    )

    output = existing.similarity_search("foo", k=1)
    assert _no_id(output) == [Document(page_content="\nname: foo")]

    drop_vector_indexes(existing)


def test_agensgraph_vector_from_existing_graph_multiple_properties() -> None:
    """Test from_existing_graph with a two property."""
    graph = AgensgraphVector.from_texts(
        texts=["test"],
        embedding=FakeEmbeddingsWithOsDimension(),
        url=url,
        index_name="foo",
        node_label="Foo",
        embedding_node_property="vector",
        text_node_property="info",
        pre_delete_collection=True,
    )
    graph.query("MATCH (n) DETACH DELETE n")

    graph.query("""CREATE (:"Test" {name:'Foo', name2: 'Fooz'}),(:"Test" {name:'Bar'})""")

    existing = AgensgraphVector.from_existing_graph(
        embedding=FakeEmbeddingsWithOsDimension(),
        url=url,
        index_name="vector",
        node_label="Test",
        text_node_properties=["name", "name2"],
        embedding_node_property="embedding",
    )

    output = existing.similarity_search("foo", k=1)
    assert _no_id(output) == [Document(page_content="\nname: Foo\nname2: Fooz")]

    drop_vector_indexes(existing)


def test_agensgraph_vector_from_existing_graph_multiple_properties_hybrid() -> None:
    """Test from_existing_graph with a two property."""
    graph = AgensgraphVector.from_texts(
        texts=["test"],
        embedding=FakeEmbeddingsWithOsDimension(),
        url=url,
        index_name="foo",
        node_label="Foo",
        embedding_node_property="vector",
        text_node_property="info",
        pre_delete_collection=True,
    )
    graph.query("MATCH (n) DETACH DELETE n")

    graph.query("""CREATE (:"Test" {name:'Foo', name2: 'Fooz'}),(:"Test" {name:'Bar'})""")

    existing = AgensgraphVector.from_existing_graph(
        embedding=FakeEmbeddingsWithOsDimension(),
        url=url,
        index_name="vector",
        node_label="Test",
        text_node_properties=["name", "name2"],
        embedding_node_property="embedding",
        search_type=SearchType.HYBRID,
    )

    output = existing.similarity_search("foo", k=1)
    assert _no_id(output) == [Document(page_content="\nname: Foo\nname2: Fooz")]

    drop_vector_indexes(existing)


def test_agensgraph_vector_special_character() -> None:
    """Test removing lucene."""
    text_embeddings = FakeEmbeddingsWithOsDimension().embed_documents(texts)
    text_embedding_pairs = list(zip(texts, text_embeddings))
    docsearch = AgensgraphVector.from_embeddings(
        text_embeddings=text_embedding_pairs,
        embedding=FakeEmbeddingsWithOsDimension(),
        url=url,
        pre_delete_collection=True,
        search_type=SearchType.HYBRID,
    )
    output = docsearch.similarity_search(
        "It is the end of the world. Take shelter!", k=1
    )
    assert _no_id(output) == [
        Document(page_content="It is the end of the world. Take shelter!", metadata={})
    ]

    drop_vector_indexes(docsearch)

def test_index_fetching() -> None:
    """testing correct index creation and fetching"""
    embeddings = FakeEmbeddings()

    def create_store(
        node_label: str, index: str, text_properties: List[str]
    ) -> AgensgraphVector:
        return AgensgraphVector.from_existing_graph(
            embedding=embeddings,
            url=url,
            index_name=index,
            node_label=node_label,
            text_node_properties=text_properties,
            embedding_node_property="embedding",
        )

    def fetch_store(index_name: str) -> AgensgraphVector:
        store = AgensgraphVector.from_existing_index(
            embedding=embeddings,
            url=url,
            index_name=index_name,
        )
        return store

    # create index 0
    index_0_str = "index0"
    create_store("label0", index_0_str, ["text"])

    # create index 1
    index_1_str = "index1"
    create_store("label1", index_1_str, ["text"])

    index_1_store = fetch_store(index_1_str)
    assert index_1_store.index_name == index_1_str

    index_0_store = fetch_store(index_0_str)
    assert index_0_store.index_name == index_0_str
    drop_vector_indexes(index_1_store)
    drop_vector_indexes(index_0_store)


def test_retrieval_params() -> None:
    """Test if we use parameters in retrieval query"""
    docsearch = AgensgraphVector.from_texts(
        texts=texts,
        embedding=FakeEmbeddings(),
        pre_delete_collection=True,
        retrieval_query="""
        RETURN %(test)s as text, score, {{test: %(test1)s}} AS metadata
        """,
        url=url,
    )

    # Passed plainly. Before the driver, a string parameter reached the server with no
    # declared type and the Cypher parser *parsed* it as JSON, so every value had to be
    # `json.dumps`-ed first -- and a value that happened to read as JSON, like an order
    # number or a postcode, silently matched nothing. A string is now sent as text.
    output = docsearch.similarity_search(
        "Foo", k=2, params={"test": "test", "test1": "test1"}
    )
    assert _no_id(output) == [
        Document(page_content="test", metadata={"test": "test1"}),
        Document(page_content="test", metadata={"test": "test1"}),
    ]
    drop_vector_indexes(docsearch)


def test_retrieval_dictionary() -> None:
    """Test if we use parameters in retrieval query"""
    docsearch = AgensgraphVector.from_texts(
        texts=texts,
        embedding=FakeEmbeddings(),
        pre_delete_collection=True,
        retrieval_query="""
        RETURN {{
            name:'John', 
            age: 30,
            skills: ['Python', 'Data Analysis', 'Machine Learning']}} as text, 
            score, {{}} AS metadata
        """,
        url=url,
    )
    expected_output = [
        Document(
            page_content=(
                "skills:\n- Python\n- Data Analysis\n- "
                "Machine Learning\nage: 30\nname: John\n"
            )
        )
    ]

    output = docsearch.similarity_search("Foo", k=1)

    def parse_document(doc: Document) -> Any:
        return safe_load(doc.page_content)

    parsed_expected = [parse_document(doc) for doc in expected_output]
    parsed_output = [parse_document(doc) for doc in output]

    assert parsed_output == parsed_expected
    drop_vector_indexes(docsearch)


def test_metadata_filters_type1() -> None:
    """Test metadata filters"""
    docsearch = AgensgraphVector.from_documents(
        DOCUMENTS,
        embedding=FakeEmbeddings(),
        pre_delete_collection=True,
        url=url,
    )
    # We don't test type 5, because LIKE has very SQL specific examples
    for example in (
        TYPE_1_FILTERING_TEST_CASES
        + TYPE_2_FILTERING_TEST_CASES
        + TYPE_3_FILTERING_TEST_CASES
        + TYPE_4_FILTERING_TEST_CASES
    ):
        filter_dict = cast(Dict[str, Any], example[0])
        output = docsearch.similarity_search("Foo", filter=filter_dict)
        indices = cast(List[int], example[1])
        adjusted_indices = [index - 1 for index in indices]
        expected_output = [DOCUMENTS[index] for index in adjusted_indices]
        # 0.2.0 separated system id (__id__) from user metadata, so user's
        # metadata["id"] now round-trips. Only strip None-valued keys.
        for doc in expected_output:
            keys_with_none = [
                key for key, value in doc.metadata.items() if value is None
            ]
            for key in keys_with_none:
                del doc.metadata[key]

        assert _no_id(output) == expected_output
    drop_vector_indexes(docsearch)


def test_agensgraph_vector_relationship_index() -> None:
    """Test end to end construction and search."""
    embeddings = FakeEmbeddingsWithOsDimension()
    docsearch = AgensgraphVector.from_texts(
        texts=texts,
        embedding=embeddings,
        url=url,
        pre_delete_collection=True,
    )
    # Ingest data
    docsearch.query(
        (
            """CREATE ()-[:"REL" {text: 'foo', embedding: %(e1)s}]->()"""
            """, ()-[:"REL" {text: 'far', embedding: %(e2)s}]->()"""
        ),
        params={
            "e1": embeddings.embed_query("foo"),
            "e2": embeddings.embed_query("bar"),
        },
    )
    # Create relationship index
    docsearch.query(
        """CREATE PROPERTY INDEX "relationship"
           ON "REL" USING hnsw ((embedding::vector(1536)) vector_cosine_ops)
        """
    )
    relationship_index = AgensgraphVector.from_existing_relationship_index(
        embeddings, index_name="relationship", url=url
    )

    output = relationship_index.similarity_search("foo", k=1)
    assert _no_id(output) == [Document(page_content="foo")]

    drop_vector_indexes(docsearch)


def test_agensgraph_vector_relationship_index_retrieval() -> None:
    """Test end to end construction and search."""
    embeddings = FakeEmbeddingsWithOsDimension()
    docsearch = AgensgraphVector.from_texts(
        texts=texts,
        embedding=embeddings,
        url=url,
        pre_delete_collection=True,
    )
    # Ingest data
    docsearch.query(
        (
            """CREATE ({node:'text'})-[:"REL" {text: 'foo', embedding: %(e1)s}]->()"""
            """, ({node:'text'})-[:"REL" {text: 'far', embedding: %(e2)s}]->()"""
        ),
        params={
            "e1": embeddings.embed_query("foo"),
            "e2": embeddings.embed_query("bar"),
        },
    )
    # Create relationship index
    docsearch.query(
        """CREATE PROPERTY INDEX "relationship"
           ON "REL" USING hnsw ((embedding::vector(1536)) vector_cosine_ops)
        """
    )
    retrieval_query = (
        "RETURN relationship.text + '-' + startNode(relationship).node "
        "AS text, score, {{foo:'bar'}} AS metadata"
    )
    relationship_index = AgensgraphVector.from_existing_relationship_index(
        embeddings, index_name="relationship", retrieval_query=retrieval_query, url=url
    )

    output = relationship_index.similarity_search("foo", k=1)
    assert _no_id(output) == [Document(page_content="foo-text", metadata={"foo": "bar"})]

    drop_vector_indexes(docsearch)


def test_agensgraph_max_marginal_relevance_search() -> None:
    """
    Test end to end construction and MMR search.
    The embedding function used here ensures `texts` become
    the following vectors on a circle (numbered v0 through v3):

           ______ v2
          /      \
         /        |  v1
    v3  |     .    | query
         |        /  v0
          |______/                 (N.B. very crude drawing)

    With fetch_k==3 and k==2, when query is at (1, ),
    one expects that v2 and v0 are returned (in some order).
    """
    texts = ["-0.124", "+0.127", "+0.25", "+1.0"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = AgensgraphVector.from_texts(
        texts,
        metadatas=metadatas,
        embedding=AngularTwoDimensionalEmbeddings(),
        pre_delete_collection=True,
        url=url,
    )

    expected_set = {
        ("+0.25", 2),
        ("-0.124", 0),
    }

    output = docsearch.max_marginal_relevance_search("0.0", k=2, fetch_k=3)
    output_set = {
        (mmr_doc.page_content, mmr_doc.metadata["page"]) for mmr_doc in output
    }
    assert output_set == expected_set

    drop_vector_indexes(docsearch)


def test_agensgraph_vector_passing_graph_object() -> None:
    """Test end to end construction and search with passing graph object."""
    graph = AgensGraph(conf=conf, graph_name="test", create=True)
    # Rewrite env vars to make sure it fails if env is used
    os.environ["AGENSGRAPH_URI"] = "foo"
    docsearch = AgensgraphVector.from_texts(
        texts=texts,
        embedding=FakeEmbeddingsWithOsDimension(),
        graph=graph,
        pre_delete_collection=True,
        url=url,
    )
    output = docsearch.similarity_search("foo", k=1)
    assert _no_id(output) == [Document(page_content="foo")]

    drop_vector_indexes(docsearch)


class TestABitColumnIsUsable:
    """``HAMMING`` and ``JACCARD`` measure a ``bit`` column, and both are reachable.

    A bit column holds a string of ones and zeros, so an embedding bound for one is
    binary quantised into that string. Written as a list it would be cast from its printed
    form and read as a bit string of that whole length.
    """

    class BitEmbeddings:
        """A deterministic 0/1 embedding, which is what a bit column holds."""

        dims = 12

        def _vec(self, text: str) -> List[float]:
            digest = int(md5(text.encode()).hexdigest(), 16)
            return [float((digest >> i) & 1) for i in range(self.dims)]

        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            return [self._vec(t) for t in texts]

        def embed_query(self, text: str) -> List[float]:
            return self._vec(text)

    @pytest.mark.parametrize(
        "strategy", [DistanceStrategy.HAMMING, DistanceStrategy.JACCARD]
    )
    def test_a_bit_distance_indexes_ingests_and_searches(self, strategy) -> None:
        store = AgensgraphVector.from_texts(
            texts=["alpha", "beta", "gamma"],
            embedding=self.BitEmbeddings(),
            url=url,
            graph_name="vector_it",
            node_label="BitChunk",
            index_name=f"bit_{strategy.value.lower()}",
            distance_strategy=strategy,
            vector_type="bit",
            pre_delete_collection=True,
        )
        try:
            stored = store.query(
                'MATCH (n:"BitChunk") RETURN n.embedding AS e LIMIT 1'
            )[0]["e"]
            assert set(stored) <= {"0", "1"} and len(stored) == 12
            assert store.similarity_search("beta", k=1)[0].page_content == "beta"
        finally:
            drop_vector_indexes(store)


class TestMetadataDoesNotOverwriteWhatTheStoreWrites:
    """The text, the embedding and the id are written after a caller's metadata.

    A metadata key of the same name would otherwise land on top of them -- and an
    embedding replaced by a string is refused by the vector index, so the write fails
    outright rather than quietly storing the wrong thing.
    """

    def test_reserved_keys_in_metadata(self) -> None:
        store = AgensgraphVector.from_texts(
            texts=["the quick brown fox"],
            embedding=FakeEmbeddings(),
            metadatas=[
                {"embedding": "clobbered", "text": "clobbered", "__id__": "clobbered"}
            ],
            url=url,
            graph_name="vector_it",
            node_label="ReservedChunk",
            index_name="reserved",
            pre_delete_collection=True,
        )
        try:
            row = store.query(
                'MATCH (n:"ReservedChunk") RETURN n.text AS text, '
                "n.__id__ AS id, array_size(n.embedding) AS dims"
            )[0]
            assert row["text"] == "the quick brown fox"
            assert row["id"] != "clobbered"
            assert row["dims"] == len(FakeEmbeddings().embed_query("x"))
        finally:
            drop_vector_indexes(store)


class TestAMissingPropertyAnswersTheSame:
    """``$ne`` and ``$nin`` are the same question about an element that lacks the key.

    An absent property is unequal to anything, so it satisfies both.
    """

    @pytest.mark.parametrize(
        "flt",
        [{"kind": {"$ne": "a"}}, {"kind": {"$nin": ["a"]}}],
    )
    def test_an_element_without_the_property(self, flt) -> None:
        store = AgensgraphVector.from_texts(
            texts=["has it", "missing it"],
            embedding=FakeEmbeddings(),
            metadatas=[{"kind": "a"}, {}],
            url=url,
            graph_name="vector_it",
            node_label="MissingChunk",
            index_name="missing",
            pre_delete_collection=True,
        )
        try:
            found = store.similarity_search("has it", k=10, filter=flt)
            assert [d.page_content for d in found] == ["missing it"]
        finally:
            drop_vector_indexes(store)


class TestAGraphThatWasAlreadyThere:
    """What ``from_existing_graph`` has to get right for a graph it did not create."""

    @staticmethod
    def _seed() -> AgensgraphVector:
        graph = AgensGraph("vector_existing_it", conf, create=True)
        graph.query("MATCH (n) DETACH DELETE n")
        for title, content in [
            ("quantum computing", "qubits and gates"),
            ("cooking pasta", "boil water add salt"),
            ("quantum mechanics", "wave functions"),
        ]:
            graph.query(
                'CREATE (:"Doc" {"title": %(t)s, "content": %(c)s})',
                {"t": title, "c": content},
            )
        graph.close()
        return AgensgraphVector.from_existing_graph(
            embedding=FakeEmbeddings(),
            node_label="Doc",
            embedding_node_property="embedding",
            text_node_properties=["title", "content"],
            url=url,
            graph_name="vector_existing_it",
            index_name="existing_vector",
        )

    def test_a_document_has_an_id_and_can_be_read_back_by_it(self) -> None:
        store = self._seed()
        try:
            hits = store.similarity_search("quantum", k=3)
            assert all(doc.id for doc in hits)
            fetched = store.get_by_ids([hits[0].id])
            assert fetched and fetched[0].id == hits[0].id
        finally:
            drop_vector_indexes(store)

    def test_maximal_marginal_relevance_reads_the_embeddings_it_needs(self) -> None:
        """The retrieval query shapes the document and carries no embedding."""
        store = self._seed()
        try:
            picked = store.max_marginal_relevance_search("quantum", k=2, fetch_k=3)
            assert len(picked) == 2
            assert all("_embedding_" not in doc.metadata for doc in picked)
        finally:
            drop_vector_indexes(store)

    @pytest.mark.asyncio
    async def test_the_async_path_agrees(self) -> None:
        store = self._seed()
        try:
            picked = await store.amax_marginal_relevance_search(
                "quantum", k=2, fetch_k=3
            )
            assert len(picked) == 2
        finally:
            drop_vector_indexes(store)


class TestHybridScoresBothHalves:
    """A hybrid search fuses two rankings, and an element has to be found in both.

    The halves are joined on the element's own identity. ``__id__`` is a property only
    ``add_embeddings`` writes, so over a graph that was already there it was null on both
    sides, nothing joined, and every hit was scored by one half of the fusion.
    """

    def test_an_element_matched_by_both_scores_more_than_one_matched_by_one(self) -> None:
        graph = AgensGraph("vector_hybrid_it", conf, create=True)
        graph.query("MATCH (n) DETACH DELETE n")
        for title, content in [
            ("quantum computing", "qubits and gates"),
            ("cooking pasta", "boil water add salt"),
        ]:
            graph.query(
                'CREATE (:"Doc" {"title": %(t)s, "content": %(c)s})',
                {"t": title, "c": content},
            )
        graph.close()
        store = AgensgraphVector.from_existing_graph(
            embedding=FakeEmbeddings(),
            node_label="Doc",
            embedding_node_property="embedding",
            text_node_properties=["title", "content"],
            search_type=SearchType.HYBRID,
            url=url,
            graph_name="vector_hybrid_it",
            index_name="hybrid_vector",
            keyword_index_name="hybrid_keyword",
        )
        try:
            hits = store.similarity_search_with_score("quantum computing", k=5)
            scores = sorted((score for _, score in hits), reverse=True)
            # Two terms of the fusion against one: the best hit is matched by the
            # keyword half as well, so it scores about twice the one that is not.
            assert scores[0] > scores[-1] * 1.8
        finally:
            drop_vector_indexes(store)
            drop_fulltext_indexes(store)

    def test_the_keyword_half_reads_every_property_it_indexed(self) -> None:
        """A term only in the second text property still matches."""
        graph = AgensGraph("vector_hybrid_it", conf, create=True)
        graph.query("MATCH (n) DETACH DELETE n")
        graph.query(
            'CREATE (:"Doc" {"title": %(t)s, "content": %(c)s})',
            {"t": "an unremarkable heading", "c": "supercalifragilistic contents"},
        )
        graph.close()
        store = AgensgraphVector.from_existing_graph(
            embedding=FakeEmbeddings(),
            node_label="Doc",
            embedding_node_property="embedding",
            text_node_properties=["title", "content"],
            search_type=SearchType.HYBRID,
            url=url,
            graph_name="vector_hybrid_it",
            index_name="hybrid_vector2",
            keyword_index_name="hybrid_keyword2",
        )
        try:
            match, _ = store._keyword_expressions()
            rendered = match.as_string(store.connection)
            assert '"title"' in rendered and '"content"' in rendered
            hits = store.similarity_search_with_score("supercalifragilistic", k=1)
            # One document, matched by both halves, so both terms of the fusion.
            assert hits and hits[0][1] > 1.0 / 61
        finally:
            drop_vector_indexes(store)
            drop_fulltext_indexes(store)


class TestTheStoreInstallsNothing:
    """Building a store adds no function of ours to somebody else's database.

    What an index covers is read from the definition the server prints for it. Asking the
    question with a function of our own meant creating one on every construction, into
    whatever ``search_path`` happened to resolve to.
    """

    @staticmethod
    def _function_count(store: AgensgraphVector) -> int:
        return int(
            store.query("SELECT count(*) AS c FROM pg_catalog.pg_proc")[0]["c"]
        )

    def test_pg_proc_is_the_same_size_afterwards(self) -> None:
        graph = AgensGraph("vector_install_it", conf, create=True)
        graph.query("MATCH (n) DETACH DELETE n")
        # A database may already hold such functions from elsewhere. They are dropped
        # first, so that what is asserted is whether building a store creates any.
        for leftover in graph.query(
            "SELECT p.oid::regprocedure AS signature FROM pg_catalog.pg_proc p "
            "WHERE p.proname LIKE 'ag\\_list%'"
        ):
            graph.query(f"DROP FUNCTION IF EXISTS {leftover['signature']}")
        before = int(
            graph.query("SELECT count(*) AS c FROM pg_catalog.pg_proc")[0]["c"]
        )
        store = AgensgraphVector.from_texts(
            texts=["alpha", "beta"],
            embedding=FakeEmbeddings(),
            graph=graph,
            node_label="InstallChunk",
            index_name="install_vec",
            keyword_index_name="install_kw",
            search_type=SearchType.HYBRID,
            pre_delete_collection=True,
        )
        try:
            assert store.similarity_search("alpha", k=1)
            assert self._function_count(store) == before
            named = store.query(
                "SELECT count(*) AS c FROM pg_catalog.pg_proc "
                "WHERE proname LIKE 'ag\\_list%'"
            )
            assert int(named[0]["c"]) == 0
        finally:
            drop_vector_indexes(store)
            drop_fulltext_indexes(store)
            graph.close()

    def test_the_driver_still_knows_which_graph_is_selected(self) -> None:
        """A raw ``SET graph_path`` would leave it unable to read the catalogs."""
        graph = AgensGraph("vector_install_it", conf, create=True)
        store = AgensgraphVector.from_texts(
            texts=["alpha"],
            embedding=FakeEmbeddings(),
            graph=graph,
            node_label="SelectedChunk",
            index_name="selected_vec",
            pre_delete_collection=True,
        )
        try:
            assert graph.connection.label_table.graph == "vector_install_it"
            assert graph.connection.indexes()  # would raise if none were selected
        finally:
            drop_vector_indexes(store)
            graph.close()


class TestNamingElementsByIdReachesTheIndex:
    """The perf contract is only real if a test enforces it.

    A bound list is compared by containment -- the list is asked whether it holds the
    property -- and containment is not a comparison the unique index on the id can
    answer, so the label is read whole. The rows come back either way, so only the plan
    shows it.

    Enforced over enough elements for the choice to matter. Below a few thousand, reading
    the label once and joining is genuinely the cheaper plan and the server picks it,
    which says nothing about whether the index can be used at a size where it counts.
    """

    ROWS = 20000

    @pytest.fixture
    def loaded(self):
        graph = AgensGraph("vector_plan_it", conf, create=True)
        held = graph.query('MATCH (n:"PlanChunk") RETURN count(*) AS c')[0]["c"]
        store = AgensgraphVector(
            FakeEmbeddings(),
            graph=graph,
            node_label="PlanChunk",
            index_name="plan_vec",
        )
        if held < self.ROWS:
            graph.query("MATCH (n) DETACH DELETE n")
            store.add_texts(
                [f"chunk {i}" for i in range(self.ROWS)],
                ids=[f"id{i}" for i in range(self.ROWS)],
                batch_size=2500,
            )
            graph.query(f'ANALYZE "{graph.graph_name}"."PlanChunk"')
        yield store
        graph.close()

    def _plan(self, store, statement, params) -> str:
        rows = store.query(
            "EXPLAIN (COSTS OFF) " + statement.as_string(store.connection), params
        )
        return "\n".join(str(next(iter(r.values()))) for r in rows)

    def test_reading_by_id_is_an_index_scan(self, loaded) -> None:
        params: dict = {}
        statement = sql.SQL("{by} RETURN n.__id__ AS id").format(
            by=loaded._named_by_id([f"id{i}" for i in range(20)], params)
        )
        plan = self._plan(loaded, statement, params)
        assert "Index Scan" in plan
        assert "Seq Scan" not in plan

    def test_reading_many_by_id_does_not_grow_the_statement(self, loaded) -> None:
        """One probe repeated, not one term per id.

        Written out as a term each, the statement grows with the list and the planner pays
        for every term -- and with no index to answer them, each is a pass over the label.
        """
        params: dict = {}
        few = loaded._named_by_id(["id1"], params).as_string(loaded.connection)
        many = loaded._named_by_id(
            [f"id{i}" for i in range(200)], {}
        ).as_string(loaded.connection)
        assert few == many, "the statement says the same thing however many ids"

    def test_get_by_ids_returns_what_was_asked_for(self, loaded) -> None:
        wanted = [f"id{i}" for i in range(0, self.ROWS, self.ROWS // 20)][:20]
        found = loaded.get_by_ids(wanted)
        assert sorted(d.id for d in found) == sorted(wanted)

    def test_delete_removes_only_those(self, loaded) -> None:
        before = loaded.query('MATCH (n:"PlanChunk") RETURN count(*) AS c')[0]["c"]
        loaded.delete(["id1", "id2", "id3"])
        after = loaded.query('MATCH (n:"PlanChunk") RETURN count(*) AS c')[0]["c"]
        assert int(before) - int(after) == 3
        loaded.add_texts(["chunk 1", "chunk 2", "chunk 3"], ids=["id1", "id2", "id3"])

    @pytest.mark.asyncio
    async def test_the_async_twins_agree(self, loaded) -> None:
        wanted = ["id10", "id11"]
        found = await loaded.aget_by_ids(wanted)
        assert sorted(d.id for d in found) == sorted(wanted)


class TestSearchTuningEndsWithTheSearch:
    """``hnsw.ef_search`` decides how many candidates the index looks at.

    Set for the session it stays on a pooled connection and tunes every later borrower's
    search -- a recall setting one caller asked for, silently applied to everyone who gets
    that connection next, and a cost they did not ask to pay.
    """

    def test_the_connection_is_left_as_it_was_found(self) -> None:
        from langchain_agensgraph.engine import AgensEngine

        # One connection, so the next borrower certainly gets the same one.
        engine = AgensEngine.from_url(url, min_size=1, max_size=1)
        graph = AgensGraph(
            "vector_tuning_it", conf, create=True, engine=engine, refresh_schema=False
        )
        store = AgensgraphVector.from_texts(
            texts=["alpha", "beta"],
            embedding=FakeEmbeddings(),
            graph=graph,
            node_label="TunedChunk",
            index_name="tuned_vec",
            pre_delete_collection=True,
            search_options={"hnsw.ef_search": 200},
        )
        try:
            with engine.connection() as conn:
                default = conn.execute("SHOW hnsw.ef_search").fetchone()[0]

            assert store.similarity_search("alpha", k=1)

            with engine.connection() as conn:
                after = conn.execute("SHOW hnsw.ef_search").fetchone()[0]
            assert after == default, "the tuning outlived the search that asked for it"
        finally:
            drop_vector_indexes(store)
            graph.close()
            engine.close()


class TestBuildingAStoreCostsNoEmbedding:
    """How wide an embedding is comes from the index, not from asking the model.

    Asking is a request to whatever is behind the embedding function -- for a hosted
    model, a wait and a charge -- and it was made on every construction, including one
    that only ever reads.
    """

    class Counting:
        calls = 0

        def embed_documents(self, texts):
            type(self).calls += len(texts)
            return [[0.1] * 10 for _ in texts]

        def embed_query(self, text):
            type(self).calls += 1
            return [0.1] * 10

    @pytest.fixture
    def seeded(self):
        graph = AgensGraph("vector_width_it", conf, create=True)
        graph.query("MATCH (n) DETACH DELETE n")
        store = AgensgraphVector.from_texts(
            texts=["a", "b"],
            embedding=self.Counting(),
            graph=graph,
            node_label="WidthChunk",
            index_name="width_vec",
            pre_delete_collection=True,
        )
        yield store
        drop_vector_indexes(store)
        graph.close()

    def test_building_over_an_existing_index_asks_nothing(self, seeded) -> None:
        self.Counting.calls = 0
        AgensgraphVector.from_existing_index(
            embedding=self.Counting(),
            url=url,
            graph_name="vector_width_it",
            index_name="width_vec",
            node_label="WidthChunk",
        )
        assert self.Counting.calls == 0

    def test_the_width_still_comes_out_right(self, seeded) -> None:
        store = AgensgraphVector.from_existing_index(
            embedding=self.Counting(),
            url=url,
            graph_name="vector_width_it",
            index_name="width_vec",
            node_label="WidthChunk",
        )
        assert store.embedding_dimension == 10

    def test_a_model_of_the_wrong_width_is_refused_on_the_first_write(
        self, seeded
    ) -> None:
        class Wider(self.Counting):
            def embed_documents(self, texts):
                return [[0.1] * 99 for _ in texts]

            def embed_query(self, text):
                return [0.1] * 99

        store = AgensgraphVector.from_existing_index(
            embedding=Wider(),
            url=url,
            graph_name="vector_width_it",
            index_name="width_vec",
            node_label="WidthChunk",
        )
        with pytest.raises(ValueError, match="dimensions do not match"):
            store.add_texts(["one of the wrong width"])


class TestPerCallRetrievalQuery:
    """One store, many shapes: a search names its own retrieval query.

    The override exists so several retrievers can share one store -- and its one
    connection pool -- while each reads a different context. These tests hold the two
    properties that make that safe: the override shapes only the call that carries it,
    and the store's own query is untouched afterwards.
    """

    OVERRIDE = """
        RETURN 'overridden' AS text, score, node.__id__ AS doc_id,
        {{shape: 'override'}} AS metadata
    """

    def test_override_shapes_one_call_only(self) -> None:
        docsearch = AgensgraphVector.from_texts(
            texts=texts,
            embedding=FakeEmbeddings(),
            pre_delete_collection=True,
            url=url,
        )
        shaped = docsearch.similarity_search("foo", k=1, retrieval_query=self.OVERRIDE)
        assert _no_id(shaped) == [
            Document(page_content="overridden", metadata={"shape": "override"})
        ]
        # The next plain search still reads the store's default shape.
        plain = docsearch.similarity_search("foo", k=1)
        assert _no_id(plain) == [Document(page_content="foo")]
        drop_vector_indexes(docsearch)

    def test_override_wins_over_the_stores_own(self) -> None:
        docsearch = AgensgraphVector.from_texts(
            texts=texts,
            embedding=FakeEmbeddings(),
            pre_delete_collection=True,
            retrieval_query="RETURN 'constructor' AS text, score, {{}} AS metadata",
            url=url,
        )
        shaped = docsearch.similarity_search("foo", k=1, retrieval_query=self.OVERRIDE)
        assert shaped[0].page_content == "overridden"
        # And without the override the constructor's query still answers.
        assert docsearch.similarity_search("foo", k=1)[0].page_content == "constructor"
        drop_vector_indexes(docsearch)

    async def test_the_async_path_carries_the_override(self) -> None:
        docsearch = AgensgraphVector.from_texts(
            texts=texts,
            embedding=FakeEmbeddings(),
            pre_delete_collection=True,
            url=url,
        )
        shaped = await docsearch.asimilarity_search(
            "foo", k=1, retrieval_query=self.OVERRIDE
        )
        assert _no_id(shaped) == [
            Document(page_content="overridden", metadata={"shape": "override"})
        ]
        await docsearch.aclose()
        drop_vector_indexes(docsearch)


class TestAsyncHybridConfig:
    """The async twin takes the fusion knobs the blocking one does.

    Before, ``hybrid_config`` reached `_build_search` only by falling through
    ``**kwargs``; naming it is what this test pins, by asserting the two paths fuse
    identically under a knob that is not the default.
    """

    async def test_the_twins_fuse_alike(self) -> None:
        from langchain_agensgraph.vectorstores.agensgraph_vector import (
            HybridSearchConfig,
        )

        docsearch = AgensgraphVector.from_texts(
            texts=texts,
            embedding=FakeEmbeddings(),
            pre_delete_collection=True,
            search_type=SearchType.HYBRID,
            url=url,
        )
        sharp = HybridSearchConfig(rank_constant=1, keyword_weight=2.0)
        wanted = docsearch.similarity_search_with_score(
            "foo", k=3, hybrid_config=sharp
        )
        got = await docsearch.asimilarity_search_with_score(
            "foo", k=3, hybrid_config=sharp
        )
        assert [(d.page_content, s) for d, s in _no_id(wanted)] == [
            (d.page_content, s) for d, s in _no_id(got)
        ]
        await docsearch.aclose()
        drop_fulltext_indexes(docsearch)
        drop_vector_indexes(docsearch)
