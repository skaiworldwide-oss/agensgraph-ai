"""Test AgensgraphVector functionality."""

import os, json
from math import isclose
from typing import Any, Dict, List, cast

from langchain_core.documents import Document
from yaml import safe_load

from langchain_agensgraph.graphs.agensgraph import AgensGraph
from langchain_agensgraph.vectorstores.agensgraph_vector import (
    AgensgraphVector,
    SearchType,
)
from langchain_community.vectorstores.utils import DistanceStrategy
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
    all_indexes = store.query(
        """
            SELECT name FROM ag_list_vector_indexes()
                              """
    )
    for index in all_indexes:
        store.query(f"""DROP PROPERTY INDEX "{index['name']}" CASCADE""")

    store.query("MATCH (n) DETACH DELETE n;")

def drop_fulltext_indexes(store: AgensgraphVector) -> None:
    """Cleanup all vector indexes"""
    all_indexes = store.query(
        """
            SELECT name FROM ag_list_text_indexes()
                              """
    )
    print("DEBUG - all_indexes", all_indexes)
    for index in all_indexes:
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
    assert output == [Document(page_content="foo")]

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
    assert output == [Document(page_content="foo")]

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
    assert output == [Document(page_content="foo")]

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
    assert output == [Document(page_content="foo")]

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
    assert output == [Document(page_content="foo")]

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
    assert output == [Document(page_content="foo", metadata={"page": "0"})]

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
    assert output == [(Document(page_content="foo", metadata={"page": "0"}), 1.0)]

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
    assert output == [
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
    assert output == [Document(page_content="foo", metadata={"test": "test"})]

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
    assert output == [Document(page_content="bar", metadata={})]
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
    assert output == [Document(page_content="foo")]

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
    assert output == [
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
    assert output == [Document(page_content="moo", metadata={"test": "test"})]

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
    assert output == [Document(page_content="foo", metadata={"test": "test"})]

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
    assert output == [Document(page_content="foo")]

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
    assert output == [Document(page_content="\nname: Foo")]

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
    assert output == [Document(page_content="\nname: foo")]

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
    assert output == [Document(page_content="\nname: Foo\nname2: Fooz")]

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
    assert output == [Document(page_content="\nname: Foo\nname2: Fooz")]

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
    assert output == [
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

    output = docsearch.similarity_search(
        "Foo", k=2, params={"test": json.dumps("test"), "test1": json.dumps("test1")}
    )
    assert output == [
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
        # We don't return id properties from similarity search by default
        # Also remove any key where the value is None
        for doc in expected_output:
            if "id" in doc.metadata:
                del doc.metadata["id"]
            keys_with_none = [
                key for key, value in doc.metadata.items() if value is None
            ]
            for key in keys_with_none:
                del doc.metadata[key]

        assert output == expected_output
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
    assert output == [Document(page_content="foo")]

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
    assert output == [Document(page_content="foo-text", metadata={"foo": "bar"})]

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
    os.environ["NEO4J_URI"] = "foo"
    docsearch = AgensgraphVector.from_texts(
        texts=texts,
        embedding=FakeEmbeddingsWithOsDimension(),
        graph=graph,
        pre_delete_collection=True,
        url=url,
    )
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]

    drop_vector_indexes(docsearch)
