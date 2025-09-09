# 🦜️🔗 LangChain AgensGraph

This plugin provides support for GraphStore and VectorStore integration in [langchain](https://www.langchain.com/), for persisting graphs directly in [AgensGraph](https://github.com/skaiworldwide-oss/agensgraph).

See the associated guide below:

- [Agens Graph Store](./examples/agensgraph.ipynb)

## 📦 Installation

```bash
pip install -U langchain-agensgraph
```

## 💻 Usage

### AgensGraph
The AgensGraph class provides functionality to interact with agensgraph as
a graphstore in langchain.

```python
from langchain_agensgraph.graphs.agensgraph import AgensGraph

conf = {
    "dbname": "",
    "user": "",
    "password": "",
    "host": "",
    "port": ,
}

graph = AgensGraph(graph_name="", conf=conf)
graph.query("MATCH (n) RETURN n LIMIT 1;")
```

### AgensgraphVector

The `AgensgraphVector` class provides functionality for managing an AgensGraph vector store.
It enables you to create new vector indexes, add vectors to existing indexes, and perform queries using vector indexes.

```python
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings

from langchain_agensgraph.vectorstores.agensgraph_vector import AgensgraphVector

# Create a vector store from some documents and embeddings
docs = [
    Document(
        page_content=(
            "LangChain is a framework to build "
            "with LLMs by chaining interoperable components."
        ),
    )
]
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key="sk-...",  # Replace with your OpenAI API key
)
db = AgensgraphVector.from_documents(
    docs,
    embeddings,
    url="postgresql://username:password@host:port/dbname"
)
# Query the vector store for similar documents
docs_with_score = db.similarity_search_with_score("What is LangChain?", k=1)
```
