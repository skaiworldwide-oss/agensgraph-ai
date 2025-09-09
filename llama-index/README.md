# LlamaIndex AgensGraph

This plugin provides support for AgensGraphStore integration in [llamaindex](https://www.llamaindex.ai/), for persisting graphs directly in [AgensGraph](https://github.com/skaiworldwide-oss/agensgraph). Additionally, we support the PropertyGraphIndex and VectorStore, which allows you to store and query property graphs and vectors in AgensGraph.

- Property Graph Store: `AgensPropertyGraphStore`
- Knowledege Graph Store: `AgensGraphStore`
- Vector Store: `AgensgraphVectorStore`

See the associated guides below:

- [Agens Graph Store](./examples/index_structs/knowledge_graph/AgensgraphDemo.ipynb)
- [Agens Property Graph Store](./examples//property_graph/property_graph_agensgraph.ipynb)
- [Agensgraph Vector Store](./examples/vector_stores/AgensgraphVectorDemo.ipynb)


## Installation

```shell
pip install llama-index llama-index-agensgraph
```

## Usage

### Property Graph Store

```python
import os
import urllib.request
import nest_asyncio
from llama_index.core import SimpleDirectoryReader, PropertyGraphIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor

from llama_index_agensgraph.graph_stores.agensgraph import AgensPropertyGraphStore

os.environ[
    "OPENAI_API_KEY"
] = "<YOUR_API_KEY>"  # Replace with your OpenAI API key

os.makedirs("data/paul_graham/", exist_ok=True)

url = "https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt"
output_path = "data/paul_graham/paul_graham_essay.txt"
urllib.request.urlretrieve(url, output_path)

nest_asyncio.apply()

with open(output_path, "r", encoding="utf-8") as file:
    content = file.read()

modified_content = content.replace("'", "\\'")

with open(output_path, "w", encoding="utf-8") as file:
    file.write(modified_content)

documents = SimpleDirectoryReader("./data/paul_graham/").load_data()

# Setup AgensGraph connection (ensure AgensGraph is running)
conf = {
    "dbname": "",
    "user": "",
    "password": "",
    "host": "",
    "port": "",
}

graph_store = AgensPropertyGraphStore(
    graph_name="graph",
    conf=conf
)

index = PropertyGraphIndex.from_documents(
    documents,
    embed_model=OpenAIEmbedding(model_name="text-embedding-ada-002"),
    kg_extractors=[
        SchemaLLMPathExtractor(
            llm=OpenAI(model="gpt-3.5-turbo", temperature=0.0),
        )
    ],
    property_graph_store=graph_store,
    show_progress=True,
)

query_engine = index.as_query_engine(include_text=True)

response = query_engine.query("What happened at Interleaf and Viaweb?")
print("\nDetailed Query Response:")
print(str(response))
```

### Knowledge Graph Store

```python
import os
import logging
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from llama_index.core import (
    KnowledgeGraphIndex,
    SimpleDirectoryReader,
    StorageContext,
)
from llama_index_agensgraph.graph_stores.agensgraph import AgensGraphStore

os.environ[
    "OPENAI_API_KEY"
] = "<YOUR_API_KEY>"  # Replace with your OpenAI API key

logging.basicConfig(level=logging.INFO)

llm = OpenAI(temperature=0, model="gpt-3.5-turbo")
Settings.llm = llm
Settings.chunk_size = 512

documents = {
    "doc1.txt": "Python is a popular programming language known for its readability and simplicity. It was created by Guido van Rossum and first released in 1991. Python supports multiple programming paradigms, including procedural, object-oriented, and functional programming. It is widely used in web development, data science, artificial intelligence, and scientific computing.",
    "doc2.txt": "JavaScript is a high-level programming language primarily used for web development. It was created by Brendan Eich and first appeared in 1995. JavaScript is a core technology of the World Wide Web, alongside HTML and CSS. It enables interactive web pages and is an essential part of web applications. JavaScript is also used in server-side development with environments like Node.js.",
    "doc3.txt": "Java is a high-level, class-based, object-oriented programming language that is designed to have as few implementation dependencies as possible. It was developed by James Gosling and first released by Sun Microsystems in 1995. Java is widely used for building enterprise-scale applications, mobile applications, and large systems development.",
}

for filename, content in documents.items():
    with open(filename, "w") as file:
        file.write(content)

loaded_documents = SimpleDirectoryReader(".").load_data()

# Setup AgensGraph connection (ensure AgensGraph is running)
conf = {
    "dbname": "",
    "user": "",
    "password": "",
    "host": "",
    "port": "",
}

graph_store = AgensGraphStore(
    graph_name="graph",
    conf=conf
)

storage_context = StorageContext.from_defaults(graph_store=graph_store)

index = KnowledgeGraphIndex.from_documents(
    loaded_documents,
    storage_context=storage_context,
    max_triplets_per_chunk=3,
)

query_engine = index.as_query_engine(
    include_text=False, response_mode="tree_summarize"
)
response = query_engine.query("Tell me about Python and its uses")

print("Query Response:")
print(response)
```

### Vector Store
```python
import os
import urllib.request
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index_agensgraph.vector_stores.agensgraph.base import AgensgraphVectorStore

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "<YOUR_API_KEY>"  # Replace with your key

# Download example data
os.makedirs("data/paul_graham/", exist_ok=True)
url = "https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt"
output_path = "data/paul_graham/paul_graham_essay.txt"
urllib.request.urlretrieve(url, output_path)

# Load documents
documents = SimpleDirectoryReader("./data/paul_graham").load_data()

# Setup AgensGraph connection (ensure AgensGraph is running)
url = "postgresql://username:password@host:port/database_name"
embed_dim = 1536

# Initialize vector store
vector_store = AgensgraphVectorStore(url=url, embedding_dimension=embed_dim)

# Build index
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

# Query
query_engine = index.as_query_engine()
response = query_engine.query("What happened at Interleaf?")
print("\nQuery Response:")
print(str(response))
```