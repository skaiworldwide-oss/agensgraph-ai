# 04 · Code graph — turn a codebase into queryable memory

cognee has a dedicated **code pipeline**: it parses a Python repository into a
graph of files, classes, functions, and imports, stored in AgensGraph
(`cognee_code`). Then you can search the code semantically (`SearchType.CODE`),
inspect the extracted structure, and visualize it.

> Needs the `tree-sitter` + `tree-sitter-python` parsers (`pip install tree-sitter
> tree-sitter-python`) — that's what cognee's code pipeline uses to parse Python.

📓 **Guided tour:** [`code_graph.ipynb`](./code_graph.ipynb).

## Run

```bash
# from cognee/  (clones a small package by default; first run pulls it)
.venv/bin/python examples/demos/04_code_graph/build.py
.venv/bin/python examples/demos/04_code_graph/ask.py
.venv/bin/python examples/demos/04_code_graph/ask.py "how are sessions handled?"
```

Knobs: `CODE_REPO` (path to a local Python package to analyze — skips cloning),
`CODE_REPO_URL` (git URL to shallow-clone, default a small well-known library),
`CODE_RESET=0`.

## The pattern

```python
from cognee.api.v1.cognify.code_graph_pipeline import run_code_graph_pipeline
async for status in run_code_graph_pipeline(repo_path, include_docs=False):
    ...                                              # parses + builds the code graph

await cognee.search(query_text="send a request", query_type=SearchType.CODE)   # find the right files
nodes, _ = await (await get_graph_engine()).get_graph_data()                   # FunctionDefinition / ClassDefinition / ImportStatement / CodeFile
await cognee.visualize_graph("code_graph.html")
```

## What you get

A semantic + structural map of a codebase — searchable and traversable — in the
same AgensGraph-backed cognee memory you use for documents.
