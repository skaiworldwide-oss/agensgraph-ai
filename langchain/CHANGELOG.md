# Changelog

## 0.4.0

Runs on AgensGraph 2.14.1 and later, on `agensgraph-python` 2.1.

### Security

- Model-written Cypher ran in a read-only transaction that `…; COMMIT; …` escaped, so the QA chain
  and the text2cypher retriever could write to, or drop, the graph. It now runs only through
  `AgensGraph.query_untrusted`. The text is checked before it is sent, then run as a role made for
  it (`untrusted_conninfo`), in a read-only transaction that is always rolled back.

### Added

- `AgensEngine` pools connections by kind (application, maintenance, untrusted, writer).
  Components handed one conninfo string share an engine. `engine.transaction()` is a transaction
  every call through the engine joins.
- Agent state is readable from Cypher: `cypher_subquery()` on `AgensChatMessageHistory`,
  `AgensStore` and `AgensSaver` (with `queryable_channels`).
- `langchain_agensgraph.roles`: `render_roles` and `render_grants` print the SQL for the roles the
  package expects.
- `AsyncAgensStore`, a store for an event loop.
- Vector store: hybrid search with `fusion`, `fusion_weights` and `rank_constant`,
  `hybrid_search_with_details`, `delete(filter=)`, `search_with_embeddings`, `build_index`,
  `index_status`, `clear` and `drop`.
- QA chain: `allow_writes`, `return_direct`, `include_types`, `exclude_types`, `use_function_response`,
  `cypher_llm_kwargs`, `qa_llm_kwargs`, `schema="frozen"`, and a correction loop (`max_retries`).
- `LLMGraphTransformer`: relationship triples, `relationship_properties`, property whitelists,
  `additional_instructions`, `ignore_tool_usage`, `max_concurrency` and `on_error`.

### Removed

| removed | use instead |
|---|---|
| `AgensEngine.from_conf`, `conf=` | an engine or a conninfo string |
| vector store `url=`, `AGENSGRAPH_URL` | `engine=` |
| `AgensGraph.read_only`, `aread_only` | `query_untrusted` |
| `allow_server_programs` | `untrusted_conninfo`, and `unsafe_shared_role` on 2.16 and later |
| `AsyncAgensSaver` | `AgensSaver`, which serves async graphs too |
| `pre_delete_collection` | `clear()` |
| `HybridSearchConfig`, `hybrid_config=` | `fusion`, `fusion_weights`, `rank_constant` |
| `_embedding_` in metadata, `query(binary_=)` on the vector store | `search_with_embeddings` |
| `merging`, `register_vectors`, `clean_graph_labels`, `retry_attempts`, `create_id_index`, `effective_search_ratio`, `remove_lucene_chars`, `build_expansion_query` | — |
| `AgensStore(promoted=, label=, vector_schema=)` | — |
| `is_write_query`, `CYPHER_SYSTEM` | `render_cypher_system`, `CYPHER_PROMPT` |

### Renamed

| 0.3 | 0.4 |
|---|---|
| `AgensGraph("name", conf)` | `AgensGraph(engine, graph_name="name")` |
| `create=` | `create_graph=` |
| `create_labels` | `ensure_labels` |
| `AgensQueryException` | `AgensQueryError` |
| `node_label`, `text_node_property`, `embedding_node_property`, `embedding_dimension` | `label`, `text_property`, `embedding_property`, `dims` |
| `text_node_properties` | `KeywordConfig(properties=)` |
| `create_new_index` | `build_index` |
| `filter_properties` | `filter_keys` |
| retriever `store=`, `expand_by_hops` | `vectorstore=`, `hops` |
| the QA prompt's `{results}` | `{context}` |

### Changed

- `graph_name` is required everywhere; `create_graph` is False everywhere.
- `AgensGraph` reads its schema on first use and keeps it for 60 seconds. Its counts are
  estimates.
- `*_with_score` returns the raw distance. A hybrid search returns the fused score. The relevance
  of an inner-product distance is `(1 + inner product) / 2`, which keeps the order of nearness.
- Vector store ids default to a uuid4, metadata drops `None` values, and the store's reserved keys
  are refused. `delete()` with no ids raises.
- `allow_dangerous_requests=True` is a required acknowledgement; writes need `allow_writes=True`.
- Chat history, checkpoints and the store live in plain tables in `"<graph>_store"`.
- A store search without a query returns the newest items first, and an empty prefix searches
  every namespace, as `InMemoryStore` does.
- With `include_source=True`, a source document's vertex carries its `text`.
- Vector store filters take `$exists`.
- `LLMGraphTransformer` raises a failed document by default (`on_error="raise"`).
- `langgraph` is an extra: `pip install "langchain-agensgraph[langgraph]"`.
- `log_queries` is scoped to the calling context.
- pgvector 0.8.0 or later.
