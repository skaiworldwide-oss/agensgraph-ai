# Text2Cypher eval — the AgensGraph dialect, measured

Models write Neo4j-flavored Cypher; this engine diverges from that habit in ways
that are silent as often as loud. This eval turns "how well does text2cypher
work here" into a number: a dataset of question/gold-query pairs concentrated on
the divergences, scored by **execution** — the model's query runs, the gold
query runs, and the result sets must agree values-only. A query that reads
plausibly but returns the wrong rows scores zero.

## Layout

- `dataset.jsonl` — 94 entries. Each has a `question`, a dialect-correct `gold`
  query that `query_untrusted` runs, a `category`, the driver capabilities it
  `requires`, and — for the traps — the `habit` query a model writes elsewhere,
  with its declared `habit_outcome` (`error`, `different` from gold, or
  `unindexed`). Gold results are never stored; they are computed at run time
  against deterministic fixture graphs.
- `t2c.py` — the dataset loader, the values-only result canon, and the builders
  for the two fixture graphs: `t2c_movies` (a small catalog with a mixed-case
  label) and `t2c_traps` (a quoted upper-case relationship type, a stringly
  number, a list holding a null, an inheritance chain, a weighted road
  network). Five `arxiv` entries run only where demo 01's graph exists.
- `run.py` — the scorer. Generation goes through `AgensText2CypherRetriever`'s
  own path: the pre-pass, `query_untrusted` as the untrusted role, a timeout
  per entry, and optionally the correction loop.
- `tests/integration_tests/test_text2cypher_dataset.py` — the dataset's own
  gate, no model involved, on every release: every gold runs as `llm_ro` and
  returns what its entry declares, and every habit errors, is refused or
  disagrees with gold. An engine change that invalidates an entry fails CI
  naming the entry.

## Running

```bash
cd langchain
.venv/bin/python evals/text2cypher/run.py --graphs t2c_movies,t2c_traps,arxiv
.venv/bin/python evals/text2cypher/run.py --examples 12 ...   # DIALECT_EXAMPLES few-shot
.venv/bin/python evals/text2cypher/run.py --retry ...         # max_retries=2 + retry_on_empty
.venv/bin/python evals/text2cypher/run.py --model gpt-4o ...
```

The owner's connection and the key come from the demos' configuration
(`examples/demos/.env`), and the untrusted role's conninfo from
`AGENSGRAPH_UNTRUSTED_URL`. Each run prints the table and writes a per-item
`results_<tag>.jsonl` beside this file for diffing runs.

## Scoring plans, not just rows

Entries marked `expect_index` also score the PLAN of the generated query:
`plan_has_index_path` EXPLAINs it under `enable_seqscan = off`, where a plan
the planner marks disabled (`Disabled: true` from PostgreSQL 18, a cost of at
least 10^10 before it) is its own statement that no index path exists. The
`unindexed` habit outcome encodes the silent failure this exists for — a query
that returns exactly the right rows while reading the whole label — and the
validation test proves both sides of every such claim.

A model's output varies between runs even at temperature zero, so treat
single-digit differences as noise and re-run before believing them.
