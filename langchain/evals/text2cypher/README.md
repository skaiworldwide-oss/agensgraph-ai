# Text2Cypher eval — the AgensGraph dialect, measured

Models write Neo4j-flavored Cypher; this engine diverges from that habit in ways
that are silent as often as loud. This eval turns "how well does text2cypher
work here" into a number: a dataset of question/gold-query pairs concentrated on
the divergences, scored by **execution** — the model's query runs, the gold
query runs, and the result sets must agree values-only. A query that reads
plausibly but returns the wrong rows scores zero.

## Layout

- `dataset.jsonl` — 90 entries. Each has a `question`, a dialect-correct `gold`
  query, a `category`, optional capability `requires`, and — for the traps —
  the `habit` query a model writes elsewhere, with its declared `habit_outcome`
  (`error`, or `different` from gold). Gold results are never stored; they are
  computed at run time against deterministic fixture graphs.
- `t2c.py` — the dataset loader, the values-only result canon, and the builders
  for the two fixture graphs: `t2c_movies` (a small catalog with a mixed-case
  label) and `t2c_traps` (a quoted upper-case relationship type, a stringly
  number, a list holding a null, an inheritance chain, a weighted road
  network). Five `arxiv` entries run only where demo 01's graph exists.
- `run.py` — the scorer. Generation goes through `AgensText2CypherRetriever`'s
  own pipeline: write refusal, EXPLAIN check, read-only transaction, a timeout
  per entry, and optionally the self-correction loop.
- `tests/integration_tests/test_text2cypher_dataset.py` — the dataset's own
  gate, no model involved: every gold executes and returns what its entry
  declares, and every habit errors or disagrees with gold. An engine change
  that invalidates an entry fails CI naming the entry.

## Running

```bash
cd langchain
.venv/bin/python evals/text2cypher/run.py --graphs t2c_movies,t2c_traps,arxiv
.venv/bin/python evals/text2cypher/run.py --examples 12 ...   # DIALECT_EXAMPLES few-shot
.venv/bin/python evals/text2cypher/run.py --retry ...         # max_retries=1 + retry_on_empty
.venv/bin/python evals/text2cypher/run.py --model gpt-4o ...
```

Connection and key come from the demos' configuration (`examples/demos/.env`).
Each run prints the table and writes a per-item `results_<tag>.jsonl` beside
this file for diffing runs.

## Scoring plans, not just rows

Entries marked `expect_index` also score the PLAN of the generated query:
`plan_has_index_path` EXPLAINs it under `enable_seqscan = off`, where a plan
carrying `Disabled: true` is the planner's own statement that no index path
exists. The `unindexed` habit outcome encodes the silent failure this exists
for — a query that returns exactly the right rows while reading the whole
label — and the validation test proves both sides of every such claim.

## Measured — gpt-4o-mini, 99 entries, 2.18-devel

Commands exactly as above (the pack is 17 pairs; `--examples 17`), all four
configurations on the same server and fixtures, with the schema's index
section and the prompt's index rules in place:

| configuration      | executable | execution accuracy | index-served |
| ------------------ | ---------- | ------------------ | ------------ |
| 0-shot             | 90%        | 60%                | 92%          |
| 0-shot + retry     | 94%        | 68%                | 92%          |
| 17-shot            | 83%        | 66%                | 92%          |
| 17-shot + retry    | 90%        | **73%**            | **92%**      |

What the numbers say: the index rules and the schema's index section carry the
index-served rate to 92% on their own — it is 12/13 in every configuration,
few-shot adds nothing there. The one recurring miss is the honest kind: asked
"which cities are not Seoul?", the model writes `<> 'Seoul'` — right rows, full
label read — and only the plan check notices. The full configuration answers
the `sargability` category 9/9 and `direction` 6/6; `paths` stays the floor
(1/6): `shortestpath` wants pre-bound endpoints and `dijkstra` its own capture
form, and prompting has not taught them yet. The 17-pair pack without the
retry loop dips the executable rate (83%); the retry recovers it.

A model's output varies between runs even at temperature zero, so treat
single-digit differences as noise and re-run before believing them.
