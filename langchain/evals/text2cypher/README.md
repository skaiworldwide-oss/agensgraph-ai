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

## Measured — gpt-4o-mini, 90 entries, 2.18-devel

Commands exactly as above, all four on the same server and fixtures:

| configuration      | executable | execution accuracy |
| ------------------ | ---------- | ------------------ |
| 0-shot             | 87%        | 52%                |
| 0-shot + retry     | 92%        | 59%                |
| 12-shot            | 87%        | **70%**            |
| 12-shot + retry    | 92%        | **72%**            |

What the deltas say: the twelve `DIALECT_EXAMPLES` pairs are the big lever
(+18 points — direction went 2/6 to 5/6, jsonb semantics 6/10 to 9/10,
function idioms 4/12 to 8/12); the self-correction loop adds +7 alone and
raises the executable rate to 92% in both cases. The hardest category in every
configuration is `paths` (1/6 at best): `shortestpath` requires pre-bound
endpoints and `dijkstra` has its own capture form, and no amount of prompting
has taught those yet — the per-item files name each miss.

A model's output varies between runs even at temperature zero, so treat
single-digit differences as noise and re-run before believing them.
