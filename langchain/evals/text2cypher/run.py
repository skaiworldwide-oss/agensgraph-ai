"""Score a model's Cypher against the dataset, by running what it writes.

Every entry is scored on execution, not on text: the model's query is generated
and run through the retriever's own pipeline (write refusal, EXPLAIN check,
read-only transaction, per-entry timeout), the gold query runs in the same
session, and the two result sets are compared values-only. A query that reads
well but returns the wrong rows scores zero, which is the point.

    cd langchain
    .venv/bin/python evals/text2cypher/run.py                     # 0-shot, no retry
    .venv/bin/python evals/text2cypher/run.py --examples 12       # few-shot
    .venv/bin/python evals/text2cypher/run.py --retry             # self-correction on
    .venv/bin/python evals/text2cypher/run.py --graphs t2c_movies,t2c_traps,arxiv

Uses the demos' connection settings (AGENSGRAPH_* / examples/demos/.env) and
OPENAI_API_KEY. Emits a per-item JSONL next to this file and prints the table.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import defaultdict
from typing import Any, Dict, List

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "examples" / "demos"))

import t2c
from _common import config, console, models

from langchain_agensgraph.chains.cypher_qa import DIALECT_EXAMPLES
from langchain_agensgraph.graphs.agensgraph import AgensGraph
from langchain_agensgraph.retrievers import AgensText2CypherRetriever


def classify(exc: Exception) -> str:
    text = str(exc)
    if "Refusing to run" in text:
        return "refused-write"
    if "not runnable" in text:
        return "not-runnable"
    return "runtime-error"


def score_graph(
    graph_name: str,
    entries: List[Dict[str, Any]],
    llm: Any,
    shots: int,
    retry: bool,
) -> List[Dict[str, Any]]:
    conf = config.conf()
    graph = AgensGraph(graph_name, conf, create=False, refresh_schema=True)
    examples = DIALECT_EXAMPLES[:shots] if shots else None
    retriever = AgensText2CypherRetriever(
        graph=graph,
        llm=llm,
        k=10,
        timeout=30.0,
        examples=examples,
        max_retries=1 if retry else 0,
        retry_on_empty=retry,
        # The eval connects as the cluster's superuser; a deployment would not.
        allow_server_programs=True,
    )
    outcomes = []
    for entry in entries:
        unmet = t2c.requires_met(entry, graph.capabilities)
        if unmet:
            outcomes.append({**_tag(entry), "skipped": unmet})
            continue
        with graph.read_only(allow_server_programs=True):
            gold_rows = graph.query(entry["gold"], timeout=30)
        outcome: Dict[str, Any] = _tag(entry)
        try:
            docs = retriever.invoke(entry["question"])
        except Exception as exc:
            outcome.update(ok=False, match=False, error=classify(exc))
            outcomes.append(outcome)
            continue
        got_rows = [json.loads(d.page_content) for d in docs]
        cypher = docs[0].metadata["cypher"] if docs else None
        matched = t2c.results_match(got_rows, gold_rows, entry.get("ordered", False))
        outcome.update(ok=True, match=matched, cypher=cypher)
        if not matched:
            outcome["error"] = "empty" if not got_rows and gold_rows else "mismatch"
        outcomes.append(outcome)
    graph.close()
    return outcomes


def _tag(entry: Dict[str, Any]) -> Dict[str, Any]:
    return {"id": entry["id"], "category": entry["category"], "graph": entry["graph"]}


def arxiv_present() -> bool:
    try:
        graph = AgensGraph("arxiv", config.conf(), create=False, refresh_schema=False)
        rows = graph.query('MATCH (p:"Paper") RETURN count(*) AS n LIMIT 1', timeout=10)
        graph.close()
        return bool(rows and rows[0]["n"])
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None, help="chat model name")
    parser.add_argument(
        "--examples", type=int, default=0, help="few-shot pairs (0=off)"
    )
    parser.add_argument(
        "--retry", action="store_true", help="max_retries=1 + retry_on_empty"
    )
    parser.add_argument("--graphs", default="t2c_movies,t2c_traps")
    parser.add_argument("--out", default=None, help="per-item JSONL path")
    args = parser.parse_args()

    wanted = [g.strip() for g in args.graphs.split(",") if g.strip()]
    dataset = t2c.load_dataset()
    llm = models.get_llm(model=args.model)
    tag = "{}-{}shot{}".format(
        (args.model or "default").replace("/", "_"),
        args.examples,
        "-retry" if args.retry else "",
    )

    conf = config.conf()
    all_outcomes: List[Dict[str, Any]] = []
    for graph_name in wanted:
        entries = [e for e in dataset if e["graph"] == graph_name]
        if not entries:
            continue
        if graph_name in t2c.BUILDERS:
            fixture = AgensGraph(graph_name, conf, create=True, refresh_schema=False)
            t2c.BUILDERS[graph_name](fixture)
            fixture.close()
        elif graph_name == "arxiv" and not arxiv_present():
            console.kv(graph_name, "absent — skipped")
            continue
        with console.timer(f"{graph_name}: {len(entries)} entries"):
            all_outcomes += score_graph(
                graph_name, entries, llm, args.examples, args.retry
            )

    scored = [o for o in all_outcomes if "skipped" not in o]
    ran = [o for o in scored if o["ok"]]
    matched = [o for o in scored if o.get("match")]
    console.section(f"results — {tag}")
    console.kv("entries", len(scored))
    total = max(1, len(scored))
    console.kv(
        "executable", f"{len(ran)}/{len(scored)} ({100 * len(ran) / total:.0f}%)"
    )
    console.kv(
        "execution accuracy",
        f"{len(matched)}/{len(scored)} ({100 * len(matched) / total:.0f}%)",
    )

    by_category: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for o in scored:
        by_category[o["category"]].append(o)
    console.sub("per category (matched/total)")
    for category in sorted(by_category):
        group = by_category[category]
        hits = sum(1 for o in group if o.get("match"))
        console.kv(category, f"{hits}/{len(group)}")

    errors: Dict[str, int] = defaultdict(int)
    for o in scored:
        if o.get("error"):
            errors[o["error"]] += 1
    if errors:
        console.sub("failure taxonomy")
        for kind in sorted(errors, key=errors.get, reverse=True):
            console.kv(kind, errors[kind])

    out = pathlib.Path(args.out) if args.out else HERE / f"results_{tag}.jsonl"
    lines = [json.dumps(o, ensure_ascii=False) for o in all_outcomes]
    out.write_text("\n".join(lines) + "\n")
    console.kv("per-item results", str(out))


if __name__ == "__main__":
    main()
