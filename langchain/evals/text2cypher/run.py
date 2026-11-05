"""Score a model's Cypher against the dataset, by running what it writes.

Every entry is scored on execution, not on text: the model's statement goes through the
retriever's own path (the pre-pass, then ``query_untrusted`` as the untrusted role, with
corrections when ``--retry`` is given), the gold statement runs the same way, and the two results
are compared values only. A statement that reads well but returns the wrong rows scores zero,
which is the point.

    cd langchain
    .venv/bin/python evals/text2cypher/run.py                     # 0-shot, no retry
    .venv/bin/python evals/text2cypher/run.py --examples 12       # few-shot
    .venv/bin/python evals/text2cypher/run.py --retry             # corrections on
    .venv/bin/python evals/text2cypher/run.py --graphs t2c_movies,t2c_traps,arxiv

Uses the demos' connection settings (AGENSGRAPH_* / examples/demos/.env) for the owner, which
builds the fixture graphs and grants them, ``AGENSGRAPH_UNTRUSTED_URL`` for the role the model's
statements run as (``llm_ro`` from ``langchain_agensgraph.roles.render_roles``), and
OPENAI_API_KEY. Emits a per-item JSONL next to this file and prints the table.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from collections import defaultdict
from typing import Any

import psycopg

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "examples" / "demos"))

import t2c
from _common import config, console, models

from langchain_agensgraph import AgensEngine
from langchain_agensgraph.chains import DIALECT_EXAMPLES
from langchain_agensgraph.errors import AgensQueryError, UnsafeCypherError
from langchain_agensgraph.graphs import AgensGraph
from langchain_agensgraph.retrievers import AgensText2CypherRetriever
from langchain_agensgraph.roles import render_grants


def classify(exc: Exception) -> str:
    if isinstance(exc, UnsafeCypherError):
        return "refused"
    if isinstance(exc, AgensQueryError):
        return "not-runnable"
    return "runtime-error"


def score_graph(
    engine: AgensEngine,
    owner: str,
    graph_name: str,
    entries: list[dict[str, Any]],
    llm: Any,
    shots: int,
    retry: bool,
) -> list[dict[str, Any]]:
    graph = AgensGraph(engine, graph_name=graph_name, refresh_schema=True)
    retriever = AgensText2CypherRetriever(
        graph=graph,
        llm=llm,
        k=10,
        timeout=30.0,
        examples=DIALECT_EXAMPLES[:shots] if shots else None,
        max_retries=2 if retry else 0,
        retry_on_empty=retry,
        allow_dangerous_requests=True,
    )
    outcomes = []
    for entry in entries:
        unmet = t2c.requires_met(entry, graph.capabilities)
        if unmet:
            outcomes.append({**_tag(entry), "skipped": unmet})
            continue
        # Every gold fits in the retriever's ten rows, as test_text2cypher_dataset holds it to; an
        # eleventh row would show one that does not, and could never be matched.
        gold_rows = graph.query_untrusted(entry["gold"], max_rows=11, timeout=30)
        outcome: dict[str, Any] = _tag(entry)
        if entry.get("expect_index"):
            # Not proven indexed until the generated plan says so.
            outcome["indexed"] = False
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
        if entry.get("expect_index") and cypher:
            try:
                outcome["indexed"] = t2c.plan_has_index_path(owner, graph_name, cypher)
            except Exception:
                outcome["indexed"] = False
        outcomes.append(outcome)
    return outcomes


def _tag(entry: dict[str, Any]) -> dict[str, Any]:
    return {"id": entry["id"], "category": entry["category"], "graph": entry["graph"]}


def arxiv_present(engine: AgensEngine) -> bool:
    try:
        rows = AgensGraph(engine, graph_name="arxiv").query(
            'MATCH (p:"Paper") RETURN count(*) AS n LIMIT 1', timeout=10
        )
        return bool(rows and rows[0]["n"])
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None, help="chat model name")
    parser.add_argument("--examples", type=int, default=0, help="few-shot pairs (0=off)")
    parser.add_argument("--retry", action="store_true", help="max_retries=2 + retry_on_empty")
    parser.add_argument("--graphs", default="t2c_movies,t2c_traps")
    parser.add_argument("--out", default=None, help="per-item JSONL path")
    args = parser.parse_args()

    untrusted = os.getenv("AGENSGRAPH_UNTRUSTED_URL")
    if not untrusted:
        raise SystemExit(
            "AGENSGRAPH_UNTRUSTED_URL is not set: the conninfo of the role model-written Cypher "
            "runs as, such as llm_ro (see langchain_agensgraph.roles.render_roles)."
        )
    wanted = [g.strip() for g in args.graphs.split(",") if g.strip()]
    dataset = t2c.load_dataset()
    llm = models.get_llm(model=args.model)
    tag = "{}-{}shot{}".format(
        (args.model or "default").replace("/", "_"),
        args.examples,
        "-retry" if args.retry else "",
    )

    owner = config.url()
    all_outcomes: list[dict[str, Any]] = []
    with AgensEngine(owner, untrusted_conninfo=untrusted) as engine:
        for graph_name in wanted:
            entries = [e for e in dataset if e["graph"] == graph_name]
            if not entries:
                continue
            if graph_name in t2c.BUILDERS:
                t2c.BUILDERS[graph_name](
                    AgensGraph(engine, graph_name=graph_name, create_graph=True)
                )
            elif graph_name == "arxiv" and not arxiv_present(engine):
                console.kv(graph_name, "absent — skipped")
                continue
            with psycopg.connect(owner, autocommit=True) as conn:
                conn.execute(render_grants(graph_name))
            with console.timer(f"{graph_name}: {len(entries)} entries"):
                all_outcomes += score_graph(
                    engine, owner, graph_name, entries, llm, args.examples, args.retry
                )

    scored = [o for o in all_outcomes if "skipped" not in o]
    ran = [o for o in scored if o["ok"]]
    matched = [o for o in scored if o.get("match")]
    console.section(f"results — {tag}")
    console.kv("entries", len(scored))
    total = max(1, len(scored))
    console.kv("executable", f"{len(ran)}/{len(scored)} ({100 * len(ran) / total:.0f}%)")
    console.kv(
        "execution accuracy",
        f"{len(matched)}/{len(scored)} ({100 * len(matched) / total:.0f}%)",
    )
    checked = [o for o in scored if "indexed" in o]
    if checked:
        served = sum(1 for o in checked if o["indexed"])
        console.kv(
            "index-served",
            f"{served}/{len(checked)} "
            f"({100 * served / len(checked):.0f}% of index-checked entries)",
        )

    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for o in scored:
        by_category[o["category"]].append(o)
    console.sub("per category (matched/total)")
    for category in sorted(by_category):
        group = by_category[category]
        hits = sum(1 for o in group if o.get("match"))
        console.kv(category, f"{hits}/{len(group)}")

    errors: dict[str, int] = defaultdict(int)
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
