"""03 · build — a multi-dataset memory layer in one database.

cognee is a *memory* layer: data lives in named **datasets** you can build
incrementally and query in isolation or together. This builds two distinct
datasets in one `cognee_memory` database — an `encyclopedia` (Wikipedia) and a
`news` feed (CC-News) — cognifying them one after the other so you can watch the
memory grow without a rebuild.

    cd cognee
    MEM_LIMIT=15 .venv/bin/python examples/demos/03_memory/build.py   # tiny dry-run
    .venv/bin/python examples/demos/03_memory/build.py               # ~60 + ~60 docs

Knobs: MEM_LIMIT (docs per dataset), MEM_CHARS, MEM_RESET=0 (keep existing memory).
"""

from __future__ import annotations

import asyncio
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from _common import config, console
from _common.datautil import env_int, stream_hf
from _common.models import count_tokens, print_cost_estimate

DB = "cognee_memory"
LIMIT = env_int("MEM_LIMIT", 60)
CHARS = env_int("MEM_CHARS", 2000)


def wiki_docs():
    for rec in stream_hf("wikimedia/wikipedia", config="20231101.en", limit=LIMIT * 2):
        text, title = (rec.get("text") or "").strip(), (rec.get("title") or "").strip()
        if len(text) < 400 or not title:
            continue
        yield f"# {title}\n\n{text[:CHARS]}"


def news_docs():
    for rec in stream_hf("vblagoje/cc_news", limit=LIMIT * 3):
        text = (rec.get("text") or "").strip()
        if len(text) < 400:
            continue
        yield text[:CHARS]


def take(gen, n):
    out = []
    for d in gen:
        out.append(d)
        if len(out) >= n:
            break
    return out


async def graph_size():
    from cognee.infrastructure.databases.graph import get_graph_engine
    m = await (await get_graph_engine()).get_graph_metrics(include_optional=False)
    return m.get("num_nodes"), m.get("num_edges")


async def main() -> None:
    config.require_openai_key()
    config.quiet()
    config.ensure_db(DB)
    config.configure(DB)
    import cognee

    console.section("Collecting two datasets")
    encyclopedia = take(wiki_docs(), LIMIT)
    news = take(news_docs(), LIMIT)
    console.kv("encyclopedia (wikipedia)", len(encyclopedia))
    console.kv("news (cc-news)", len(news))
    print_cost_estimate(sum(count_tokens(d) for d in encyclopedia + news))

    if env_int("MEM_RESET", 1):
        console.sub("MEM_RESET=1 — pruning existing memory")
        await config.aprune()

    # Each dataset is also tagged with a node_set (a lightweight grouping label).
    console.section("Wave 1 — build the `encyclopedia` dataset")
    await cognee.add(encyclopedia, dataset_name="encyclopedia", node_set=["reference"])
    with console.timer("cognify encyclopedia"):
        await cognee.cognify(["encyclopedia"])
    n1, e1 = await graph_size()
    console.kv("memory after encyclopedia", f"{n1} nodes / {e1} edges")

    console.section("Wave 2 — add the `news` dataset (incremental, no rebuild)")
    await cognee.add(news, dataset_name="news", node_set=["current_events"])
    with console.timer("cognify news"):
        await cognee.cognify(["news"])
    n2, e2 = await graph_size()
    console.kv("memory after news", f"{n2} nodes / {e2} edges  (+{n2 - n1} nodes)")

    print("\n  Two datasets in one database. Query them with: "
          "python examples/demos/03_memory/ask.py")


if __name__ == "__main__":
    asyncio.run(main())
