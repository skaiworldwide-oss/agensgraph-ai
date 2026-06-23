"""04 · build — turn a Python codebase into a knowledge graph.

cognee has a dedicated code pipeline: it parses a repository's Python files and
their dependencies into a graph of modules/classes/functions, which you can then
search (`SearchType.CODE`) and traverse. This builds that graph in AgensGraph
(`cognee_code`) from a small, well-known package.

    cd cognee
    .venv/bin/python examples/demos/04_code_graph/build.py

Knobs: CODE_REPO (path to a local Python package to analyze — overrides the
default clone), CODE_REPO_URL (git URL to shallow-clone), CODE_RESET=0.
"""

from __future__ import annotations

import asyncio
import os
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from _common import config, console
from _common.datautil import env_int

DB = "cognee_code"
REPO_URL = os.getenv("CODE_REPO_URL", "https://github.com/psf/requests")
CLONE_DIR = config.DATA_DIR / "code_repo"


def resolve_target() -> str:
    """A local Python package directory to analyze (bounded for cost)."""
    override = os.getenv("CODE_REPO")
    if override:
        return override
    if not CLONE_DIR.exists():
        console.sub(f"cloning {REPO_URL} (shallow)")
        subprocess.run(
            ["git", "clone", "--depth", "1", REPO_URL, str(CLONE_DIR)],
            check=True, capture_output=True,
        )
    # Prefer the package dir to keep the graph focused/bounded.
    for cand in (CLONE_DIR / "src" / "requests", CLONE_DIR / "src", CLONE_DIR):
        if cand.exists():
            return str(cand)
    return str(CLONE_DIR)


async def main() -> None:
    config.require_openai_key()
    config.quiet()
    config.ensure_db(DB)
    config.configure(DB)

    import cognee
    from cognee.api.v1.cognify.code_graph_pipeline import run_code_graph_pipeline
    from cognee.infrastructure.databases.graph import get_graph_engine

    target = resolve_target()
    n_py = sum(1 for _ in pathlib.Path(target).rglob("*.py"))
    console.section("Code-graph build")
    console.kv("analyzing", target)
    console.kv("python files", n_py)

    if env_int("CODE_RESET", 1):
        console.sub("CODE_RESET=1 — pruning existing memory")
        await config.aprune()

    console.section("Running the code-graph pipeline (parse + extract)")
    with console.timer("run_code_graph_pipeline"):
        async for status in run_code_graph_pipeline(target, include_docs=False):
            state = str(getattr(status, "status", ""))
            if "Error" in state:
                raise SystemExit(f"code-graph pipeline failed: {getattr(status, 'payload', state)}")

    metrics = await (await get_graph_engine()).get_graph_metrics(include_optional=False)
    console.kv("nodes", metrics.get("num_nodes"))
    console.kv("edges", metrics.get("num_edges"))
    print("\n  Code graph built. Query it with: python examples/demos/04_code_graph/ask.py")


if __name__ == "__main__":
    asyncio.run(main())
