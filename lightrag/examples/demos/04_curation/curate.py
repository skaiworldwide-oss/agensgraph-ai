"""04 · curate — a LightRAG knowledge graph is maintainable, not write-once.

Real graphs need cleanup: the extractor sometimes emits the same entity twice,
descriptions need fixing, and documents get retracted. LightRAG supports the full
lifecycle — merge duplicate entities, edit entities/relations, delete an entity
(and purge its vectors), and delete a whole document (regenerating what it
contributed). This runs on a small, deterministic `lightrag_curation` graph built
from a handful of crafted paragraphs, so the before/after is easy to see.

    cd lightrag
    .venv/bin/python examples/demos/04_curation/curate.py

Knob: CURATION_RESET=0 to keep the graph between runs (default rebuilds it).
"""

from __future__ import annotations

import asyncio
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from lightrag.kg.shared_storage import initialize_pipeline_status

from _common import config, console
from _common.datautil import env_int
from _common.rag import build_rag, reset_rag

DB = "lightrag_curation"

DOCS = [
    "OpenAI is an artificial intelligence research company based in San Francisco. "
    "Sam Altman is the chief executive of OpenAI. OpenAI created the GPT-4 language model.",
    "Microsoft is a technology company headquartered in Redmond. Satya Nadella leads "
    "Microsoft as its CEO. Microsoft invested heavily in OpenAI and integrated GPT-4.",
    "Open AI partnered with Microsoft to deploy models on the Azure cloud platform. "
    "The collaboration brought GPT-4 to Azure customers worldwide.",
    "Sam Altman previously served as president of Y Combinator before joining OpenAI. "
    "He is a prominent figure in Silicon Valley.",
    "Azure is Microsoft's cloud computing platform. It hosts large language models "
    "including GPT-4 for enterprise customers.",
    "Google DeepMind is a rival AI lab. It competes with OpenAI in developing models.",
]


def _norm(s: str) -> str:
    return "".join(ch.lower() for ch in s if ch.isalnum())


def pick_merge_pair(labels):
    """Find two labels that look like the same entity (else fall back to top two)."""
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a, b = _norm(labels[i]), _norm(labels[j])
            if a and b and (a == b or a in b or b in a):
                return labels[i], labels[j]
    return (labels[0], labels[1]) if len(labels) >= 2 else (None, None)


async def main() -> None:
    config.require_openai_key()
    config.ensure_db(DB)
    rag = build_rag(DB)
    await rag.initialize_storages()
    await initialize_pipeline_status()
    g = rag.chunk_entity_relation_graph
    try:
        if env_int("CURATION_RESET", 1):
            await reset_rag(rag)
        console.section("Build a small graph (6 crafted documents)")
        with console.timer("ingest"):
            await rag.ainsert(DOCS, ids=[f"doc-{i}" for i in range(len(DOCS))],
                              file_paths=[f"crafted-{i}" for i in range(len(DOCS))])
        labels = await g.get_all_labels()
        console.kv("entities", len(labels))
        print("  " + ", ".join(labels))

        # 1) MERGE duplicate entities -----------------------------------------
        src, tgt = pick_merge_pair(labels)
        console.section(f"merge_entities — fold '{src}' into '{tgt}'")
        if src and tgt:
            before = await g.node_degree(tgt)
            await rag.amerge_entities(source_entities=[src], target_entity=tgt)
            print(f"  '{tgt}' degree {before} → {await g.node_degree(tgt)}; "
                  f"'{src}' still present: {await g.has_node(src)}")
            labels = await g.get_all_labels()
            console.kv("entities after merge", len(labels))

        # 2) EDIT an entity ----------------------------------------------------
        console.section(f"aedit_entity — fix '{tgt}' description/type")
        await rag.aedit_entity(tgt, {"description": f"{tgt} (curated description, verified).",
                                     "entity_type": "organization"})
        node = await g.get_node(tgt)
        print(f"  type={node.get('entity_type')}, desc='{(node.get('description') or '')[:80]}'")

        # 3) EDIT a relation ---------------------------------------------------
        edges = await g.get_node_edges(tgt)
        if edges:
            s, t = edges[0]
            console.section(f"aedit_relation — annotate ({s}) — ({t})")
            await rag.aedit_relation(s, t, {"description": "curated relationship", "weight": 9.0})
            edge = await g.get_edge(s, t)
            print(f"  weight={edge.get('weight')}, desc='{(edge.get('description') or '')[:60]}'")

        # 4) DELETE an entity (and its vectors) --------------------------------
        leaf = (await g.get_all_labels())[-1]
        console.section(f"adelete_by_entity — remove '{leaf}'")
        await rag.adelete_by_entity(leaf)
        print(f"  present after delete: {await g.has_node(leaf)}; "
              f"entities now: {len(await g.get_all_labels())}")

        # 5) DELETE a whole document -------------------------------------------
        rows, total = await rag.doc_status.get_docs_paginated(page=1, page_size=1)
        console.section("adelete_by_doc_id — retract one source document")
        if rows:
            doc_id = rows[0][0]
            ents_before = len(await g.get_all_labels())
            result = await rag.adelete_by_doc_id(doc_id)
            counts = await rag.doc_status.get_all_status_counts()
            print(f"  deleted {doc_id}: status={getattr(result, 'status', result)}")
            print(f"  documents now: {counts.get('all')}, entities: {ents_before} → "
                  f"{len(await g.get_all_labels())}")
    finally:
        await rag.finalize_storages()


if __name__ == "__main__":
    asyncio.run(main())
