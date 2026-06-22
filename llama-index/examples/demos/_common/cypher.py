"""Read-only Text2Cypher helpers for the demos.

The AgensGraph-dialect Text2Cypher prompt now ships as the store's default
``text_to_cypher_template`` (the store keeps every node on one ``"__Node__"``
vertex label with its type in a ``labels`` list, which the default LlamaIndex
prompt doesn't know about), so demos just rely on that default and add:

- ``read_only_validator`` — strips markdown and refuses any non-read-only query.
- ``SafeTextToCypherRetriever`` — won't crash the query engine on a bad generation.
"""

from __future__ import annotations

import logging
import re

from llama_index.core.indices.property_graph import TextToCypherRetriever

logger = logging.getLogger(__name__)

_WRITE = re.compile(r"\b(CREATE|MERGE|SET|DELETE|REMOVE|DROP|DETACH)\b", re.IGNORECASE)
_FENCE = re.compile(r"```(?:cypher)?", re.IGNORECASE)


def read_only_validator(cypher: str) -> str:
    """Clean the LLM's output and reject any write.

    Used as ``TextToCypherRetriever(cypher_validator=read_only_validator)``: it
    strips markdown fences / trailing semicolons and raises if the query is not
    read-only, so a hallucinated mutation can never reach the database.
    """
    text = _FENCE.sub("", cypher).strip().rstrip(";").strip()
    if _WRITE.search(text):
        raise ValueError(f"Refusing to run a non-read-only Cypher query:\n{text}")
    return text


class SafeTextToCypherRetriever(TextToCypherRetriever):
    """``TextToCypherRetriever`` that never crashes the query.

    LlamaIndex's retriever runs the generated Cypher with no error handling, so a
    single query the LLM gets wrong raises and aborts the whole query engine. This
    subclass catches that, logs it, and returns no nodes — the other sub-retrievers
    still answer.
    """

    def retrieve_from_graph(self, query_bundle):  # type: ignore[override]
        try:
            return super().retrieve_from_graph(query_bundle)
        except Exception as e:  # noqa: BLE001 — any execution/parse error is non-fatal here
            logger.warning(
                "Text2Cypher query failed, skipping: %s",
                str(e).splitlines()[0][:160],
            )
            return []
