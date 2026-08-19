"""Retrievers for statements the model wrote.

``TextToCypherRetriever`` hands whatever the model produced straight to the store
and lets anything it raises out. :class:`SafeTextToCypherRetriever` runs it inside
the store's read-only block and answers with nothing when it fails.
"""

from __future__ import annotations

import logging
import re
from typing import Any, List

from agensgraph.errors import ConfigurationError
from llama_index.core.indices.property_graph import TextToCypherRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle

logger = logging.getLogger(__name__)

FENCE = re.compile(r"```(?:cypher)?", re.IGNORECASE)


def strip_markdown(cypher: str) -> str:
    """Take a statement out of the code fence a model tends to put it in.

    This is presentation, not safety. What the statement is allowed to do is the
    server's decision, taken inside :meth:`AgensPropertyGraphStore.read_only`.
    """
    return FENCE.sub("", cypher).strip().rstrip(";").strip()


class SafeTextToCypherRetriever(TextToCypherRetriever):
    """Runs the model's Cypher read-only, and does not take the query down with it.

    Two things the base retriever does not do.

    It runs the statement on the same connection everything else uses, with nothing
    stopping a write. A keyword list is not the answer -- it is PostgreSQL
    underneath, so ``INSERT``, ``TRUNCATE``, ``GRANT`` and ``COPY`` are all
    available and none of them is Cypher, while a read whose text merely mentions
    DELETE looks like a write. Here the server decides, in a transaction that
    cannot write and ends by rolling back.

    And it lets an execution error out, which aborts the whole query engine over
    one statement the model got wrong. The other sub-retrievers still have answers,
    so this one returns none of its own and says so in the log.
    """

    def __init__(self, *args: Any, allow_server_programs: bool = False, **kwargs: Any):
        """
        Args:
            allow_server_programs: run even though the connected role may run a
                command on the server's host. ``COPY ... TO PROGRAM`` takes rows
                out rather than putting any in, so a read-only transaction does
                not refuse it. A superuser holds that, which is why connecting as
                one is refused here rather than left to be discovered.
        """
        super().__init__(*args, **kwargs)
        self._allow_server_programs = allow_server_programs

    def _read_only(self) -> Any:
        return self.graph_store.read_only(
            allow_server_programs=self._allow_server_programs
        )

    def retrieve_from_graph(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        try:
            with self._read_only():
                return super().retrieve_from_graph(query_bundle)
        except ConfigurationError:
            # Not this statement's fault and not fixed by another question: the
            # role cannot be held to a read. Swallowed, every query would answer
            # with nothing and say only that the Cypher did not run.
            raise
        except Exception as e:  # noqa: BLE001 -- one bad generation is not fatal
            logger.warning(
                "the generated Cypher did not run, skipping it: %s",
                str(e).splitlines()[0][:160],
            )
            return []

    async def aretrieve_from_graph(
        self, query_bundle: QueryBundle
    ) -> List[NodeWithScore]:
        try:
            with self._read_only():
                return await super().aretrieve_from_graph(query_bundle)
        except ConfigurationError:
            raise
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "the generated Cypher did not run, skipping it: %s",
                str(e).splitlines()[0][:160],
            )
            return []


__all__ = ["SafeTextToCypherRetriever", "strip_markdown"]
