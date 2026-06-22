"""AgensGraph-dialect Text2Cypher prompt + a read-only validator.

LlamaIndex's ``TextToCypherRetriever`` defaults to ``DEFAULT_CYPHER_TEMPALTE``,
which gives the LLM no dialect guidance and feeds it a Neo4j-style schema
(``(:Paper)-[:AUTHORED_BY]->(:Author)``). That produces Cypher the
``AgensPropertyGraphStore`` cannot run, because the store keeps EVERY node under
one ``"__Node__"`` vertex label and records the entity type as a string in each
node's ``labels`` list property — there is no ``Paper`` vertex label to match.

So we override the prompt to teach the real storage model, and pass a validator
that strips markdown and refuses anything that isn't read-only.
"""

from __future__ import annotations

import logging
import re

from llama_index.core import PromptTemplate
from llama_index.core.indices.property_graph import TextToCypherRetriever

logger = logging.getLogger(__name__)

# Filled by TextToCypherRetriever with {schema} (from store.get_schema_str) and
# {question}. The schema lists entity *types* as if they were labels — the rules
# below tell the LLM they are values in the `labels` list, not Cypher labels.
AGENS_CYPHER_TEMPLATE = """\
You translate a question into a SINGLE read-only AgensGraph (openCypher) query.

How this graph is stored (important):
- EVERY node uses one vertex label: "__Node__". Match nodes as (n:"__Node__").
- The entity TYPES in the schema below (e.g. Paper, Author, Person) are NOT Cypher
  labels. They are string values held in each node's `labels` list property.
  To restrict to a type, filter the list: WHERE 'Paper' IN n.labels.
- The human-readable name of an entity is the property n.name. Other properties
  are exactly as named in the schema.
- Relationship TYPES are real edge labels. Either write them double-quoted, e.g.
  (a)-[r:"AUTHORED_BY"]->(b), or use an untyped (a)-[r]->(b) and read type(r).
  Use ONLY relationship types that appear in the schema.

Hard rules:
- Read-only ONLY: MATCH / OPTIONAL MATCH / WHERE / WITH / RETURN / ORDER BY / LIMIT.
  NEVER CREATE / MERGE / SET / DELETE / REMOVE / DROP / DETACH.
- AgensGraph is NOT Neo4j. Do NOT use Neo4j-only constructs: no pattern expressions
  inside expressions, e.g. size((n)--()) or [(n)-->(m) | m]; no COUNT { ... };
  no EXISTS { ... }; no apoc.* ; no CALL { ... } subqueries. To count an entity's
  degree, MATCH its relationships and use count(), as shown below.
- Use ONLY the entity types, relationship types and properties shown in the schema.
- Always end with a LIMIT of at most 50.
- Return ONLY the Cypher query — no markdown fences, no explanation.

Examples (note the "__Node__" label and the `labels` filter):
  Q: How many entities of each type are there?
  MATCH (n:"__Node__") UNWIND n.labels AS t
  WITH t WHERE t <> '__Entity__'
  RETURN t AS type, count(*) AS n ORDER BY n DESC LIMIT 50

  Q: Which authors have the most papers?
  MATCH (a:"__Node__") WHERE 'Author' IN a.labels
  MATCH (a)<-[:"AUTHORED_BY"]-(p:"__Node__")
  RETURN a.name AS author, count(p) AS papers ORDER BY papers DESC LIMIT 10

  Q: Which entities are connected to the most other entities? (degree)
  MATCH (n:"__Node__")-[r]-(m:"__Node__")
  RETURN n.name AS entity, count(DISTINCT m) AS connections
  ORDER BY connections DESC LIMIT 5

Schema:
{schema}

Question: {question}
Cypher query:"""

AGENS_CYPHER_PROMPT = PromptTemplate(AGENS_CYPHER_TEMPLATE)

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
    single query the LLM gets wrong (e.g. a Neo4j-ism AgensGraph rejects) raises
    and aborts the whole query engine. This subclass catches that, logs it, and
    returns no nodes — the other sub-retrievers still answer.
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
