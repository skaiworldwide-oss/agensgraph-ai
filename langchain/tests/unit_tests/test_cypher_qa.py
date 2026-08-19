"""Unit tests for the text2cypher pipeline — no database, no real model."""

import pytest

from langchain_agensgraph.chains.cypher_qa import (
    CYPHER_SYSTEM,
    case_sensitive_names,
    is_write_query,
    quote_identifiers,
    strip_fences,
)


class TestStripFences:
    def test_removes_markdown_fences(self):
        assert strip_fences("```cypher\nMATCH (n) RETURN n\n```") == "MATCH (n) RETURN n"

    def test_removes_a_bare_fence(self):
        assert strip_fences("```\nMATCH (n) RETURN n\n```") == "MATCH (n) RETURN n"

    def test_removes_a_trailing_semicolon(self):
        assert strip_fences("MATCH (n) RETURN n;") == "MATCH (n) RETURN n"

    def test_leaves_a_clean_query_alone(self):
        assert strip_fences("MATCH (n) RETURN n") == "MATCH (n) RETURN n"


class TestQuoteIdentifiers:
    """Folding runs both ways, so only the schema can say what may be quoted.

    A name the graph stores folded must be left bare; quoting it would turn a working
    query into one that silently matches nothing — the very failure this guards against.
    """

    KNOWN = {"Person", "firstName", "WORKS_AT", "Name"}

    def q(self, cypher):
        return quote_identifiers(cypher, self.KNOWN)

    def test_quotes_a_known_label(self):
        assert self.q("MATCH (n:Person) RETURN n") == 'MATCH (n:"Person") RETURN n'

    def test_quotes_a_known_relationship_type(self):
        got = self.q("MATCH (a)-[r:WORKS_AT]->(b) RETURN r")
        assert got == 'MATCH (a)-[r:"WORKS_AT"]->(b) RETURN r'

    def test_quotes_a_known_property_that_starts_lower_case(self):
        assert self.q("MATCH (n) RETURN n.firstName") == 'MATCH (n) RETURN n."firstName"'

    def test_quotes_a_known_map_key(self):
        got = self.q("MATCH (n {Name: 'x', age: 3}) RETURN n")
        assert got == 'MATCH (n {"Name": \'x\', age: 3}) RETURN n'

    def test_leaves_a_name_the_graph_stores_folded_alone(self):
        query = "MATCH (n:person) RETURN n.firstname"
        assert self.q(query) == query

    def test_leaves_an_unknown_name_alone(self):
        query = "MATCH (n:Unknown) RETURN n.nope"
        assert self.q(query) == query

    def test_quotes_nothing_without_a_schema(self):
        query = "MATCH (n:Person) RETURN n.firstName"
        assert quote_identifiers(query) == query

    def test_leaves_already_quoted_alone(self):
        query = 'MATCH (n:"Person") RETURN n."firstName"'
        assert self.q(query) == query

    def test_does_not_touch_string_literals(self):
        query = "MATCH (n) WHERE n.x = 'has:Person and .firstName' RETURN n"
        assert self.q(query) == query

    def test_does_not_mistake_a_cast_for_a_label(self):
        query = "MATCH (n) RETURN n.emb::vector(3)"
        assert self.q(query) == query


class TestCaseSensitiveNames:
    def test_collects_labels_and_properties_that_are_not_lower_case(self):
        schema = {
            "node_props": {
                "Person": [{"property": "firstName"}, {"property": "name"}],
                "city": [{"property": "pop"}],
            },
            "rel_props": {},
            "relationships": [{"type": "WORKS_AT", "start": "Person", "end": "city"}],
        }
        assert case_sensitive_names(schema) == {"Person", "firstName", "WORKS_AT"}

    def test_an_empty_schema_yields_nothing(self):
        assert case_sensitive_names({}) == set()


class TestIsWriteQuery:
    @pytest.mark.parametrize(
        "query",
        [
            "CREATE (n:Person)",
            "MATCH (n) SET n.x = 1",
            "MATCH (n) DETACH DELETE n",
            "MERGE (n:Person {name: 'a'})",
            "MATCH (n) REMOVE n.x",
        ],
    )
    def test_detects_writes(self, query):
        assert is_write_query(query)

    @pytest.mark.parametrize(
        "query",
        [
            "MATCH (n) RETURN n",
            "MATCH (n) WHERE n.name = 'CREATE' RETURN n",
            'MATCH (n:"Person") WHERE n."deleted" IS NULL RETURN n',
            "MATCH (n) RETURN n -- CREATE was mentioned in a comment",
            "MATCH (n) RETURN n /* CREATE was mentioned in a comment */",
        ],
    )
    def test_a_keyword_in_a_string_or_comment_is_not_a_write(self, query):
        assert not is_write_query(query)

    @pytest.mark.parametrize(
        "query",
        [
            "MATCH (n) RETURN n.set",
            "MATCH (n) RETURN n.create AS created",
            "MATCH (n) RETURN n.remove",
        ],
    )
    def test_a_property_named_after_a_clause_is_not_a_write(self, query):
        """A read that a first opinion must not refuse.

        The name follows a dot, so it is a key rather than a clause, and a chain that
        refuses this answers nothing for a graph whose data happens to use the word.
        """
        assert not is_write_query(query)

    def test_a_second_statement_after_a_semicolon_is_read(self):
        """A parameterless statement goes over the simple query protocol, which runs
        every statement in it, so the write in the second one is a write in the text."""
        assert is_write_query("MATCH (n) RETURN n; CREATE (m:Evil)")


class TestPrompt:
    """The prompt is the feature: these rules are what make the Cypher runnable."""

    @pytest.mark.parametrize(
        "rule",
        [
            'n:"Person"',      # quoting
            "fold to lower case",
            "apoc",            # Neo4j-only constructs ruled out by name
            "COUNT { ... }",
            "EXISTS { ... }",
            "count(*)",        # the performance rule
            "Read-only",
        ],
    )
    def test_prompt_states_the_dialect_rule(self, rule):
        assert rule in CYPHER_SYSTEM

    def test_prompt_parameterizes_the_row_limit(self):
        assert "{top_k}" in CYPHER_SYSTEM

    @pytest.mark.parametrize(
        "rule",
        [
            "Compare the raw property",
            "Match the stored type exactly",
            "half-open range",
            "tolower(n.prop)",
            "ORDER BY an indexed property with LIMIT",
        ],
    )
    def test_prompt_states_the_index_rules(self, rule):
        assert rule in CYPHER_SYSTEM

    def test_the_pack_teaches_no_property_wrapping(self):
        # The two forms the sargability measurements ruled out: a cast or
        # arithmetic around an indexed property has no index path at all.
        from langchain_agensgraph.chains.cypher_qa import DIALECT_EXAMPLES

        taught = " ".join(cypher for _q, cypher in DIALECT_EXAMPLES)
        assert "::int4" not in taught
        assert "::float8" not in taught


class TestFewShotExamples:
    """Example pairs become real conversation turns ahead of the question."""

    def _chain(self, **kwargs):
        from langchain_core.language_models import FakeListChatModel

        from langchain_agensgraph.chains.cypher_qa import AgensCypherQAChain

        llm = FakeListChatModel(responses=["MATCH (n) RETURN n"])
        return AgensCypherQAChain.from_llm(llm, graph=object(), **kwargs)

    def test_examples_become_alternating_turns(self):
        chain = self._chain(examples=[("q1", "c1"), ("q2", "c2")])
        messages = chain.cypher_prompt.format_messages(schema="S", question="Q")
        assert [m.type for m in messages] == [
            "system",
            "human",
            "ai",
            "human",
            "ai",
            "human",
        ]
        assert messages[1].content == "q1"
        assert messages[2].content == "c1"
        assert messages[-1].content.endswith("Question: Q")

    def test_braces_in_an_example_survive_templating(self):
        chain = self._chain(examples=[("map?", "RETURN {a: 1} AS m")])
        messages = chain.cypher_prompt.format_messages(schema="S", question="Q")
        assert messages[2].content == "RETURN {a: 1} AS m"

    def test_examples_and_a_custom_prompt_refuse_each_other(self):
        from langchain_core.prompts import ChatPromptTemplate

        prompt = ChatPromptTemplate.from_messages([("human", "{schema} {question}")])
        with pytest.raises(ValueError, match="not both"):
            self._chain(examples=[("q", "c")], cypher_prompt=prompt)

    def test_the_dialect_pack_renders_whole(self):
        from langchain_agensgraph.chains.cypher_qa import DIALECT_EXAMPLES

        assert len(DIALECT_EXAMPLES) >= 10
        chain = self._chain(examples=DIALECT_EXAMPLES)
        messages = chain.cypher_prompt.format_messages(schema="S", question="Q")
        assert len(messages) == 2 + 2 * len(DIALECT_EXAMPLES)
        # Every taught query is stated as the assistant's own turn, verbatim.
        taught = [m.content for m in messages if m.type == "ai"]
        assert taught == [cypher for _q, cypher in DIALECT_EXAMPLES]
