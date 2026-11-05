"""The rewrites model-written Cypher gets before it is sent, per server and per schema."""

from __future__ import annotations

import pytest
from agensgraph import Capabilities

from langchain_agensgraph.chains.prepass import (
    case_sensitive_names,
    correct_directions,
    prepare,
    quote_identifiers,
    strip_fences,
)
from langchain_agensgraph.errors import UnsafeCypherError

RELEASES = ["2.14.1", "2.15.0", "2.16.0", "2.17.0", "2.18.4.0", "2.18.6.1"]


@pytest.mark.parametrize(
    "written",
    [
        "MATCH (n) RETURN n",
        "MATCH (n) RETURN n;",
        "```cypher\nMATCH (n) RETURN n\n```",
        "```\nMATCH (n) RETURN n;\n```",
        "Here is the query:\n```cypher\nMATCH (n) RETURN n\n```\nIt returns every node.",
    ],
)
def test_the_statement_comes_out_of_what_the_model_wrapped_it_in(written: str) -> None:
    assert strip_fences(written) == "MATCH (n) RETURN n"


class TestQuoteIdentifiers:
    KNOWN = {"Person", "firstName", "WORKS_AT", "Name"}

    @pytest.mark.parametrize(
        ("written", "sent"),
        [
            ("MATCH (n:Person) RETURN n", 'MATCH (n:"Person") RETURN n'),
            ("MATCH (a)-[r:WORKS_AT]->(b) RETURN r", 'MATCH (a)-[r:"WORKS_AT"]->(b) RETURN r'),
            ("MATCH (n) RETURN n.firstName", 'MATCH (n) RETURN n."firstName"'),
            (
                "MATCH (n {Name: 'x', age: 3}) RETURN n",
                "MATCH (n {\"Name\": 'x', age: 3}) RETURN n",
            ),
        ],
    )
    def test_a_name_the_graph_stores_with_capitals_is_quoted(self, written: str, sent: str) -> None:
        assert quote_identifiers(written, self.KNOWN) == sent

    @pytest.mark.parametrize(
        "written",
        [
            "MATCH (n:person) RETURN n.firstname",
            "MATCH (n:Unknown) RETURN n.nope",
            'MATCH (n:"Person") RETURN n."firstName"',
            "MATCH (n) WHERE n.x = 'has:Person and .firstName' RETURN n",
            "MATCH (n) RETURN n.emb::vector(3)",
            "MATCH (n) // a :Person by .firstName\nRETURN n",
        ],
    )
    def test_anything_else_is_left_as_written(self, written: str) -> None:
        assert quote_identifiers(written, self.KNOWN) == written

    def test_without_a_schema_nothing_is_quoted(self) -> None:
        assert quote_identifiers("MATCH (n:Person) RETURN n") == "MATCH (n:Person) RETURN n"

    def test_the_names_come_from_the_schema(self) -> None:
        schema = {
            "node_props": {
                "Person": [{"property": "firstName"}, {"property": "name"}],
                "city": [{"property": "pop"}],
            },
            "rel_props": {},
            "relationships": [{"type": "WORKS_AT", "start": "Person", "end": "city"}],
        }
        assert case_sensitive_names(schema) == {"Person", "firstName", "WORKS_AT"}
        assert case_sensitive_names({}) == set()


class TestCorrectDirections:
    TRIPLES = [
        {"start": "Person", "type": "WORKS_AT", "end": "Company"},
        {"start": "Person", "type": "KNOWS", "end": "Person"},
        {"start": "Paper", "type": "AUTHORED_BY", "end": "Author"},
    ]

    @pytest.mark.parametrize(
        ("written", "sent"),
        [
            (
                'MATCH (c:"Company")-[:"WORKS_AT"]->(p:"Person") RETURN p',
                'MATCH (c:"Company")<-[:"WORKS_AT"]-(p:"Person") RETURN p',
            ),
            (
                'MATCH (p:"Person")<-[r:"WORKS_AT"]-(c:"Company") RETURN p',
                'MATCH (p:"Person")-[r:"WORKS_AT"]->(c:"Company") RETURN p',
            ),
            (
                'MATCH (c:"Company"), (p:"Person") MATCH (c)-[:"WORKS_AT"]->(p) RETURN p',
                'MATCH (c:"Company"), (p:"Person") MATCH (c)<-[:"WORKS_AT"]-(p) RETURN p',
            ),
            (
                'MATCH (a:"Author")-[:"AUTHORED_BY"]->(p:"Paper")-[:"AUTHORED_BY"]->(b:"Author") '
                "RETURN b",
                'MATCH (a:"Author")<-[:"AUTHORED_BY"]-(p:"Paper")-[:"AUTHORED_BY"]->(b:"Author") '
                "RETURN b",
            ),
        ],
    )
    def test_an_edge_against_its_only_direction_is_turned(self, written: str, sent: str) -> None:
        assert correct_directions(written, self.TRIPLES) == sent

    @pytest.mark.parametrize(
        "written",
        [
            'MATCH (p:"Person")-[:"WORKS_AT"]->(c:"Company") RETURN p',
            'MATCH (a:"Person")<-[:"KNOWS"]-(b:"Person") RETURN a',
            'MATCH (c:"Company")-[:"WORKS_AT"]-(p:"Person") RETURN p',
            'MATCH (c:"Company")-[r:"WORKS_AT"*1..2]->(p:"Person") RETURN p',
            'MATCH (c)-[:"WORKS_AT"]->(p:"Person") RETURN p',
            'MATCH (c:"Company")-[r]->(p:"Person") RETURN p',
            'MATCH (c:"City")-[:"WORKS_AT"]->(p:"Person") RETURN p',
            'MATCH (n) WHERE n.note = \'(c:"Company")-[:"WORKS_AT"]->(p:"Person")\' RETURN n',
        ],
    )
    def test_anything_the_schema_does_not_settle_is_left(self, written: str) -> None:
        assert correct_directions(written, self.TRIPLES) == written

    def test_a_bare_name_is_read_folded(self) -> None:
        triples = [{"start": "person", "type": "works_at", "end": "company"}]
        written = "MATCH (c:Company)-[:WORKS_AT]->(p:Person) RETURN p"
        assert correct_directions(written, triples) == (
            "MATCH (c:Company)<-[:WORKS_AT]-(p:Person) RETURN p"
        )


class TestPrepare:
    @pytest.mark.parametrize("release", RELEASES)
    def test_bare_arrows_get_brackets_on_every_release(self, release: str) -> None:
        caps = Capabilities(release)
        assert prepare("MATCH (a)-->(b)<--(c)--(d) RETURN a", caps) == (
            "MATCH (a)-[]->(b)<-[]-(c)-[]-(d) RETURN a"
        )

    @pytest.mark.parametrize("release", RELEASES)
    def test_an_unnamed_walk_is_named_where_it_would_be_wrong(self, release: str) -> None:
        sent = prepare('MATCH (a)-[:"KNOWS"*1..3]->(b) RETURN b', Capabilities(release))
        if release < "2.16":
            assert sent == 'MATCH (a)-[vle0:"KNOWS"*1..3]->(b) RETURN b'
        else:
            assert sent == 'MATCH (a)-[:"KNOWS"*1..3]->(b) RETURN b'

    @pytest.mark.parametrize("release", RELEASES)
    def test_a_walk_inside_a_shortest_path_call_is_left_unnamed(self, release: str) -> None:
        written = (
            'MATCH (a), (c) OPTIONAL MATCH (a)-[:"KNOWS"*1..2]->(x) '
            'RETURN length(shortestPath((a)-[:"KNOWS"*1..5]->(c))) AS n, '
            "size(allshortestpaths((a)-[*1..2]-(c))) AS m, x"
        )
        sent = prepare(written, Capabilities(release))
        if release < "2.16":
            assert sent == written.replace('[:"KNOWS"*1..2]', '[vle0:"KNOWS"*1..2]')
        else:
            assert sent == written

    @pytest.mark.parametrize("release", RELEASES)
    def test_a_negative_index_becomes_last_before_2_18(self, release: str) -> None:
        sent = prepare(
            "MATCH p = (a)-[r*1..2]->(b) RETURN nodes(p)[-1] AS x", Capabilities(release)
        )
        expected = "last(nodes(p))" if release < "2.18" else "nodes(p)[-1]"
        assert f"RETURN {expected} AS x" in sent

    @pytest.mark.parametrize("release", RELEASES)
    def test_an_uncast_min_is_refused_where_it_compares_text(self, release: str) -> None:
        caps = Capabilities(release)
        written = "MATCH (n) RETURN min(n.age) AS youngest"
        if "2.15" <= release < "2.18":
            with pytest.raises(UnsafeCypherError, match="cast it inside"):
                prepare(written, caps)
        else:
            assert prepare(written, caps) == written
        cast = "MATCH (n) RETURN min(n.age::int4) AS youngest"
        assert prepare(cast, caps) == cast

    @pytest.mark.parametrize("release", RELEASES)
    def test_number_conversions_are_refused_for_casts(self, release: str) -> None:
        with pytest.raises(UnsafeCypherError, match="x::float8"):
            prepare("MATCH (n) RETURN toFloat(n.score) AS s", Capabilities(release))

    def test_strings_and_quoted_names_are_left_alone(self) -> None:
        written = "MATCH (n:\"a-->b\") WHERE n.s = 'toFloat(x) --> min(n.y)' RETURN n"
        assert prepare(written, Capabilities("2.15.0")) == written
