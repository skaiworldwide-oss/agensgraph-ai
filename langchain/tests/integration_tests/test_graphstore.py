'''
Copyright (c) 2025, SKAI Worldwide Co., Ltd.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

import os
import unittest

from langchain_core.documents import Document

from langchain_agensgraph.graphs.agensgraph import AgensGraph
from langchain_agensgraph.graphs.graph_document import (
    GraphDocument,
    Node,
    Relationship,
)

test_data = [
    GraphDocument(
        nodes=[
            Node(id="foo", type="foo"),
            Node(id="bar", type="bar"),
            Node(id="foo", type="foo", properties={"property_a": "a"}),
        ],
        relationships=[
            Relationship(
                source=Node(id="foo", type="foo"),
                target=Node(id="bar", type="bar"),
                type="REL",
            )
        ],
        source=Document(page_content="source document"),
    )
]

conf = {
    "dbname": os.getenv("AGENSGRAPH_DB"),
    "user": os.getenv("AGENSGRAPH_USER"),
    "password": os.getenv("AGENSGRAPH_PASSWORD"),
    "host": os.getenv("AGENSGRAPH_HOST", "localhost"),
    "port": int(os.getenv("AGENSGRAPH_PORT", 5432)),
}

class TestAgensGraph(unittest.TestCase):
    def setUp(self) -> None:
        self.assertIsNotNone(conf["dbname"])
        self.assertIsNotNone(conf["user"])

        self.graph = AgensGraph("test", conf, create=True)
        self.graph.query("MATCH (n) DETACH DELETE n")

    def _seed(self) -> None:
        self.graph.query(
            """
            CREATE ELABEL IF NOT EXISTS "REL_TYPE";
            CREATE (la:"LabelA" {property_a: 'a'})
            CREATE (lb:"LabelB")
            CREATE (lc:"LabelC")
            MERGE (la)-[:"REL_TYPE"]-> (lb)
            MERGE (la)-[:"REL_TYPE" {rel_prop: 'abc'}]-> (lc)
            """
        )
        self.graph.refresh_schema(force=True)

    def test_node_properties(self) -> None:
        self._seed()
        node_props = self.graph.get_structured_schema["node_props"]
        self.assertEqual(
            node_props["LabelA"], [{"property": "property_a", "type": "string"}]
        )
        # A label with no properties is still a label, and is still reported.
        self.assertIn("LabelB", node_props)

    def test_edge_properties(self) -> None:
        self._seed()
        rel_props = self.graph.get_structured_schema["rel_props"]
        # An edge's own properties, which are only reachable if the decoder keeps them.
        self.assertEqual(
            rel_props["REL_TYPE"], [{"property": "rel_prop", "type": "string"}]
        )

    def test_relationships(self) -> None:
        self._seed()
        rels = self.graph.get_structured_schema["relationships"]
        self.assertEqual(
            sorted(rels, key=lambda x: x["end"]),
            [
                {"start": "LabelA", "type": "REL_TYPE", "end": "LabelB"},
                {"start": "LabelA", "type": "REL_TYPE", "end": "LabelC"},
            ],
        )

    def test_counts_come_with_the_schema(self) -> None:
        """Counts are free -- they arrive from the catalogs with everything else."""
        self._seed()
        counts = self.graph.get_structured_schema["counts"]
        self.assertEqual(counts["LabelA"], 1)
        self.assertEqual(counts["REL_TYPE"], 2)

    def test_add_documents(self) -> None:
        # Create two nodes and a relationship
        self.graph.add_graph_documents(test_data)
        output = self.graph.query(
            "MATCH (n) RETURN label(n) AS label, count(*) AS count ORDER BY label"
        )
        self.assertEqual(
            output, [{"label": "bar", "count": 1}, {"label": "foo", "count": 1}]
        )

    def test_add_documents_source(self) -> None:
        # Create two nodes and a relationship
        self.graph.add_graph_documents(test_data, include_source=True)
        output = self.graph.query(
            "MATCH (n) RETURN label(n) AS label, count(*) AS count ORDER BY label"
        )

        # One `foo`, not two: both mentions carry id "foo" and an element is merged on
        # its id. Previously `include_source=True` merged on the whole property map, so
        # the same entity became two nodes depending on the flag.
        self.assertEqual(
            {row["label"]: row["count"] for row in output},
            {"Document": 1, "bar": 1, "foo": 1},
        )

    def test_get_schema(self) -> None:
        """The schema, empty and then populated.

        Asserted on the structured form plus the rendered string's sections, rather than
        on a verbatim repr of a Python dict. The old test compared the whole rendering
        character for character, which made it sensitive to dict ordering and to the
        spelling of a type name -- neither of which is what the schema is for.
        """
        self.graph.refresh_schema(force=True)

        empty = self.graph.get_schema
        for section in (
            "Node properties are the following:",
            "Relationship properties are the following:",
            "The relationships are the following:",
        ):
            self.assertIn(section, empty)

        actual_structured = self.graph.get_structured_schema
        self.assertEqual(actual_structured["node_props"], {})
        self.assertEqual(actual_structured["rel_props"], {})
        self.assertEqual(actual_structured["relationships"], [])
        self.assertIn("metadata", actual_structured)

        self.graph.query(
            """
            CREATE VLABEL IF NOT EXISTS a;
            CREATE VLABEL IF NOT EXISTS c;
            CREATE ELABEL IF NOT EXISTS b;
            MERGE (a:a {id: 1})-[b:b {id: 2}]-> (c:c {id: 3})
            """
        )

        # The schema does not update without a refresh.
        stale = self.graph.get_structured_schema
        self.assertEqual(stale["node_props"], {})
        self.assertEqual(stale["relationships"], [])

        self.graph.refresh_schema(force=True)
        refreshed = self.graph.get_structured_schema

        # `integer` rather than `INTEGER`: the type is `jsonb_typeof`'s, with a whole
        # number told from a fractional one.
        self.assertEqual(
            refreshed["node_props"],
            {
                "a": [{"property": "id", "type": "integer"}],
                "c": [{"property": "id", "type": "integer"}],
            },
        )
        self.assertEqual(
            refreshed["rel_props"], {"b": [{"property": "id", "type": "integer"}]}
        )
        self.assertEqual(
            refreshed["relationships"], [{"start": "a", "type": "b", "end": "c"}]
        )
        # Counts arrive from the catalogs with everything else.
        self.assertEqual(refreshed["counts"], {"a": 1, "c": 1, "b": 1})
        self.assertIn('(:"a")-[:"b"]->(:"c")', self.graph.get_schema)


class TestSchemaIndexes(unittest.TestCase):
    """The schema names the graph's btree property indexes.

    A predicate an index cannot serve returns the right rows while reading the
    whole label, so which indexes exist is a fact the query writer needs -- the
    schema is where it learns everything else about the graph.
    """

    def setUp(self) -> None:
        self.graph = AgensGraph(
            "schema_idx_test", conf, create=True, refresh_schema=False
        )
        self.graph.query("CREATE VLABEL IF NOT EXISTS doc")
        self.graph.query("MATCH (n) DETACH DELETE n")
        self.graph.query("CREATE (:doc {title: 'one', code: 'x1'})")
        for ddl in (
            "CREATE PROPERTY INDEX IF NOT EXISTS sit_title ON doc (title)",
            "CREATE UNIQUE PROPERTY INDEX IF NOT EXISTS sit_code ON doc (code)",
            "CREATE PROPERTY INDEX IF NOT EXISTS sit_lower ON doc ((tolower(title)))",
        ):
            self.graph.query(ddl)

    def tearDown(self) -> None:
        self.graph.close()

    def test_indexes_arrive_in_the_schema(self) -> None:
        self.graph.refresh_schema(force=True)
        entries = self.graph.get_structured_schema["indexes"]
        by_on = {entry["on"]: entry for entry in entries if entry["label"] == "doc"}
        self.assertIn("(title)", by_on)
        self.assertFalse(by_on["(title)"]["unique"])
        self.assertIn("(code)", by_on)
        self.assertTrue(by_on["(code)"]["unique"])
        self.assertIn("((tolower(title)))", by_on)
        rendered = self.graph.get_schema
        self.assertIn("Property indexes are the following", rendered)
        self.assertIn('(:"doc") ON (title)', rendered)
        self.assertIn('(:"doc") ON (code) UNIQUE', rendered)
        self.assertIn('(:"doc") ON ((tolower(title)))', rendered)

    def test_the_section_can_be_turned_off(self) -> None:
        quiet = AgensGraph(
            "schema_idx_test", conf, include_indexes=False, refresh_schema=False
        )
        quiet.refresh_schema(force=True)
        self.assertEqual(quiet.get_structured_schema["indexes"], [])
        self.assertNotIn("Property indexes", quiet.get_schema)
        quiet.close()
