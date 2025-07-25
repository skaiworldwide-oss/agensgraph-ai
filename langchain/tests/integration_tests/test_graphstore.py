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
import re
import unittest, pytest
from typing import Any, Dict

from langchain_core.documents import Document

from langchain_agensgraph.graphs.agensgraph import AgensGraph
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship

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
        self.assertIsNotNone(conf["password"])

        self.graph = AgensGraph("test", conf)
        self.graph.query("MATCH (n) DETACH DELETE n")

    def test_node_properties(self) -> None:
        # Create two nodes and a relationship
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
        # Refresh schema information
        # self.graph.refresh_schema()

        node_properties = self.graph._get_node_properties()

        expected_node_properties = [
            {
                "properties": [{"property": "property_a", "type": "STRING"}],
                "labels": "LabelA",
            }
        ]

        self.assertEqual(
            sorted(node_properties, key=lambda x: x["labels"]), expected_node_properties
        )

    def test_edge_properties(self) -> None:
        # Create two nodes and a relationship
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
        # Refresh schema information
        # self.graph.refresh_schema()

        relationships_properties = self.graph._get_edge_properties()

        expected_relationships_properties = [
            {
                "type": "REL_TYPE",
                "properties": [{"property": "rel_prop", "type": "STRING"}],
            }
        ]

        self.assertEqual(relationships_properties, expected_relationships_properties)

    def test_relationships(self) -> None:
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
        # Refresh schema information
        # self.graph.refresh_schema()

        relationships = self.graph._get_triples()

        expected_relationships = [
            {"start": "LabelA", "type": "REL_TYPE", "end": "LabelB"},
            {"start": "LabelA", "type": "REL_TYPE", "end": "LabelC"},
        ]

        self.assertEqual(
            sorted(relationships, key=lambda x: x["end"]), expected_relationships
        )

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

        expected = [
            {"label": "bar", "count": 1},
            {"label": "Document", "count": 1},
            {"label": "foo", "count": 2},
        ]
        self.assertEqual(output, expected)

    def test_get_schema(self) -> None:
        self.graph.refresh_schema()

        expected = """
            Node properties are the following:
            []
            Relationship properties are the following:
            []
            The relationships are the following:
            []
            """
        # check that works on empty schema
        graph_schema = self.graph.get_schema

        self.assertEqual(
            re.sub(r"\s", "", graph_schema), re.sub(r"\s", "", expected)
        )

        expected_structured: Dict[str, Any] = {
            "node_props": {},
            "rel_props": {},
            "relationships": [],
            "metadata": {},
        }

        self.assertEqual(self.graph.get_structured_schema, expected_structured)

        # Create two nodes and a relationship
        self.graph.query(
            """
            CREATE VLABEL IF NOT EXISTS a;
            CREATE VLABEL IF NOT EXISTS c;
            CREATE ELABEL IF NOT EXISTS b;
            MERGE (a:a {id: 1})-[b:b {id: 2}]-> (c:c {id: 3})
            """
        )

        # check that schema doesn't update without refresh
        self.assertEqual(
            re.sub(r"\s", "", self.graph.get_schema), re.sub(r"\s", "", expected)
        )
        self.assertEqual(self.graph.get_structured_schema, expected_structured)

        # two possible orderings of node props
        expected_possibilities = [
            """
            Node properties are the following:
            [
                {'labels': 'a', 'properties': [{'type': 'INTEGER', 'property': 'id'}]},
                {'labels': 'c', 'properties': [{'type': 'INTEGER', 'property': 'id'}]}
            ]
            Relationship properties are the following:
            [
                {'type': 'b', 'properties': [{'type': 'INTEGER', 'property': 'id'}]}
            ]
            The relationships are the following:
            [
                '(:"a")-[:"b"]->(:"c")'
            ]
            """,
            """
            Node properties are the following:
            [
                {'labels': 'c', 'properties': [{'type': 'INTEGER', 'property': 'id'}]},
                {'labels': 'a', 'properties': [{'type': 'INTEGER', 'property': 'id'}]}
            ]
            Relationship properties are the following:
            [
                {'type': 'b', 'properties': [{'type': 'INTEGER', 'property': 'id'}]}
            ]
            The relationships are the following:
            [
                '(:"a")-[:"b"]->(:"c")'
            ]
            """,
        ]

        expected_structured2 = {
            "node_props": {
                "a": [{"property": "id", "type": "INTEGER"}],
                "c": [{"property": "id", "type": "INTEGER"}],
            },
            "rel_props": {"b": [{"property": "id", "type": "INTEGER"}]},
            "relationships": [{"start": "a", "type": "b", "end": "c"}],
            "metadata": {},
        }

        self.graph.refresh_schema()

        # check that schema is refreshed
        self.assertIn(
            re.sub(r"\s", "", self.graph.get_schema),
            [re.sub(r"\s", "", x) for x in expected_possibilities],
        )
        self.assertEqual(self.graph.get_structured_schema, expected_structured2)