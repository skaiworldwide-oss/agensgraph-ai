"""Unit tests for the graph store's pure helpers.

What can be tested without a server is label cleaning, the sanitizer, and the shape the
schema is rendered in. Decoding the wire format is the driver's, and is tested there.
"""

from __future__ import annotations

import unittest
from typing import Any, Dict, List

from langchain_agensgraph.graphs.agensgraph import AgensGraph


class TestAgensGraphHelpers(unittest.TestCase):
    def test_clean_graph_labels(self) -> None:
        inputs = ["label", "label 1", "label#$"]

        expected = ["label", "label_1", "label_"]

        for idx, value in enumerate(inputs):
            self.assertEqual(AgensGraph.clean_graph_labels(value), expected[idx])

    def test_structured_schema_splits_vertex_and_edge_properties(self) -> None:
        """A label's kind decides which half of the schema it lands in.

        An edge's property map has to survive decoding for the relationship half to hold
        anything at all.
        """
        from agensgraph import GraphDescription, PropertyShape, Triple
        from agensgraph.introspect import Label

        description = GraphDescription(
            graph="g",
            labels=(
                Label(3, "Person", "v", "ag_vertex"),
                Label(4, "KNOWS", "e", "ag_edge"),
            ),
            properties={
                "Person": (PropertyShape("name", "string", False),),
                "KNOWS": (PropertyShape("since", "integer", False),),
            },
            triples=(Triple("Person", "KNOWS", "Person", 7),),
            counts={"Person": 2, "KNOWS": 7},
            meta_gathered=True,
        )

        structured = AgensGraph._structured(_FakeGraph(), description)

        self.assertEqual(
            structured["node_props"], {"Person": [{"property": "name", "type": "string"}]}
        )
        self.assertEqual(
            structured["rel_props"], {"KNOWS": [{"property": "since", "type": "integer"}]}
        )
        self.assertEqual(
            structured["relationships"],
            [{"start": "Person", "type": "KNOWS", "end": "Person"}],
        )
        self.assertEqual(structured["counts"], {"Person": 2, "KNOWS": 7})
        self.assertTrue(structured["metadata"]["meta_gathered"])

    def test_rendered_schema_names_each_section(self) -> None:
        """The prompt's framing is contractual -- a chain's few-shot examples read it."""
        structured: Dict[str, Any] = {
            "node_props": {"Person": [{"property": "name", "type": "string"}]},
            "rel_props": {"KNOWS": [{"property": "since", "type": "integer"}]},
            "relationships": [{"start": "Person", "type": "KNOWS", "end": "Person"}],
            "counts": {"Person": 2},
            "metadata": {},
        }
        rendered = AgensGraph._rendered(None, structured)

        self.assertIn("Node properties are the following:", rendered)
        self.assertIn("Relationship properties are the following:", rendered)
        self.assertIn("The relationships are the following:", rendered)
        # Counts are new, and free: they come from the catalogs with everything else, and a
        # model picking between two ways of matching writes a better query knowing them.
        self.assertIn("Element counts are the following:", rendered)
        self.assertIn('(:"Person")-[:"KNOWS"]->(:"Person")', rendered)


class _FakeGraph:
    """Enough of an AgensGraph for `_structured`, which only reads `capabilities`."""

    class capabilities:  # noqa: N801 - stands in for a property
        version = (2, 18)


def _entries(props: List[Dict[str, str]]) -> List[str]:
    return [p["property"] for p in props]
