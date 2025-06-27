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

import unittest
from collections import namedtuple
from typing import Any, Dict, List

from langchain.graphs.agensgraph import AgensGraph


class TestAgensGraph(unittest.TestCase):
    def test_format_triples(self) -> None:
        test_input = [
            {"start": "from_a", "type": "edge_a", "end": "to_a"},
            {"start": "from_b", "type": "edge_b", "end": "to_b"},
        ]

        expected = [
            '(:"from_a")-[:"edge_a"]->(:"to_a")',
            '(:"from_b")-[:"edge_b"]->(:"to_b")',
        ]

        self.assertEqual(AgensGraph._format_triples(test_input), expected)

    def test_format_properties(self) -> None:
        inputs: List[Dict[str, Any]] = [{}, {"a": "b"}, {"a": "b", "c": 1, "d": True}]

        expected = ["{}", '{"a": \'b\'}', '{"a": \'b\', "c": 1, "d": True}']

        for idx, value in enumerate(inputs):
            self.assertEqual(AgensGraph._format_properties(value), expected[idx])

    def test_clean_graph_labels(self) -> None:
        inputs = ["label", "label 1", "label#$"]

        expected = ["label", "label_1", "label_"]

        for idx, value in enumerate(inputs):
            self.assertEqual(AgensGraph.clean_graph_labels(value), expected[idx])

    def test_record_to_dict(self) -> None:
        Record = namedtuple("Record", ["node1", "edge", "node2"])
        r = Record(
            node1='label1[1.1]{"prop": "a"}',
            edge='edge[2.1][1.1,1.2]{"test": "abc"}',
            node2='label1[1.2]{"prop": "b"}'
        )

        result = AgensGraph._record_to_dict(r)

        expected = {
            "node1": {"prop": 'a'},
            "edge": ({"prop": 'a'}, "edge", {"prop": 'b'}),
            "node2": {"prop": 'b'},
        }

        self.assertEqual(result, expected)

        Record2 = namedtuple("Record2", ["string", "int", "float", "bool", "null"])
        r2 = Record2('"test"', "1", "1.5", "true", None)

        result = AgensGraph._record_to_dict(r2)

        expected2 = {
            "string": "test",
            "int": 1,
            "float": 1.5,
            "bool": True,
            "null": None,
        }

        self.assertEqual(result, expected2)
