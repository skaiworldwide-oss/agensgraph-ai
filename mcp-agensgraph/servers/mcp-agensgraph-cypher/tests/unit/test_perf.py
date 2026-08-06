"""Unit tests for plan analysis — no database."""

from mcp_agensgraph_cypher.perf import (
    analyze_plan,
    format_findings,
    indexed_properties,
    relname_to_label,
)


def plan(node):
    return [{"Plan": node}]


class TestMissingIndex:
    def test_a_big_label_scanned_with_a_property_filter_is_flagged(self):
        findings = analyze_plan(
            plan(
                {
                    "Node Type": "Seq Scan",
                    "Relation Name": "item",
                    "Filter": "(properties.'k'::text = '\"x\"'::jsonb)",
                }
            ),
            {"item": 50000},
            {},
        )
        assert [f["kind"] for f in findings] == ["missing_index"]
        assert findings[0]["properties"] == ["k"]
        assert 'CREATE PROPERTY INDEX ON "item" (k);' == findings[0]["suggestion"]

    def test_a_small_label_is_left_alone(self):
        """Reading a handful of rows end to end is the right plan."""
        findings = analyze_plan(
            plan(
                {
                    "Node Type": "Seq Scan",
                    "Relation Name": "item",
                    "Filter": "(properties.'k'::text = '\"x\"'::jsonb)",
                }
            ),
            {"item": 5},
            {},
        )
        assert findings == []

    def test_a_property_already_indexed_is_not_recommended_again(self):
        findings = analyze_plan(
            plan(
                {
                    "Node Type": "Seq Scan",
                    "Relation Name": "item",
                    "Filter": "(properties.'k'::text = '\"x\"'::jsonb)",
                }
            ),
            {"item": 50000},
            {"item": ["k"]},
        )
        assert findings == []

    def test_an_index_scan_is_not_flagged(self):
        findings = analyze_plan(
            plan(
                {
                    "Node Type": "Index Scan",
                    "Relation Name": "item",
                    "Index Cond": "(properties.'k'::text = '\"x\"'::jsonb)",
                }
            ),
            {"item": 50000},
            {},
        )
        assert findings == []

    def test_nested_plan_nodes_are_searched(self):
        findings = analyze_plan(
            plan(
                {
                    "Node Type": "Limit",
                    "Plans": [
                        {
                            "Node Type": "Seq Scan",
                            "Relation Name": "item",
                            "Filter": "(properties.'k'::text = '\"x\"'::jsonb)",
                        }
                    ],
                }
            ),
            {"item": 50000},
            {},
        )
        assert [f["kind"] for f in findings] == ["missing_index"]


class TestKnownAntiPatterns:
    def test_starts_with_is_flagged_however_big_the_label(self):
        findings = analyze_plan(
            plan(
                {
                    "Node Type": "Seq Scan",
                    "Relation Name": "item",
                    "Filter": "string_starts_with(properties.'k'::text, '\"a\"'::jsonb)",
                }
            ),
            {"item": 5},
            {},
        )
        kinds = [f["kind"] for f in findings]
        assert "starts_with_not_indexable" in kinds
        assert "AGV2-514" in findings[kinds.index("starts_with_not_indexable")]["suggestion"]

    def test_a_jsonb_containment_test_is_flagged(self):
        findings = analyze_plan(
            plan(
                {
                    "Node Type": "Bitmap Heap Scan",
                    "Relation Name": "item",
                    "Filter": "('[\"a\"]'::jsonb @> properties.'k'::text)",
                }
            ),
            {"item": 50000},
            {},
        )
        kinds = [f["kind"] for f in findings]
        assert "bound_in_list_not_indexable" in kinds
        finding = findings[kinds.index("bound_in_list_not_indexable")]
        assert "AGV2-515" in finding["suggestion"]


class TestReporting:
    def test_findings_are_labelled_as_unverified(self):
        """The tool must never imply it costed an index it could not build."""
        out = format_findings([])
        assert out["verified"] is False
        assert "cannot be simulated" in out["note"]

    def test_indexed_properties_reads_the_rendered_definition(self):
        rows = [
            {
                "label": "item",
                "definition": "CREATE PROPERTY INDEX i ON item ((properties.'k'::text))",
            }
        ]
        assert indexed_properties(rows) == {"item": ["k"]}

    def test_indexed_properties_falls_back_to_a_shorthand_definition(self):
        rows = [{"label": "item", "definition": "CREATE PROPERTY INDEX i ON item (a, b)"}]
        assert indexed_properties(rows) == {"item": ["a", "b"]}

    def test_row_counts_are_keyed_by_the_name_a_plan_reports(self):
        rows = [{"relname": "item", "approx_rows": 42, "label": "item"}]
        assert relname_to_label(rows) == {"item": 42}
