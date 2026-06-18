"""Integration test for enhanced_schema sampling."""

from __future__ import annotations

import os

from langchain_agensgraph import AgensGraph


def _conf():
    return {
        "dbname": os.getenv("AGENSGRAPH_DB"),
        "user": os.getenv("AGENSGRAPH_USER"),
        "password": os.getenv("AGENSGRAPH_PASSWORD"),
        "host": os.getenv("AGENSGRAPH_HOST", "localhost"),
        "port": int(os.getenv("AGENSGRAPH_PORT", 5432)),
    }


def test_enhanced_schema_includes_examples():
    g = AgensGraph("enhanced", _conf(), create=True, enhanced_schema=True)
    g.query("MATCH (n) DETACH DELETE n")
    g.query("CREATE VLABEL IF NOT EXISTS Person")
    g.query("CREATE (n:Person {name: 'Alice', city: 'Berlin'})")
    g.query("CREATE (n:Person {name: 'Bob', city: 'Paris'})")
    g.refresh_schema(force=True)

    # unquoted labels fold to lowercase in AgensGraph
    person = g.get_structured_schema["node_props"].get("person", [])
    by_name = {p["property"]: p for p in person}
    assert "examples" in by_name["name"]
    assert set(by_name["name"]["examples"]) <= {"Alice", "Bob"}
    # the formatted schema string surfaces the examples too
    assert "examples" in g.get_schema
    g.query("MATCH (n) DETACH DELETE n")
    g.close()


def test_enhanced_schema_off_has_no_examples():
    g = AgensGraph("enhanced", _conf(), create=True)  # default off
    g.query("MATCH (n) DETACH DELETE n")
    g.query("CREATE VLABEL IF NOT EXISTS Widget")
    g.query("CREATE (n:Widget {sku: 'A1'})")
    g.refresh_schema(force=True)
    widget = g.get_structured_schema["node_props"].get("widget", [])
    assert widget  # property was detected
    assert all("examples" not in p for p in widget)
    g.query("MATCH (n) DETACH DELETE n")
    g.close()
