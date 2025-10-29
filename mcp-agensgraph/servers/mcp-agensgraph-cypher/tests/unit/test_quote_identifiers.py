"""Unit tests for identifier quoting functionality."""

import pytest

from mcp_agensgraph_cypher.utils import _quote_identifiers


class TestQuoteIdentifiers:
    """Test cases for _quote_identifiers function."""

    def test_simple_label_with_uppercase(self):
        """Test that labels starting with uppercase are quoted."""
        query = "MATCH (p:Person) RETURN p"
        result = _quote_identifiers(query)
        assert result == 'MATCH (p:"Person") RETURN p'

    def test_multiple_labels(self):
        """Test multiple labels with uppercase."""
        query = "MATCH (a:User)-[:FRIEND]->(b:Person) RETURN a, b"
        result = _quote_identifiers(query)
        assert result == 'MATCH (a:"User")-[:"FRIEND"]->(b:"Person") RETURN a, b'

    def test_property_access_with_uppercase(self):
        """Test property access with uppercase letters."""
        query = "MATCH (p:Person) RETURN p.FirstName, p.LastName"
        result = _quote_identifiers(query)
        assert result == 'MATCH (p:"Person") RETURN p."FirstName", p."LastName"'

    def test_property_in_pattern_with_uppercase(self):
        """Test property names in CREATE/MERGE patterns."""
        query = "CREATE (n:Person {FirstName: 'John', LastName: 'Doe'})"
        result = _quote_identifiers(query)
        assert result == 'CREATE (n:"Person" {"FirstName": \'John\', "LastName": \'Doe\'})'

    def test_lowercase_labels_unchanged(self):
        """Test that lowercase labels are not quoted."""
        query = "MATCH (p:person) RETURN p.name"
        result = _quote_identifiers(query)
        assert result == "MATCH (p:person) RETURN p.name"

    def test_lowercase_properties_unchanged(self):
        """Test that lowercase properties are not quoted."""
        query = "MATCH (p:Person) RETURN p.firstname"
        result = _quote_identifiers(query)
        assert result == 'MATCH (p:"Person") RETURN p.firstname'

    def test_already_quoted_labels_unchanged(self):
        """Test that already quoted labels are not double-quoted."""
        query = 'MATCH (p:"Person") RETURN p'
        result = _quote_identifiers(query)
        assert result == 'MATCH (p:"Person") RETURN p'

    def test_already_quoted_properties_unchanged(self):
        """Test that already quoted properties are not double-quoted."""
        query = 'MATCH (p:Person) RETURN p."FirstName"'
        result = _quote_identifiers(query)
        assert result == 'MATCH (p:"Person") RETURN p."FirstName"'

    def test_mixed_case_label(self):
        """Test labels with mixed case (camelCase, PascalCase)."""
        query = "MATCH (n:MyCustomLabel) RETURN n"
        result = _quote_identifiers(query)
        assert result == 'MATCH (n:"MyCustomLabel") RETURN n'

    def test_relationship_type_with_uppercase(self):
        """Test relationship types with uppercase."""
        query = "MATCH (a)-[:FriendOf]->(b) RETURN a, b"
        result = _quote_identifiers(query)
        assert result == 'MATCH (a)-[:"FriendOf"]->(b) RETURN a, b'

    def test_complex_query(self):
        """Test a complex query with multiple identifier types."""
        query = """
        MATCH (user:User {UserId: 123})
        -[:Follows]->(friend:Person)
        WHERE friend.Age > 18
        RETURN user.UserName, friend.FullName
        """
        result = _quote_identifiers(query)

        # Check all identifiers are properly quoted
        assert ':"User"' in result
        assert '{"UserId":' in result
        assert ':"Follows"' in result
        assert ':"Person"' in result
        assert 'friend."Age"' in result
        assert 'user."UserName"' in result
        assert 'friend."FullName"' in result

    def test_create_with_multiple_properties(self):
        """Test CREATE statement with multiple properties."""
        query = "CREATE (n:Person {Name: 'Alice', Age: 30, City: 'NYC'})"
        result = _quote_identifiers(query)
        assert result == 'CREATE (n:"Person" {"Name": \'Alice\', "Age": 30, "City": \'NYC\'})'

    def test_merge_statement(self):
        """Test MERGE statement with uppercase identifiers."""
        query = "MERGE (p:Person {PersonId: 1})"
        result = _quote_identifiers(query)
        assert result == 'MERGE (p:"Person" {"PersonId": 1})'

    def test_set_statement_with_properties(self):
        """Test SET statement with property updates."""
        query = "MATCH (p:Person) SET p.FirstName = 'John'"
        result = _quote_identifiers(query)
        assert result == 'MATCH (p:"Person") SET p."FirstName" = \'John\''

    def test_where_clause_with_properties(self):
        """Test WHERE clause with property comparisons."""
        query = "MATCH (p:Person) WHERE p.Age > 18 AND p.Country = 'USA' RETURN p"
        result = _quote_identifiers(query)
        assert result == 'MATCH (p:"Person") WHERE p."Age" > 18 AND p."Country" = \'USA\' RETURN p'

    def test_order_by_with_properties(self):
        """Test ORDER BY clause with properties."""
        query = "MATCH (p:Person) RETURN p ORDER BY p.LastName, p.FirstName"
        result = _quote_identifiers(query)
        assert result == 'MATCH (p:"Person") RETURN p ORDER BY p."LastName", p."FirstName"'

    def test_aggregation_with_properties(self):
        """Test aggregation functions with properties."""
        query = "MATCH (p:Person) RETURN count(p.UserId), avg(p.Age)"
        result = _quote_identifiers(query)
        assert result == 'MATCH (p:"Person") RETURN count(p."UserId"), avg(p."Age")'

    def test_label_starting_with_lowercase_unchanged(self):
        """Test that labels starting with lowercase are not quoted."""
        query = "MATCH (p:myLabel) RETURN p"
        result = _quote_identifiers(query)
        assert result == "MATCH (p:myLabel) RETURN p"

    def test_property_starting_with_lowercase_unchanged(self):
        """Test that properties starting with lowercase are not quoted."""
        query = "MATCH (p:Person) RETURN p.myProperty"
        result = _quote_identifiers(query)
        assert result == 'MATCH (p:"Person") RETURN p.myProperty'

    def test_empty_query(self):
        """Test empty query."""
        query = ""
        result = _quote_identifiers(query)
        assert result == ""

    def test_query_without_identifiers(self):
        """Test query without uppercase identifiers."""
        query = "MATCH (n) RETURN n"
        result = _quote_identifiers(query)
        assert result == "MATCH (n) RETURN n"

    def test_special_characters_in_strings_preserved(self):
        """Test that strings with colons are not affected."""
        query = "CREATE (n:Person {email: 'user@example.com'})"
        result = _quote_identifiers(query)
        assert result == 'CREATE (n:"Person" {email: \'user@example.com\'})'

    def test_numeric_suffixes_in_identifiers(self):
        """Test identifiers with numeric suffixes."""
        query = "MATCH (p:Person2) RETURN p.Name123"
        result = _quote_identifiers(query)
        assert result == 'MATCH (p:"Person2") RETURN p."Name123"'

    def test_underscore_in_identifiers(self):
        """Test identifiers with underscores."""
        query = "MATCH (p:User_Profile) RETURN p.First_Name"
        result = _quote_identifiers(query)
        assert result == 'MATCH (p:"User_Profile") RETURN p."First_Name"'

    def test_relationship_with_properties(self):
        """Test relationship with properties."""
        query = "MATCH (a)-[r:Knows {Since: 2020}]->(b) RETURN r"
        result = _quote_identifiers(query)
        assert result == 'MATCH (a)-[r:"Knows" {"Since": 2020}]->(b) RETURN r'

    def test_multiple_patterns(self):
        """Test multiple patterns in one query."""
        query = "MATCH (a:Person), (b:Company) WHERE a.WorksAt = b.CompanyId RETURN a, b"
        result = _quote_identifiers(query)
        assert ':"Person"' in result
        assert ':"Company"' in result
        assert 'a."WorksAt"' in result
        assert 'b."CompanyId"' in result

    def test_return_with_aliases(self):
        """Test RETURN with AS aliases."""
        query = "MATCH (p:Person) RETURN p.FirstName AS Name"
        result = _quote_identifiers(query)
        assert result == 'MATCH (p:"Person") RETURN p."FirstName" AS Name'  # AS Name shouldn't be quoted as it's an alias

    def test_case_statement(self):
        """Test CASE statement with properties."""
        query = "MATCH (p:Person) RETURN CASE WHEN p.Age >= 18 THEN 'adult' ELSE 'minor' END"
        result = _quote_identifiers(query)
        assert ':"Person"' in result
        assert 'p."Age"' in result
