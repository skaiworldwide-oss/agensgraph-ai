"""What a lead word does and does not decide about a statement."""

import pytest
from mcp_agensgraph_common.policy import PolicyRefusal, check_statement_is_allowed

class TestALeadWordIsNotTheWholeStatement:
    """A graph statement and a SQL one can begin with the same word.

    ``EXPLAIN`` takes any statement at all and ``WITH ... AS (...)`` takes a data-modifying one in
    the bracket, so reading only the first word admitted SQL that then landed its effect: an
    ``UPDATE`` behind ``EXPLAIN ANALYZE`` changed rows, a ``DELETE`` inside a ``WITH`` bracket
    removed them, and ``EXPLAIN ANALYZE CREATE TABLE ... AS SELECT`` left a table behind.
    """

    @pytest.mark.parametrize(
        "statement",
        [
            "WITH t AS (SELECT rolname FROM pg_roles) SELECT rolname FROM t",
            "EXPLAIN ANALYZE UPDATE public.audit SET v = 'x'",
            "WITH d AS (DELETE FROM public.audit RETURNING v) SELECT v FROM d",
            "EXPLAIN ANALYZE CREATE TABLE public.copied AS SELECT 1",
            "EXPLAIN ANALYZE INSERT INTO public.audit VALUES (1)",
            "CREATE INDEX ix ON public.audit (v)",
            "CREATE UNIQUE INDEX ix ON public.audit (v)",
            "MATCH (n) RETURN n; SELECT 1",
        ],
    )
    def test_sql_is_refused_wherever_it_sits(self, statement):
        with pytest.raises(PolicyRefusal):
            check_statement_is_allowed(statement)

    @pytest.mark.parametrize(
        "statement",
        [
            "MATCH (n:Person) RETURN n",
            "EXPLAIN MATCH (n:Person) RETURN n",
            "WITH 1 AS x RETURN x",
            "MATCH (n) WITH n, count(*) AS c WHERE c > 1 RETURN n",
            "CALL { MATCH (n) RETURN n AS v } RETURN v",
            "MERGE (n:P {k: 1}) SET n.v = 2",
            "UNWIND %(records)s AS r MERGE (n:L {id: r.id}) SET n += {a: r.a}",
            "MATCH (n) WHERE n.note = 'SELECT * FROM pg_roles' RETURN n",
        ],
    )
    def test_a_graph_statement_holding_no_sql_is_allowed(self, statement):
        check_statement_is_allowed(statement)


class TestAClauseThatCannotComeFirst:
    """SET, REMOVE, DELETE and DETACH need a variable something earlier bound.

    Each is a syntax error as a first word, so a statement leading with one is SQL. Reading them
    as graph leads admitted the session settings -- ``SET work_mem``, ``SET search_path`` -- which
    committed on a pooled connection and were inherited by whoever borrowed it next.
    """

    @pytest.mark.parametrize(
        "statement",
        [
            "SET work_mem = '199MB'",
            "SET search_path = 'poisoned'",
            "SET LOCAL statement_timeout = 0",
            "REMOVE n.v",
            "DELETE FROM public.audit",
            "DETACH PARTITION p FROM public.audit",
        ],
    )
    def test_a_statement_leading_with_one_is_refused(self, statement):
        with pytest.raises(PolicyRefusal):
            check_statement_is_allowed(statement)

    @pytest.mark.parametrize(
        "statement",
        [
            "MATCH (n:Person) SET n.v = 1",
            "MERGE (n:P {k: 1}) SET n += {v: 2}",
            "MATCH (n) REMOVE n.v RETURN n",
            "MATCH (n) DETACH DELETE n",
        ],
    )
    def test_the_same_clause_after_a_pattern_is_allowed(self, statement):
        check_statement_is_allowed(statement)


class TestAClauseThatNamesWhatItActsOn:
    """MERGE, CREATE and INSERT are graph clauses and SQL statements alike.

    Asked positively -- a graph one takes a pattern, so a bracket follows -- because listing SQL
    cannot answer whether something is SQL. The list missed MERGE, which PostgreSQL 15 made a
    statement that updates, deletes and inserts: ``MERGE INTO t USING s ON ... WHEN MATCHED THEN
    DELETE`` was accepted and ran both against a relational table.
    """

    @pytest.mark.parametrize(
        "statement",
        [
            "MERGE INTO public.t x USING public.s y ON x.v = y.a WHEN MATCHED THEN DELETE",
            "MERGE INTO t USING s ON t.a = s.a WHEN NOT MATCHED THEN INSERT VALUES (1)",
            "EXPLAIN ANALYZE MERGE INTO t USING s ON t.a = s.a WHEN MATCHED THEN DELETE",
            "EXPLAIN (FORMAT JSON) MERGE INTO t USING s ON t.a = s.a WHEN MATCHED THEN DELETE",
            "EXPLAIN ANALYZE VERBOSE MERGE INTO t USING s ON t.a = s.a WHEN MATCHED THEN DELETE",
        ],
    )
    def test_one_naming_a_table_is_refused(self, statement):
        with pytest.raises(PolicyRefusal):
            check_statement_is_allowed(statement)

    @pytest.mark.parametrize(
        "statement",
        [
            "MERGE (n:P {k: 1}) SET n.v = 2",
            "MERGE p = (a:A)-[:R]->(b:B) RETURN p",
            "CREATE (n:Person {a: 1})",
            "CREATE p = (a)-[r:R]->(b) RETURN p",
            "INSERT (n:Article {t: 'x'})",
            "EXPLAIN MERGE (n:P {k: 1}) SET n.v = 2",
            "EXPLAIN ANALYZE CREATE (n:L {a: 1})",
        ],
    )
    def test_one_taking_a_pattern_is_allowed(self, statement):
        check_statement_is_allowed(statement)


class TestALabelNamedAfterASqlWord:
    """A label is a name, and the server takes a reserved one there.

    ``CREATE (n:Alter {a: 1})`` and ``RETURN n.a AS truncate`` both run, so refusing them as SQL
    turns a legitimate graph query into a confusing refusal.
    """

    @pytest.mark.parametrize(
        "statement",
        [
            "MATCH (n:Select) RETURN n",
            "MATCH (n:Grant) RETURN n",
            "MATCH (n:Vacuum) RETURN n",
            "MATCH (n:Cluster) RETURN n",
            "MATCH ()-[r:GRANT]->() RETURN r",
            "RETURN 1 AS truncate",
            "MATCH (n) RETURN n.a AS cluster",
        ],
    )
    def test_it_is_not_read_as_sql(self, statement):
        check_statement_is_allowed(statement)


class TestWhatDefinitionsAreInFrontOf:
    """``WITH name AS (...)`` names a query before whatever SQL ends with.

    Reading the first word asks about the definitions rather than about what they feed, so every
    statement a CTE can carry had to be named separately -- `MERGE INTO` reached the server that
    way, and `TABLE pg_roles` after it. What the definitions are in front of is the statement.
    """

    @pytest.mark.parametrize(
        "statement",
        [
            "WITH s(a) AS (VALUES (1)) TABLE pg_roles",
            "WITH s(a) AS (VALUES (1)) VALUES (1)",
            "WITH RECURSIVE r(n) AS (VALUES (1)) TABLE pg_roles",
            "WITH a AS (VALUES (1)), b AS (VALUES (2)) TABLE pg_roles",
            "WITH s(a) AS MATERIALIZED (VALUES (1)) TABLE pg_roles",
            "WITH s(a) AS NOT MATERIALIZED (VALUES (1)) TABLE pg_roles",
            "EXPLAIN ANALYZE WITH s(a) AS (VALUES (1)) TABLE pg_roles",
            "WITH t AS (SELECT rolname FROM pg_roles) SELECT rolname FROM t",
            "WITH s(a) AS (VALUES ('x')) MERGE INTO t USING s ON t.v = s.a WHEN MATCHED THEN DELETE",
        ],
    )
    def test_the_statement_they_feed_is_the_one_judged(self, statement):
        with pytest.raises(PolicyRefusal):
            check_statement_is_allowed(statement)

    @pytest.mark.parametrize(
        "statement",
        [
            "WITH 1 AS x RETURN x",
            "MATCH (n) WITH n, count(*) AS c WHERE c > 1 RETURN n",
            "MATCH (n) WITH n AS m RETURN m",
            "MATCH (n) WITH (1 + 2) AS y RETURN y",
        ],
    )
    def test_a_graph_projection_is_not_a_definition(self, statement):
        check_statement_is_allowed(statement)


class TestANameThatWasBlankedAway:
    """Literals and quoted identifiers are blanked before any of this reads the statement.

    So a recogniser keying on an identifier can be evaded by quoting that identifier: the name a
    query definition is given is gone by the time the definitions are read past, and requiring
    one meant they never were. ``WITH "s" AS (VALUES (1)) TABLE pg_roles`` returned every row of
    ``pg_roles`` through a read tool.
    """

    @pytest.mark.parametrize(
        "statement",
        [
            'WITH "s" AS (VALUES (1)) TABLE pg_roles',
            'WITH "s" AS (VALUES (1)) VALUES (1)',
            'WITH "s" AS (VALUES (1)) SELECT 1',
            'WITH "s"(a) AS (VALUES (1)) TABLE pg_roles',
            'WITH RECURSIVE "r"(n) AS (VALUES (1)) TABLE pg_roles',
            'WITH "a" AS (VALUES (1)), "b" AS (VALUES (2)) TABLE pg_roles',
            'WITH a AS (VALUES (1)), "b" AS (VALUES (2)) TABLE pg_roles',
            'WITH "s" AS MATERIALIZED (VALUES (1)) TABLE pg_roles',
            'EXPLAIN ANALYZE WITH "s" AS (VALUES (1)) TABLE pg_roles',
        ],
    )
    def test_a_quoted_definition_is_still_a_definition(self, statement):
        with pytest.raises(PolicyRefusal):
            check_statement_is_allowed(statement)

    @pytest.mark.parametrize(
        "statement",
        [
            'MATCH (n) WITH n AS "sel" RETURN n',
            'MATCH (n:"Memory") RETURN n',
            "MATCH (n) WITH n.name AS nm ORDER BY nm RETURN nm",
            "MATCH (n) WITH (1 + 2) AS y RETURN y",
        ],
    )
    def test_a_graph_projection_is_still_left_alone(self, statement):
        check_statement_is_allowed(statement)


class TestAQuotedPathVariable:
    """The variable naming a path may be quoted, and a quoted name is blanked to spaces.

    So requiring one to be visible refused ``CREATE "p" = (a)-[:R]->(b)``, which the server
    accepts. Same cause as a quoted query definition, in the other direction.
    """

    @pytest.mark.parametrize(
        "statement",
        [
            'CREATE "p" = (a:X)-[:R]->(b:Y) RETURN "p"',
            'MERGE "p" = (a:X)-[:R]->(b:Y) RETURN "p"',
            "CREATE p = (a)-[r:R]->(b) RETURN p",
        ],
    )
    def test_it_still_takes_a_pattern(self, statement):
        check_statement_is_allowed(statement)

    @pytest.mark.parametrize(
        "statement",
        [
            'MERGE INTO "t" USING s ON t.a = s.a WHEN MATCHED THEN DELETE',
            "MERGE INTO t USING s ON t.a = s.a WHEN MATCHED THEN DELETE",
        ],
    )
    def test_and_one_naming_a_table_is_still_refused(self, statement):
        with pytest.raises(PolicyRefusal):
            check_statement_is_allowed(statement)
