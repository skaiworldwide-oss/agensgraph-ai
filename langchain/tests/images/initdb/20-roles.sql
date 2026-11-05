-- The package's runtime roles (PLAN.md §2.4), created once at initdb. Per-graph grants come from
-- langchain_agensgraph.roles.render_grants(graph_name) when a test creates its graph.
--   app_rw  the application: DML on the graph and its companion schema
--   llm_ro  model-written Cypher, read only: not a member of app_rw, never superuser
--   llm_rw  model-written Cypher with allow_writes=True: same checks, owns nothing
CREATE ROLE app_rw LOGIN PASSWORD 'agens';
CREATE ROLE llm_ro LOGIN PASSWORD 'agens' CONNECTION LIMIT 8;
CREATE ROLE llm_rw LOGIN PASSWORD 'agens' CONNECTION LIMIT 4;
ALTER ROLE llm_ro SET statement_timeout = '10s';
ALTER ROLE llm_ro SET idle_in_transaction_session_timeout = '10s';
ALTER ROLE llm_ro SET work_mem = '16MB';
ALTER ROLE llm_ro SET temp_file_limit = '256MB';
ALTER ROLE llm_rw SET statement_timeout = '10s';
ALTER ROLE llm_rw SET idle_in_transaction_session_timeout = '10s';
ALTER ROLE llm_rw SET work_mem = '16MB';
ALTER ROLE llm_rw SET temp_file_limit = '256MB';
