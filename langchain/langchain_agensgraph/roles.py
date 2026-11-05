"""The roles this package runs as, and the grants each needs on a graph.

Three roles, so that model-written Cypher can never do what the application can:

``app_rw``
    The application. Reads and writes the graph and its companion schema ``"<graph>_store"``,
    where the package keeps vectors, text, LangGraph state and chat history. Owns nothing.
``llm_ro``
    Model-written Cypher (``AgensGraph.query_untrusted``). Reads the graph's labels and nothing
    in the companion schema, is not ``app_rw`` and not a member of it, and has limits of its own.
    On 2.14 and 2.15 the server does not check a Cypher write against privileges or a read-only
    transaction, so these grants are the boundary there: the role must hold no write at all.
``llm_rw``
    Model-written Cypher allowed to write (``allow_writes=True``): the graph's labels, no DDL,
    nothing in the companion schema.

The objects themselves are made by the owner, the role that runs setup. Run :func:`render_roles`
once per cluster and :func:`render_grants` once per graph, as the owner, after the graph and its
companion schema exist::

    from langchain_agensgraph.roles import render_grants

    print(render_grants("kg"))  # and run what it prints as the owner
"""

from __future__ import annotations

from ._core.names import checked, companion_schema
from ._core.names import quoted as ident
from ._core.names import quoted_literal as literal

__all__ = ["render_grants", "render_roles"]


def render_roles(
    *,
    app_role: str = "app_rw",
    llm_role: str = "llm_ro",
    writer_role: str | None = "llm_rw",
    password: str | None = None,
    llm_connection_limit: int = 8,
    llm_statement_timeout: str = "10s",
) -> str:
    """``CREATE ROLE`` statements for the three roles, with limits on the model's roles.

    Without ``password`` the roles can log in but have no password, which suits a cluster that
    authenticates some other way; otherwise set one with ``ALTER ROLE ... PASSWORD``.
    """
    login = "LOGIN" + (f" PASSWORD {literal(password)}" if password is not None else "")
    lines = [f"CREATE ROLE {ident(checked(app_role, 'role'))} {login};"]
    model_roles = [llm_role] + ([writer_role] if writer_role else [])
    for role in model_roles:
        name = ident(checked(role, "role"))
        lines.append(f"CREATE ROLE {name} {login} CONNECTION LIMIT {int(llm_connection_limit)};")
        for setting, value in (
            ("statement_timeout", llm_statement_timeout),
            ("idle_in_transaction_session_timeout", llm_statement_timeout),
            ("work_mem", "16MB"),
            ("temp_file_limit", "256MB"),
        ):
            lines.append(f"ALTER ROLE {name} SET {setting} = {literal(value)};")
    return "\n".join(lines) + "\n"


def render_grants(
    graph_name: str,
    *,
    app_role: str = "app_rw",
    llm_role: str = "llm_ro",
    writer_role: str | None = None,
    owner: str | None = None,
) -> str:
    """The grants each role needs on one graph and its companion schema, as SQL.

    The companion schema is made if it does not exist, so the grants can be given as soon as the
    graph is. ``owner`` is the role that makes the graph's labels and tables, for the default
    privileges that extend these grants to what it makes later; the role running the statements
    otherwise.
    """
    graph = ident(checked(graph_name, "graph name"))
    store = ident(companion_schema(graph_name))
    app = ident(checked(app_role, "role"))
    llm = ident(checked(llm_role, "role"))
    by = f"FOR ROLE {ident(checked(owner, 'role'))} " if owner else ""
    dml = "SELECT, INSERT, UPDATE, DELETE"
    lines = [
        f"-- {graph} and its companion schema {store}.",
        f"CREATE SCHEMA IF NOT EXISTS {store};",
        f"GRANT USAGE ON SCHEMA {graph}, {store} TO {app};",
        f"GRANT {dml} ON ALL TABLES IN SCHEMA {graph} TO {app};",
        f"GRANT {dml}, TRUNCATE ON ALL TABLES IN SCHEMA {store} TO {app};",
        f"GRANT USAGE, SELECT, UPDATE ON ALL SEQUENCES IN SCHEMA {graph}, {store} TO {app};",
        f"ALTER DEFAULT PRIVILEGES {by}IN SCHEMA {graph} GRANT {dml} ON TABLES TO {app};",
        f"ALTER DEFAULT PRIVILEGES {by}IN SCHEMA {store} GRANT {dml}, TRUNCATE ON TABLES TO {app};",
        f"ALTER DEFAULT PRIVILEGES {by}IN SCHEMA {graph}, {store} "
        f"GRANT USAGE, SELECT, UPDATE ON SEQUENCES TO {app};",
        f"GRANT USAGE ON SCHEMA {graph} TO {llm};",
        f"GRANT SELECT ON ALL TABLES IN SCHEMA {graph} TO {llm};",
        f"ALTER DEFAULT PRIVILEGES {by}IN SCHEMA {graph} GRANT SELECT ON TABLES TO {llm};",
    ]
    if writer_role:
        writer = ident(checked(writer_role, "role"))
        lines += [
            f"GRANT USAGE ON SCHEMA {graph} TO {writer};",
            f"GRANT {dml} ON ALL TABLES IN SCHEMA {graph} TO {writer};",
            f"GRANT USAGE, SELECT, UPDATE ON ALL SEQUENCES IN SCHEMA {graph} TO {writer};",
            f"ALTER DEFAULT PRIVILEGES {by}IN SCHEMA {graph} GRANT {dml} ON TABLES TO {writer};",
            f"ALTER DEFAULT PRIVILEGES {by}IN SCHEMA {graph} "
            f"GRANT USAGE, SELECT, UPDATE ON SEQUENCES TO {writer};",
        ]
    return "\n".join(lines) + "\n"
