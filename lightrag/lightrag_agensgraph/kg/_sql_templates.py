# Copyright (c) 2025, SKAI Worldwide Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Relational table DDL for the AgensGraph LightRAG storages."""

# KV storage is opaque dict-by-id, so one generic JSONB table serves every KV
# namespace (full_docs, text_chunks, llm_response_cache, *_chunks, ...). The
# namespace column partitions them; (workspace, namespace, id) is the tenant key.
KV_TABLE = "LIGHTRAG_KV"
KV_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS LIGHTRAG_KV (
    workspace   VARCHAR(255) NOT NULL DEFAULT '',
    namespace   VARCHAR(255) NOT NULL,
    id          TEXT         NOT NULL,
    value       JSONB        NOT NULL DEFAULT '{}'::jsonb,
    create_time TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    update_time TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT LIGHTRAG_KV_PK PRIMARY KEY (workspace, namespace, id)
)
"""

# Document status. Every field LightRAG reads or filters on is a column of its
# own: the pipeline sweeps documents in (created_at, id) order page by page,
# looks them up by content hash and by source file, and counts them by status,
# so those are real columns and real indexes rather than paths into a JSON value.
# created_at is written once and never changed; it is the sort key a sweep in
# progress depends on.
DOC_STATUS_TABLE = "LIGHTRAG_DOC_STATUS"
DOC_STATUS_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS LIGHTRAG_DOC_STATUS (
    workspace       VARCHAR(255) NOT NULL DEFAULT '',
    id              TEXT         NOT NULL,
    status          VARCHAR(64)  NOT NULL,
    content_summary TEXT,
    content_length  INTEGER,
    chunks_count    INTEGER,
    chunks_list     JSONB        NOT NULL DEFAULT '[]'::jsonb,
    file_path       TEXT,
    track_id        VARCHAR(255),
    content_hash    TEXT,
    error_msg       TEXT,
    metadata        JSONB        NOT NULL DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ,
    updated_at      TIMESTAMPTZ,
    CONSTRAINT LIGHTRAG_DOC_STATUS_PK PRIMARY KEY (workspace, id)
)
"""
DOC_STATUS_INDEX_DDL = [
    # The scheduling sweep: one status at a time, in (created_at, id) order, with a
    # row lacking created_at sorting first so it stays reachable.
    "CREATE INDEX IF NOT EXISTS lightrag_doc_status_sweep_idx "
    "ON LIGHTRAG_DOC_STATUS (workspace, status, created_at NULLS FIRST, id)",
    "CREATE INDEX IF NOT EXISTS lightrag_doc_status_ws_hash_idx "
    "ON LIGHTRAG_DOC_STATUS (workspace, content_hash) "
    "WHERE content_hash IS NOT NULL AND content_hash <> ''",
    "CREATE INDEX IF NOT EXISTS lightrag_doc_status_ws_track_idx "
    "ON LIGHTRAG_DOC_STATUS (workspace, track_id)",
    "CREATE INDEX IF NOT EXISTS lightrag_doc_status_ws_path_idx "
    "ON LIGHTRAG_DOC_STATUS (workspace, file_path)",
    # The two orders the document list is paged in.
    "CREATE INDEX IF NOT EXISTS lightrag_doc_status_ws_updated_idx "
    "ON LIGHTRAG_DOC_STATUS (workspace, updated_at)",
    "CREATE INDEX IF NOT EXISTS lightrag_doc_status_ws_created_idx "
    "ON LIGHTRAG_DOC_STATUS (workspace, created_at)",
]

# The previous layout kept the whole record in one `value` JSONB column. A table
# in that shape is brought to this one in place, keeping its rows.
DOC_STATUS_UPGRADE_DDL = [
    "ALTER TABLE LIGHTRAG_DOC_STATUS "
    "ADD COLUMN IF NOT EXISTS content_summary TEXT, "
    "ADD COLUMN IF NOT EXISTS content_length INTEGER, "
    "ADD COLUMN IF NOT EXISTS chunks_count INTEGER, "
    "ADD COLUMN IF NOT EXISTS chunks_list JSONB NOT NULL DEFAULT '[]'::jsonb, "
    "ADD COLUMN IF NOT EXISTS error_msg TEXT, "
    "ADD COLUMN IF NOT EXISTS metadata JSONB NOT NULL DEFAULT '{}'::jsonb, "
    "ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ, "
    "ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ",
    "UPDATE LIGHTRAG_DOC_STATUS SET "
    "content_summary = value->>'content_summary', "
    "content_length = (value->>'content_length')::integer, "
    "chunks_count = (value->>'chunks_count')::integer, "
    "chunks_list = COALESCE(value->'chunks_list', '[]'::jsonb), "
    "error_msg = value->>'error_msg', "
    "metadata = COALESCE(value->'metadata', '{}'::jsonb), "
    "created_at = (value->>'created_at')::timestamptz, "
    "updated_at = (value->>'updated_at')::timestamptz",
    "ALTER TABLE LIGHTRAG_DOC_STATUS DROP COLUMN value",
    "DROP INDEX IF EXISTS lightrag_doc_status_ws_status_idx",
]

# Vector tables, one per LightRAG vector namespace. ``content_vector`` is typed
# VECTOR({dim}) so a search's ``<=>`` matches the HNSW index expression. The
# chunk ids of an entity or a relation are kept as LightRAG hands them over,
# one string joined with its separator.
VECTOR_ENTITY_TABLE = "LIGHTRAG_VDB_ENTITY"
VECTOR_RELATION_TABLE = "LIGHTRAG_VDB_RELATION"
VECTOR_CHUNK_TABLE = "LIGHTRAG_VDB_CHUNKS"

VECTOR_TABLE_DDL = {
    VECTOR_ENTITY_TABLE: """
    CREATE TABLE IF NOT EXISTS LIGHTRAG_VDB_ENTITY (
        workspace      VARCHAR(255) NOT NULL DEFAULT '',
        id             TEXT         NOT NULL,
        entity_name    TEXT,
        content        TEXT,
        content_vector VECTOR({dim}),
        chunk_ids      TEXT,
        file_path      TEXT,
        create_time    TIMESTAMPTZ  NOT NULL DEFAULT now(),
        update_time    TIMESTAMPTZ  NOT NULL DEFAULT now(),
        CONSTRAINT LIGHTRAG_VDB_ENTITY_PK PRIMARY KEY (workspace, id)
    )
    """,
    VECTOR_RELATION_TABLE: """
    CREATE TABLE IF NOT EXISTS LIGHTRAG_VDB_RELATION (
        workspace      VARCHAR(255) NOT NULL DEFAULT '',
        id             TEXT         NOT NULL,
        src_id         TEXT,
        tgt_id         TEXT,
        content        TEXT,
        content_vector VECTOR({dim}),
        chunk_ids      TEXT,
        file_path      TEXT,
        create_time    TIMESTAMPTZ  NOT NULL DEFAULT now(),
        update_time    TIMESTAMPTZ  NOT NULL DEFAULT now(),
        CONSTRAINT LIGHTRAG_VDB_RELATION_PK PRIMARY KEY (workspace, id)
    )
    """,
    VECTOR_CHUNK_TABLE: """
    CREATE TABLE IF NOT EXISTS LIGHTRAG_VDB_CHUNKS (
        workspace         VARCHAR(255) NOT NULL DEFAULT '',
        id                TEXT         NOT NULL,
        full_doc_id       TEXT,
        chunk_order_index INTEGER,
        tokens            INTEGER,
        content           TEXT,
        content_vector    VECTOR({dim}),
        file_path         TEXT,
        create_time       TIMESTAMPTZ  NOT NULL DEFAULT now(),
        update_time       TIMESTAMPTZ  NOT NULL DEFAULT now(),
        CONSTRAINT LIGHTRAG_VDB_CHUNKS_PK PRIMARY KEY (workspace, id)
    )
    """,
}

# The lookups that delete by name: an entity by its name, a relation by either endpoint.
VECTOR_LOOKUP_INDEX_DDL = [
    "CREATE INDEX IF NOT EXISTS lightrag_vdb_entity_name_idx ON LIGHTRAG_VDB_ENTITY (workspace, entity_name)",
    "CREATE INDEX IF NOT EXISTS lightrag_vdb_relation_src_idx ON LIGHTRAG_VDB_RELATION (workspace, src_id)",
    "CREATE INDEX IF NOT EXISTS lightrag_vdb_relation_tgt_idx ON LIGHTRAG_VDB_RELATION (workspace, tgt_id)",
]


def vector_index_name(table: str) -> str:
    return f"{table.lower()}_hnsw"


def vector_index_ddl(table: str, *, m: int, ef_construction: int, if_not_exists: bool = True) -> str:
    """The HNSW index a nearest search runs on."""
    exists = "IF NOT EXISTS " if if_not_exists else ""
    return (
        f"CREATE INDEX {exists}{vector_index_name(table)} ON {table} "
        "USING hnsw (content_vector vector_cosine_ops) "
        f"WITH (m = {int(m)}, ef_construction = {int(ef_construction)})"
    )
