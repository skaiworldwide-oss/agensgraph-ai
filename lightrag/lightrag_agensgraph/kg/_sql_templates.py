"""
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
"""

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

# Doc-status is queried/sorted by status, file_path, content_hash, track_id, so
# those are promoted to indexed columns; the full DocProcessingStatus record is
# kept in `value` JSONB for faithful round-tripping (created_at/updated_at are
# ISO strings, so sorting on value->>field is chronological).
DOC_STATUS_TABLE = "LIGHTRAG_DOC_STATUS"
DOC_STATUS_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS LIGHTRAG_DOC_STATUS (
    workspace    VARCHAR(255) NOT NULL DEFAULT '',
    id           TEXT         NOT NULL,
    status       VARCHAR(64),
    file_path    TEXT,
    content_hash TEXT,
    track_id     VARCHAR(255),
    value        JSONB        NOT NULL DEFAULT '{}'::jsonb,
    create_time  TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    update_time  TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT LIGHTRAG_DOC_STATUS_PK PRIMARY KEY (workspace, id)
)
"""
DOC_STATUS_INDEX_DDL = [
    "CREATE INDEX IF NOT EXISTS lightrag_doc_status_ws_status_idx ON LIGHTRAG_DOC_STATUS (workspace, status)",
    "CREATE INDEX IF NOT EXISTS lightrag_doc_status_ws_track_idx ON LIGHTRAG_DOC_STATUS (workspace, track_id)",
    "CREATE INDEX IF NOT EXISTS lightrag_doc_status_ws_path_idx ON LIGHTRAG_DOC_STATUS (workspace, file_path)",
    "CREATE INDEX IF NOT EXISTS lightrag_doc_status_ws_hash_idx ON LIGHTRAG_DOC_STATUS (workspace, content_hash)",
]
