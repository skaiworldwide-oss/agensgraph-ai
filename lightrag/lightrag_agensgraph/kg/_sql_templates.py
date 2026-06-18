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
