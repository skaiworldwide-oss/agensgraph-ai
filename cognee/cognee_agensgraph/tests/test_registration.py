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

import cognee_agensgraph  # noqa: F401  (registers the adapters)
from cognee.infrastructure.databases.graph.supported_databases import (
    supported_databases as graph_databases,
)
from cognee.infrastructure.databases.vector.supported_databases import (
    supported_databases as vector_databases,
)

from cognee_agensgraph.infrastructure.databases.graph.agensgraph.adapter import (
    AgensgraphAdapter,
)
from cognee_agensgraph.infrastructure.databases.vector.agensgraph.adapter import (
    AgensgraphVectorAdapter,
)


def test_graph_adapter_registered():
    # GRAPH_DATABASE_PROVIDER=agensgraph selects our graph adapter.
    assert graph_databases.get("agensgraph") is AgensgraphAdapter


def test_vector_adapter_registered():
    # VECTOR_DB_PROVIDER=agensgraph selects our vector adapter.
    assert vector_databases.get("agensgraph") is AgensgraphVectorAdapter
