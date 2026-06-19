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

"""Register the AgensGraph graph and vector adapters with Cognee."""

from cognee.infrastructure.databases.graph.use_graph_adapter import use_graph_adapter
from cognee.infrastructure.databases.vector.use_vector_adapter import use_vector_adapter

from cognee_agensgraph.infrastructure.databases.graph.agensgraph.adapter import (
    AgensgraphAdapter,
)
from cognee_agensgraph.infrastructure.databases.vector.agensgraph.adapter import (
    AgensgraphVectorAdapter,
)

use_graph_adapter("agensgraph", AgensgraphAdapter)
use_vector_adapter("agensgraph", AgensgraphVectorAdapter)
