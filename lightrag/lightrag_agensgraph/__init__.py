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

"""Register the AgensGraph storage backends with LightRAG."""

import lightrag.kg

# (storage_kind, class_name, module_path)
_IMPLEMENTATIONS = [
    ("GRAPH_STORAGE", "AgensgraphStorage", "lightrag_agensgraph.kg.agensgraph_impl"),
    ("KV_STORAGE", "AgensgraphKVStorage", "lightrag_agensgraph.kg.agensgraph_kv_impl"),
]

_ENV_REQUIREMENTS = ["AGENSGRAPH_DB", "AGENSGRAPH_USER", "AGENSGRAPH_PASSWORD"]

for _kind, _name, _module in _IMPLEMENTATIONS:
    _impls = lightrag.kg.STORAGE_IMPLEMENTATIONS[_kind]["implementations"]
    if _name not in _impls:
        _impls.append(_name)
    lightrag.kg.STORAGES[_name] = _module
    lightrag.kg.STORAGE_ENV_REQUIREMENTS[_name] = list(_ENV_REQUIREMENTS)
