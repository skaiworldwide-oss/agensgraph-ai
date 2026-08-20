'''
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
'''

import os

import pytest

# Every test here talks to a server, and each module skips itself when it cannot
# reach one. Run with no environment at all, that made the whole suite report
# success: one test passed, ninety-seven skipped, exit code 0. A run that tested
# nothing looked exactly like a run that tested everything.
REQUIRED = ("AGENS_DB", "AGENS_USER", "AGENS_PASSWORD")
ESCAPE = "AGENS_TESTS_MAY_SKIP"


def pytest_configure(config: pytest.Config) -> None:
    if os.environ.get(ESCAPE):
        return
    missing = [name for name in REQUIRED if not os.environ.get(name)]
    if missing:
        raise pytest.UsageError(
            "these tests need a server and "
            + ", ".join(missing)
            + " is not set, so every one of them would skip and the run would "
            "report success. Set them, or set "
            + ESCAPE
            + "=1 to accept a run that tests nothing.\n"
            "  AGENS_HOST=127.0.0.1 AGENS_PORT=5432 AGENS_DB=... "
            "AGENS_USER=... AGENS_PASSWORD=... pytest"
        )
