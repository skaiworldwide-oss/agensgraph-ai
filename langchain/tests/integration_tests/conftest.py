"""The integration suite.

``collect_ignore`` names the modules of the 0.3 suite. They test the 0.3 API with connection
settings of their own and are not run; each leaves the list when it is rewritten (PLAN.md
Appendix D).
"""

collect_ignore = [
    "test_async.py",
    "test_checkpoint.py",
    "test_composition.py",
    "test_enhanced_schema.py",
    "test_index_config.py",
    "test_indexes.py",
    "test_lifecycle.py",
    "test_retrievers.py",
    "test_shared_state.py",
]
