from cognee_agensgraph.tests.tasks.descriptive_metrics.metrics_test_utils import assert_metrics
import asyncio
import cognee_agensgraph


if __name__ == "__main__":
    asyncio.run(assert_metrics(provider="agensgraph", include_optional=False))
    # asyncio.run(assert_metrics(provider="agensgraph", include_optional=True))
