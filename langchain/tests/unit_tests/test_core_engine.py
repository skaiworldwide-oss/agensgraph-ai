"""The engine's decision to run a program again, without a server."""

from __future__ import annotations

import psycopg
from agensgraph import RetryPolicy
from agensgraph.retry import TokenBucket

from langchain_agensgraph._core.engine import _again

MET_ANOTHER_WRITER = (
    psycopg.errors.UniqueViolation("another writer made it first"),
    psycopg.errors.SerializationFailure("could not serialize access due to concurrent update"),
    psycopg.errors.DeadlockDetected("deadlock detected"),
)


def test_a_merge_meeting_another_writer_runs_again_without_spending_the_allowance() -> None:
    allowance = TokenBucket()
    policy = RetryPolicy(attempts=3, bucket=allowance)
    for failure in MET_ANOTHER_WRITER * 4:
        assert _again(policy, failure, number=1, merging=True) is not None
        assert _again(policy, failure, number=3, merging=True) is None
    assert allowance.tokens() == allowance.capacity


def test_a_write_that_is_no_merge_is_not_run_again_for_a_duplicate() -> None:
    policy = RetryPolicy(attempts=3, bucket=TokenBucket())
    duplicate = psycopg.errors.UniqueViolation("duplicate key")
    assert _again(policy, duplicate, number=1, merging=False) is None


def test_other_failures_are_the_policys_to_decide_within_the_allowance() -> None:
    allowance = TokenBucket()
    policy = RetryPolicy(attempts=10, bucket=allowance)
    conflict = psycopg.errors.SerializationFailure("could not serialize access")
    retried = [_again(policy, conflict, number=n, merging=False) for n in range(1, 10)]
    assert retried[0] is not None and None in retried
    assert allowance.tokens() < allowance.capacity
