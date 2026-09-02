# © Artur Czarnecki. All rights reserved.

"""Unit tests for bounded qualification attempt orchestration."""

from __future__ import annotations

import pytest

from intergrax.core.qualification.functional_qualification_attempts import (
    QualificationAttemptPolicy,
    QualificationPreconditionStatus,
    execute_bounded_attempts,
)

pytestmark = pytest.mark.unit


def test_first_attempt_satisfied_is_authoritative() -> None:
    calls = 0

    def execute_attempt(attempt_index: int) -> tuple[str, QualificationPreconditionStatus, str, str]:
        nonlocal calls
        calls += 1
        return (
            f"result-{attempt_index}",
            QualificationPreconditionStatus.SATISFIED,
            f"task-{attempt_index}",
            f"run-{attempt_index}",
        )

    outcome = execute_bounded_attempts(
        QualificationAttemptPolicy(max_attempts=3),
        execute_attempt,
        summarize_miss=lambda _result: "miss",
    )
    assert outcome.authoritative_result == "result-1"
    assert outcome.authoritative_attempt_index == 1
    assert calls == 1


def test_prerequisite_miss_then_satisfied_on_second_attempt() -> None:
    calls = 0

    def execute_attempt(attempt_index: int) -> tuple[str, QualificationPreconditionStatus, str, str]:
        nonlocal calls
        calls += 1
        status = (
            QualificationPreconditionStatus.NOT_SATISFIED
            if attempt_index == 1
            else QualificationPreconditionStatus.SATISFIED
        )
        return (f"result-{attempt_index}", status, f"task-{attempt_index}", f"run-{attempt_index}")

    outcome = execute_bounded_attempts(
        QualificationAttemptPolicy(max_attempts=3),
        execute_attempt,
        summarize_miss=lambda result: f"miss:{result}",
    )
    assert outcome.authoritative_result == "result-2"
    assert outcome.authoritative_attempt_index == 2
    assert len(outcome.attempt_records) == 2
    assert outcome.attempt_records[0].precondition_status is QualificationPreconditionStatus.NOT_SATISFIED
    assert outcome.attempt_records[0].summary == "miss:result-1"
    assert calls == 2


def test_exhausted_attempts_return_blocked_reason() -> None:
    calls = 0

    def execute_attempt(attempt_index: int) -> tuple[str, QualificationPreconditionStatus, str, str]:
        nonlocal calls
        calls += 1
        return (
            f"result-{attempt_index}",
            QualificationPreconditionStatus.NOT_SATISFIED,
            f"task-{attempt_index}",
            f"run-{attempt_index}",
        )

    outcome = execute_bounded_attempts(
        QualificationAttemptPolicy(max_attempts=3),
        execute_attempt,
        summarize_miss=lambda result: result,
    )
    assert outcome.authoritative_result is None
    assert outcome.exhausted is True
    assert outcome.blocked_reason == "prerequisite_not_reached"
    assert len(outcome.attempt_records) == 3
    assert calls == 3


def test_valid_mismatch_does_not_run_third_attempt() -> None:
    calls = 0

    def execute_attempt(attempt_index: int) -> tuple[str, QualificationPreconditionStatus, str, str]:
        nonlocal calls
        calls += 1
        if attempt_index == 1:
            return ("miss", QualificationPreconditionStatus.NOT_SATISFIED, "task-1", "run-1")
        return ("mismatch", QualificationPreconditionStatus.SATISFIED, "task-2", "run-2")

    outcome = execute_bounded_attempts(
        QualificationAttemptPolicy(max_attempts=3),
        execute_attempt,
        summarize_miss=lambda result: result,
    )
    assert outcome.authoritative_result == "mismatch"
    assert outcome.authoritative_attempt_index == 2
    assert calls == 2
