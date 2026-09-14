# © Artur Czarnecki. All rights reserved.

"""GR-1-R1 — atomic AttemptId + ExecutionId resolution for meaningful side effects."""

from __future__ import annotations

import pytest

from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
    reset_active_execution_identity,
)
from intergrax.contracts.meaningful_side_effect import (
    resolve_meaningful_side_effect_execution_identity,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

TASK_ID = mint_task_id()
RUN_ID = mint_run_id()
ATTEMPT_A1 = mint_attempt_id()
EXECUTION_E1 = mint_execution_id()
ATTEMPT_A2 = mint_attempt_id()
EXECUTION_E2 = mint_execution_id()


def _bind_active(
    *,
    run_id: str = RUN_ID,
    attempt_id: str = ATTEMPT_A1,
    execution_id: str = EXECUTION_E1,
):
    return bind_active_execution_identity(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    )


def test_partial_attempt_only_rejected() -> None:
    token = _bind_active()
    try:
        with pytest.raises(ValueError, match="supplied together"):
            resolve_meaningful_side_effect_execution_identity(
                task_id=TASK_ID,
                run_id=RUN_ID,
                attempt_id=ATTEMPT_A2,
            )
    finally:
        reset_active_execution_identity(token)


def test_partial_execution_only_rejected() -> None:
    token = _bind_active()
    try:
        with pytest.raises(ValueError, match="supplied together"):
            resolve_meaningful_side_effect_execution_identity(
                task_id=TASK_ID,
                run_id=RUN_ID,
                execution_id=EXECUTION_E2,
            )
    finally:
        reset_active_execution_identity(token)


def test_active_only_returns_full_active_identity() -> None:
    token = _bind_active()
    try:
        task_id, run_id, attempt_id, execution_id = (
            resolve_meaningful_side_effect_execution_identity(
                task_id=TASK_ID,
                run_id=RUN_ID,
            )
        )
        assert task_id == TASK_ID
        assert run_id == RUN_ID
        assert attempt_id == ATTEMPT_A1
        assert execution_id == EXECUTION_E1
    finally:
        reset_active_execution_identity(token)


def test_explicit_complete_matching_active_accepted() -> None:
    token = _bind_active()
    try:
        resolved = resolve_meaningful_side_effect_execution_identity(
            task_id=TASK_ID,
            run_id=RUN_ID,
            attempt_id=ATTEMPT_A1,
            execution_id=EXECUTION_E1,
        )
        assert resolved[2] == ATTEMPT_A1
        assert resolved[3] == EXECUTION_E1
    finally:
        reset_active_execution_identity(token)


def test_explicit_complete_cross_attempt_rejected() -> None:
    token = _bind_active()
    try:
        with pytest.raises(ValueError, match="does not match active execution"):
            resolve_meaningful_side_effect_execution_identity(
                task_id=TASK_ID,
                run_id=RUN_ID,
                attempt_id=ATTEMPT_A2,
                execution_id=EXECUTION_E2,
            )
    finally:
        reset_active_execution_identity(token)


def test_hybrid_attempt_override_rejected() -> None:
    token = _bind_active()
    try:
        with pytest.raises(ValueError, match="supplied together"):
            resolve_meaningful_side_effect_execution_identity(
                task_id=TASK_ID,
                run_id=RUN_ID,
                attempt_id=ATTEMPT_A2,
                execution_id=None,
            )
    finally:
        reset_active_execution_identity(token)
