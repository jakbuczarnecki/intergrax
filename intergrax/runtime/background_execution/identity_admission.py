# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Fail-closed admission checks for background execution identity (UE-9A)."""

from __future__ import annotations

from intergrax.contracts.execution_identity import (
    RunId,
    TaskId,
    validate_run_id,
    validate_task_id,
)
from intergrax.runtime.background_execution.bootstrap import BackgroundExecutionIdentity


class BackgroundExecutionIdentityMismatchError(ValueError):
    """Canonical identity fields disagree with resolved background execution identity."""


def _try_canonical_run_id(value: str) -> RunId | None:
    if not value.startswith("run_"):
        return None
    try:
        return validate_run_id(value)
    except (TypeError, ValueError):
        return None


def _try_canonical_task_id(value: str) -> TaskId | None:
    if not value.startswith("task_"):
        return None
    try:
        return validate_task_id(value)
    except (TypeError, ValueError):
        return None


def assert_handler_run_id_matches_identity(
    *,
    handler_run_id: str,
    execution_identity: BackgroundExecutionIdentity,
) -> None:
    canonical = _try_canonical_run_id(handler_run_id)
    if canonical is not None and canonical != execution_identity.run_id:
        raise BackgroundExecutionIdentityMismatchError(
            "handler run_id does not match background execution identity: "
            f"handler={handler_run_id!r} identity={execution_identity.run_id!r}"
        )


def assert_payload_run_id_consistent(
    *,
    payload_run_id: str,
    execution_identity: BackgroundExecutionIdentity,
) -> None:
    canonical = _try_canonical_run_id(payload_run_id)
    if canonical is not None and canonical != execution_identity.run_id:
        raise BackgroundExecutionIdentityMismatchError(
            "payload run_id conflicts with background execution identity: "
            f"payload={payload_run_id!r} identity={execution_identity.run_id!r}"
        )


def assert_payload_task_id_consistent(
    *,
    payload_task_id: str,
    execution_identity: BackgroundExecutionIdentity,
) -> None:
    canonical = _try_canonical_task_id(payload_task_id)
    if canonical is not None and canonical != execution_identity.task_id:
        raise BackgroundExecutionIdentityMismatchError(
            "payload task_id conflicts with background execution identity: "
            f"payload={payload_task_id!r} identity={execution_identity.task_id!r}"
        )
