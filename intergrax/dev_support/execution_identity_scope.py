# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical execution identity helpers for lab, scaffold, and test harness composition."""

from __future__ import annotations

from contextlib import contextmanager
from hashlib import sha256

from intergrax.contracts.execution_identity import RunId


def canonical_run_id_for_tests(run_id: str) -> RunId:
    """Canonical RunId aligned with test runtime state builders."""
    from intergrax.contracts.execution_identity import validate_run_id

    if run_id.startswith("run_") and len(run_id) == 36:
        return validate_run_id(run_id)
    digest = sha256(run_id.encode()).hexdigest()[:32]
    return validate_run_id(f"run_{digest}")


@contextmanager
def canonical_execution_identity_scope(run_id: str):
    """
    Bind canonical active execution identity for tool-loop and agent smoke runs.

    ``run_id`` may be a seed string or canonical ``run_…`` value matching
    ``RuntimeState.run_id`` from test runtime builders.
    """
    from intergrax.contracts.execution_identity import (
        bind_active_execution_identity,
        mint_attempt_id,
        mint_execution_id,
        reset_active_execution_identity,
    )

    canonical_run_id = canonical_run_id_for_tests(run_id)
    token = bind_active_execution_identity(
        run_id=canonical_run_id,
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )
    try:
        yield canonical_run_id
    finally:
        reset_active_execution_identity(token)
