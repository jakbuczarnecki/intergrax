# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.contracts.execution_lineage import build_execution_lineage_attempt_scope
from intergrax.runtime.execution.lineage.active_lineage import (
    peek_attempt_lineage_degradation,
)
from intergrax.runtime.execution.lineage.persistence import (
    InMemoryExecutionLineagePersistence,
)
from intergrax.runtime.execution.lineage.root_activation import (
    activate_root_execution_lineage,
    deactivate_root_execution_lineage,
)
from intergrax.contracts.execution_identity import mint_task_id, mint_execution_id
from intergrax.runtime.execution.identity_authority import mint_root_execution_identity
from tests.unit.runtime.execution.lineage.lineage_test_helpers import register_v1_attempt


def test_post_open_segment_degradation_binding_on_crash_resume() -> None:
    persistence = InMemoryExecutionLineagePersistence()
    identity = mint_root_execution_identity()
    e1 = mint_execution_id()
    e4 = mint_execution_id()
    scope = build_execution_lineage_attempt_scope(
        tenant_id="tenant-a",
        task_id=mint_task_id(),
        run_id=identity.run_id,
        attempt_id=identity.attempt_id,
    )
    register_v1_attempt(persistence, scope)
    persistence.open_segment(scope, e1)
    _, lineage_token, degradation_token = activate_root_execution_lineage(
        persistence=persistence,
        scope=scope,
        root_execution_id=e4,
        predecessor_root_execution_id=e1,
    )
    assert peek_attempt_lineage_degradation() is not None
    assert peek_attempt_lineage_degradation().degraded is True
    deactivate_root_execution_lineage(lineage_token, degradation_token)
    assert peek_attempt_lineage_degradation() is None


def test_degradation_context_resets_between_root_bindings() -> None:
    persistence = InMemoryExecutionLineagePersistence()
    identity_a = mint_root_execution_identity()
    identity_b = mint_root_execution_identity()
    scope_a = build_execution_lineage_attempt_scope(
        tenant_id="tenant-a",
        task_id=mint_task_id(),
        run_id=identity_a.run_id,
        attempt_id=identity_a.attempt_id,
    )
    scope_b = build_execution_lineage_attempt_scope(
        tenant_id="tenant-a",
        task_id=mint_task_id(),
        run_id=identity_b.run_id,
        attempt_id=identity_b.attempt_id,
    )
    register_v1_attempt(persistence, scope_a)
    persistence.mark_degraded(scope_a, "task-a")
    _, lineage_token_a, degradation_token_a = activate_root_execution_lineage(
        persistence=persistence,
        scope=scope_a,
        root_execution_id=identity_a.execution_id,
    )
    assert peek_attempt_lineage_degradation() is not None
    assert peek_attempt_lineage_degradation().degraded is True
    deactivate_root_execution_lineage(lineage_token_a, degradation_token_a)
    assert peek_attempt_lineage_degradation() is None

    _, lineage_token_b, degradation_token_b = activate_root_execution_lineage(
        persistence=persistence,
        scope=scope_b,
        root_execution_id=identity_b.execution_id,
    )
    assert peek_attempt_lineage_degradation() is not None
    assert peek_attempt_lineage_degradation().degraded is False
    deactivate_root_execution_lineage(lineage_token_b, degradation_token_b)
