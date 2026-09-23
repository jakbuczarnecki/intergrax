# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.contracts.execution.suspended_operation.authority_scope import (
    SuspendedOperationAuthorityScope,
)
from intergrax.contracts.execution.suspended_operation.codec import (
    SerializedSuspendedOperationEnvelope,
    SuspendedOperationKind,
)
from intergrax.contracts.execution.suspended_operation.descriptor import (
    SuspendedExecutionOperationDescriptor,
    SuspendedOperationMaterializationState,
)
from intergrax.contracts.execution.suspended_operation.authority_scope_compat import (
    CANONICAL_ORCHESTRATION_TOOL_MSE_OPERATION_ID,
)
from intergrax.contracts.execution.suspended_operation.entity_id import (
    mint_suspended_operation_id,
)
from intergrax.contracts.execution_identity import AttemptId, ExecutionId, RunId, TaskId

pytestmark = pytest.mark.unit


def _legacy_payload(**overrides: object) -> dict[str, object]:
    identity = {
        "task_id": str(TaskId("task_" + "a" * 32)),
        "run_id": str(RunId("run_" + "b" * 32)),
        "attempt_id": str(AttemptId("attempt_" + "c" * 32)),
        "execution_id": str(ExecutionId("exec_" + "d" * 32)),
    }
    base: dict[str, object] = {
        "schema_version": "suspended_execution_operation_descriptor.v1",
        "suspended_operation_id": mint_suspended_operation_id(),
        "operation_kind": SuspendedOperationKind.EXECUTION_BOUND_CATALOG_TOOL,
        "identity": identity,
        "continuation_id": "gcr_legacy",
        "invocation_scope_id": "dhr_legacyscope",
        "materialization_state": SuspendedOperationMaterializationState.PREPARED,
        "materialization_revision": 0,
        "payload_digest": "sha256:" + ("0" * 64),
        "payload": {
            "operation_kind": SuspendedOperationKind.EXECUTION_BOUND_CATALOG_TOOL,
            "payload_schema_version": "execution_bound_catalog_tool_payload.v1",
            "canonical_json": "{}",
        },
    }
    base.update(overrides)
    return base


def test_legacy_dhr_without_authority_scope_loads_declarative() -> None:
    loaded = SuspendedExecutionOperationDescriptor.model_validate(_legacy_payload())
    assert (
        loaded.authority_scope
        is SuspendedOperationAuthorityScope.DECLARATIVE_GOVERNANCE
    )
    assert loaded.invocation_scope_id == "dhr_legacyscope"


def test_legacy_mse_with_explicit_authority_scope() -> None:
    loaded = SuspendedExecutionOperationDescriptor.model_validate(
        _legacy_payload(
            invocation_scope_id=CANONICAL_ORCHESTRATION_TOOL_MSE_OPERATION_ID,
            authority_scope=SuspendedOperationAuthorityScope.MEANINGFUL_SIDE_EFFECT,
        ),
    )
    assert (
        loaded.authority_scope
        is SuspendedOperationAuthorityScope.MEANINGFUL_SIDE_EFFECT
    )


def test_unknown_legacy_scope_without_authority_rejected() -> None:
    with pytest.raises(ValueError, match="unknown invocation scope"):
        SuspendedExecutionOperationDescriptor.model_validate(
            _legacy_payload(invocation_scope_id="mystery_scope"),
        )


def test_legacy_envelope_round_trip_preserves_identity() -> None:
    raw = _legacy_payload()
    loaded = SuspendedExecutionOperationDescriptor.model_validate(raw)
    envelope = SerializedSuspendedOperationEnvelope.model_validate(raw["payload"])
    assert loaded.payload == envelope
