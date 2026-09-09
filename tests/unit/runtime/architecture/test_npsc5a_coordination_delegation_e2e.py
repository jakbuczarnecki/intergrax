# © Artur Czarnecki. All rights reserved.

"""NPSC-5A — end-to-end delegated specialist coordination proof."""

from __future__ import annotations

import pytest

from intergrax.agent_distribution.delegated_subtasks import (
    DelegatedSubtaskDelegate,
)
from intergrax.agent_distribution.multi_agent_coordination import (
    CoordinationDelegation,
    CoordinationId,
    CoordinationRequest,
    MultiAgentCoordinationService,
)
from intergrax.agent_distribution.task_capability_resolution import (
    build_task_capability_resolution_request,
    unresolved_agent_distribution_capability_need,
)
from intergrax.agent_distribution.task_scoped_agents import TaskScopedAgentLeaseId
from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.contracts.execution_identity import (
    ExecutionId,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
    peek_active_parent_execution_id,
    require_active_execution_id,
)
from intergrax.runtime.execution.boundary import (
    ExecutionBoundary,
    ExecutionIdentityBinding,
)
from tests.unit.agent_distribution.test_delegated_subtasks import (
    OcrRequest,
    OcrResult,
    _APP,
    _ENV,
    _OCR_PACKAGE,
    _discovery_candidate,
    admin_test_principal,
    build_delegated_harness,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


class _LineageSpecialist(DelegatedSubtaskDelegate[OcrRequest, OcrResult]):
    def __init__(self) -> None:
        self.child_execution_id: ExecutionId | None = None
        self.child_parent_execution_id: ExecutionId | None = None

    async def execute(self, request: OcrRequest) -> OcrResult:
        self.child_execution_id = require_active_execution_id()
        self.child_parent_execution_id = peek_active_parent_execution_id()
        return OcrResult(text=f"ocr:{request.document_ref}")


@pytest.mark.asyncio
async def test_npsc5a_end_to_end_coordination_delegation_proof() -> None:
    """Parent execution → coordination → delegated subtask → child runner → specialist."""
    specialist = _LineageSpecialist()
    harness = build_delegated_harness(
        candidates=(
            _discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),
        ),
        specialist_delegate=specialist,
    )
    coordination = MultiAgentCoordinationService(
        delegated_subtasks=harness.service,
    )
    task_scope = mint_task_id()
    harness.task_scope_authority.task_scope_id = task_scope
    root = ExecutionIdentityBinding(
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )
    parent_execution_id = root.execution_id

    class RootDelegate:
        async def execute(self, request: OcrRequest) -> OcrResult:
            result = await coordination.coordinate(
                CoordinationRequest(
                    coordination_id=CoordinationId("coordination-e2e"),
                    delegation_id="delegation-e2e",
                    task_scope_id=task_scope,
                    application_id=_APP,
                    application_environment_id=_ENV,
                    lease_id=TaskScopedAgentLeaseId("lease-e2e"),
                    capability_need=unresolved_agent_distribution_capability_need(
                        build_task_capability_resolution_request(task_kind="document.ocr"),
                    ),
                ),
                delegation=CoordinationDelegation(payload=request),
                principal=admin_test_principal(),
            )
            assert result.coordination_id == CoordinationId("coordination-e2e")
            assert result.delegated.selection_decision.outcome.value == "selected"
            assert result.delegated.lease_id == TaskScopedAgentLeaseId("lease-e2e")
            return result.result

    specialist_result = await ExecutionBoundary[OcrRequest, OcrResult](
        RootDelegate(),
        identity=root,
        authority=ParentExecutionAuthority.unrestricted_root(),
    ).execute(OcrRequest(document_ref="e2e-proof"))

    assert specialist_result.text == "ocr:e2e-proof"
    assert specialist.child_execution_id is not None
    assert parent_execution_id is not None
    assert specialist.child_execution_id != parent_execution_id
    assert specialist.child_parent_execution_id == parent_execution_id
